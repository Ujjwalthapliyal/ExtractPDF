from fastapi import FastAPI, UploadFile, File, HTTPException
import fitz
import pytesseract
from PIL import Image
import gc
import logging
import shutil
import uuid
import sqlite3
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
from enum import Enum

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=1)

TESSERACT_CONFIG = "--oem 3 --psm 6"
ZOOM = 1.4
MAX_SIZE_MB = 50
MAX_PAGES = 100

DB_PATH = "jobs.db"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


# ---------- Job storage (SQLite instead of in-memory dict) ----------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            result TEXT,
            error TEXT
        )
    """)
    conn.commit()
    conn.close()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def create_job(job_id: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, result, error) VALUES (?, ?, ?, ?)",
            (job_id, JobStatus.PENDING.value, None, None),
        )
        conn.commit()


def update_job(job_id: str, status: str, result: str = None, error: str = None):
    with get_db() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, result = ?, error = ? WHERE job_id = ?",
            (status, result, error, job_id),
        )
        conn.commit()


def get_job(job_id: str):
    with get_db() as conn:
        cur = conn.execute(
            "SELECT status, result, error FROM jobs WHERE job_id = ?", (job_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"status": row[0], "result": row[1], "error": row[2]}


# ---------- Startup ----------

@app.on_event("startup")
async def startup_event():
    init_db()
    if shutil.which("tesseract") is None:
        logger.error("Tesseract binary not found on PATH. OCR requests will fail.")
    else:
        logger.info(f"Tesseract found: {pytesseract.get_tesseract_version()}")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- Core OCR logic ----------

def process_pdf(contents: bytes) -> str:
    doc = fitz.open(stream=contents, filetype="pdf")
    try:
        if len(doc) > MAX_PAGES:
            raise ValueError(f"PDF exceeds {MAX_PAGES} page limit ({len(doc)} pages)")

        extracted_text = []
        mat = fitz.Matrix(ZOOM, ZOOM)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if len(text) > 30:
                extracted_text.append(text)
                continue

            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            try:
                ocr_text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
            except pytesseract.TesseractNotFoundError:
                raise RuntimeError("Tesseract is not installed or not on PATH on this server")

            extracted_text.append(ocr_text)
            del pix, img

        return "\n\n".join(extracted_text)
    finally:
        doc.close()
        gc.collect()


def run_job(job_id: str, contents: bytes):
    update_job(job_id, JobStatus.PROCESSING.value)
    try:
        text = process_pdf(contents)
        update_job(job_id, JobStatus.DONE.value, result=text)
    except ValueError as e:
        update_job(job_id, JobStatus.FAILED.value, error=str(e))
    except Exception as e:
        logger.error(f"OCR job {job_id} failed: {e}")
        update_job(job_id, JobStatus.FAILED.value, error=f"OCR processing failed: {str(e)}")


# ---------- Routes ----------

@app.post("/ocr")
async def submit_ocr(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB); limit is {MAX_SIZE_MB} MB",
        )

    job_id = str(uuid.uuid4())
    create_job(job_id)

    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, run_job, job_id, contents)

    return {"job_id": job_id, "status": JobStatus.PENDING}


@app.get("/ocr/{job_id}")
async def get_ocr_result(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == JobStatus.FAILED.value:
        raise HTTPException(
            status_code=422 if "limit" in (job["error"] or "") else 500,
            detail=job["error"],
        )

    return {"status": job["status"], "text": job.get("result")}
