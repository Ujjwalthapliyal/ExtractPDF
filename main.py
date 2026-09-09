from fastapi import FastAPI, UploadFile, File, HTTPException
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import gc
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=1)

TESSERACT_CONFIG = "--oem 3 --psm 6"
ZOOM = 1.4  # ~100 DPI scale factor (1.4 * 72 DPI), keeps RAM extremely low

MAX_SIZE_MB = 20
MAX_PAGES = 50


@app.on_event("startup")
async def check_tesseract_installed():
    if shutil.which("tesseract") is None:
        logger.error("Tesseract binary not found on PATH. OCR requests will fail.")
    else:
        logger.info(f"Tesseract found: {pytesseract.get_tesseract_version()}")


@app.get("/health")
async def health():
    return {"status": "ok"}


def process_pdf(contents: bytes) -> str:
    doc = fitz.open(stream=contents, filetype="pdf")
    try:
        if len(doc) > MAX_PAGES:
            raise ValueError(f"PDF exceeds {MAX_PAGES} page limit ({len(doc)} pages)")

        extracted_text = []
        mat = fitz.Matrix(ZOOM, ZOOM)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # 1. Fast-track: Check if page already has digital text (0.01s execution)
            text = page.get_text().strip()
            if len(text) > 30:
                extracted_text.append(text)
                continue

            # 2. Fallback to OCR only if the page is a scanned image
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            try:
                ocr_text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
            except pytesseract.TesseractNotFoundError:
                raise RuntimeError(
                    "Tesseract is not installed or not on PATH on this server"
                )

            extracted_text.append(ocr_text)

            # Release image memory immediately per page
            del pix, img

        return "\n\n".join(extracted_text)
    finally:
        doc.close()
        gc.collect()


@app.post("/ocr")
async def extract_pdf_text(file: UploadFile = File(...)):
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

    loop = asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(executor, process_pdf, contents)
    except ValueError as e:
        # Bad/oversized PDF - client error
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Server misconfiguration (e.g. tesseract missing)
        logger.error(f"OCR configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

    return {"text": text}
