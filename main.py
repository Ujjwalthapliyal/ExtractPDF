from fastapi import FastAPI, UploadFile, File, HTTPException
from pdf2image import convert_from_bytes
import pytesseract
import pdfplumber
import io
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=1)  # single worker to limit memory use

TESSERACT_CONFIG = "--oem 3 --psm 6"
DPI = 100  # lowered from 150 to reduce memory per page
MIN_CHARS = 20


@app.get("/health")
async def health():
    return {"status": "ok"}


def get_per_page_text_layer(contents: bytes) -> dict:
    """Extract embedded text per page (fast, low memory)."""
    page_texts = {}
    try:
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                page_texts[i] = text.strip()
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
    return page_texts


def ocr_pages_sequentially(contents: bytes, page_numbers: list) -> dict:
    """
    Process ONE page at a time: rasterize -> OCR -> discard image -> next page.
    This keeps memory usage flat instead of growing with page count.
    """
    results = {}
    for page_idx in sorted(page_numbers):
        try:
            # render just this single page (1-indexed for pdf2image)
            images = convert_from_bytes(
                contents,
                dpi=DPI,
                first_page=page_idx + 1,
                last_page=page_idx + 1,
            )
            if images:
                img = images[0]
                text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
                results[page_idx] = text
                # explicitly free memory before moving to next page
                del img
                del images
                gc.collect()
        except Exception as e:
            logger.error(f"OCR failed on page {page_idx}: {e}")
            results[page_idx] = ""
    return results


def process_pdf(contents: bytes) -> tuple:
    page_texts = get_per_page_text_layer(contents)

    if not page_texts:
        # pdfplumber couldn't open it at all — fall back to full sequential OCR
        try:
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                total_pages = len(pdf.pages)
        except Exception:
            total_pages = 1  # best guess fallback
        ocr_results = ocr_pages_sequentially(contents, list(range(total_pages)))
        full_text = "\n".join(ocr_results[i] for i in sorted(ocr_results))
        return full_text, "ocr_full_fallback"

    ocr_needed = [i for i, t in page_texts.items() if len(t) < MIN_CHARS]
    ocr_results = ocr_pages_sequentially(contents, ocr_needed) if ocr_needed else {}

    total_pages = len(page_texts)
    full_text = []
    for i in range(total_pages):
        full_text.append(ocr_results.get(i, page_texts.get(i, "")))

    method = "mixed" if ocr_needed and len(ocr_needed) < total_pages else (
        "ocr_only" if ocr_needed else "text_layer_only"
    )
    return "\n".join(full_text), method


@app.post("/ocr")
async def extract_pdf_text(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    loop = asyncio.get_event_loop()
    try:
        text, method = await loop.run_in_executor(executor, process_pdf, contents)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from this PDF")

    return {"text": text, "method": method}
