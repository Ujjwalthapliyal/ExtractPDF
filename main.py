from fastapi import FastAPI, UploadFile, File, HTTPException
from pdf2image import convert_from_bytes
import pytesseract
import pdfplumber
import io
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=2)

# Tesseract config: OEM 3 = default LSTM engine, PSM 6 = assume a uniform block of text
# (change PSM to 3 if your PDFs have complex multi-column layouts)
TESSERACT_CONFIG = "--oem 3 --psm 6"


def get_per_page_text_layer(contents: bytes) -> dict:
    """Extract embedded text per page. Returns {page_index: text_or_empty_string}."""
    page_texts = {}
    try:
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                page_texts[i] = text.strip()
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
    return page_texts


def ocr_single_page(img) -> str:
    """OCR one page image with tuned config."""
    return pytesseract.image_to_string(img, config=TESSERACT_CONFIG)


def rasterize_pages(contents: bytes, page_numbers: list[int], dpi: int = 150):
    """Rasterize only the specific pages that need OCR (1-indexed for pdf2image)."""
    if not page_numbers:
        return {}
    first, last = min(page_numbers) + 1, max(page_numbers) + 1
    images = convert_from_bytes(
        contents,
        dpi=dpi,
        first_page=first,
        last_page=last,
        thread_count=2,  # parallel page rendering inside poppler itself
    )
    # map back to actual page indices (only keep ones we actually need)
    result = {}
    offset = first - 1
    for idx, img in enumerate(images):
        page_idx = offset + idx
        if page_idx in page_numbers:
            result[page_idx] = img
    return result


def process_pdf(contents: bytes) -> tuple[str, str]:
    """
    Hybrid extraction: use embedded text per page where available,
    OCR only the pages that lack a usable text layer.
    Returns (full_text, method_summary).
    """
    page_texts = get_per_page_text_layer(contents)

    if not page_texts:
        # pdfplumber couldn't even open it — force full OCR fallback
        images = convert_from_bytes(contents, dpi=150, thread_count=2)
        with ThreadPoolExecutor(max_workers=min(4, len(images) or 1)) as pool:
            results = list(pool.map(ocr_single_page, images))
        return "\n".join(results), "ocr_full_fallback"

    # Identify which pages need OCR (empty or too-short text layer)
    MIN_CHARS = 20
    ocr_needed = [i for i, t in page_texts.items() if len(t) < MIN_CHARS]

    ocr_results = {}
    if ocr_needed:
        page_images = rasterize_pages(contents, ocr_needed, dpi=150)
        with ThreadPoolExecutor(max_workers=min(4, len(page_images) or 1)) as pool:
            futures = {i: pool.submit(ocr_single_page, img) for i, img in page_images.items()}
            for i, fut in futures.items():
                ocr_results[i] = fut.result()

    # Stitch together in correct page order
    total_pages = len(page_texts)
    full_text = []
    for i in range(total_pages):
        if i in ocr_results:
            full_text.append(ocr_results[i])
        else:
            full_text.append(page_texts[i])

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
