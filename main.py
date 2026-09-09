from fastapi import FastAPI, UploadFile, File, HTTPException
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=1)

TESSERACT_CONFIG = "--oem 3 --psm 6"
ZOOM = 1.4  # ~100 DPI scale factor (1.4 * 72 DPI), keeps RAM extremely low


@app.get("/health")
async def health():
    return {"status": "ok"}


def process_pdf(contents: bytes) -> str:
    # Open document once in C-bindings memory (no re-parsing per page)
    doc = fitz.open(stream=contents, filetype="pdf")
    extracted_text = []

    # Resolution matrix for lower memory rendering
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

        ocr_text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
        extracted_text.append(ocr_text)

        # Release image memory immediately per page
        del pix, img
        gc.collect()

    doc.close()
    return "\n\n".join(extracted_text)


@app.post("/ocr")
async def extract_pdf_text(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(executor, process_pdf, contents)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"OCR processing failed: {str(e)}"
        )

    return {"text": text}
