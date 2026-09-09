from fastapi import FastAPI, UploadFile, File, HTTPException
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
import pytesseract
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=1)

TESSERACT_CONFIG = "--oem 3 --psm 6"
DPI = 100  # lower DPI = less memory per page


@app.get("/health")
async def health():
    return {"status": "ok"}


def process_pdf(contents: bytes) -> str:
    # Get page count without rendering anything yet (cheap, low memory)
    info = pdfinfo_from_bytes(contents)
    total_pages = info["Pages"]

    extracted_text = ""
    for page_num in range(1, total_pages + 1):
        # Render ONLY this one page
        images = convert_from_bytes(
            contents,
            dpi=DPI,
            first_page=page_num,
            last_page=page_num,
        )
        img = images[0]
        text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
        extracted_text += text + "\n"

        # explicitly release memory before moving to next page
        del img
        del images
        gc.collect()

    return extracted_text


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
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

    return {"text": text}
