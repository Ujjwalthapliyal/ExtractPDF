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

# --- ADD THIS ---
@app.get("/health")
async def health():
    return {"status": "ok"}
# ----------------

TESSERACT_CONFIG = "--oem 3 --psm 6"

# ... rest of your existing code (get_per_page_text_layer, ocr_single_page, etc.) stays as-is
