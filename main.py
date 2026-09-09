from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_bytes
import pytesseract

app = FastAPI()

@app.post("/ocr")
async def extract_pdf_text(file: UploadFile = File(...)):
    contents = await file.read()
    # Convert PDF pages into PIL image buffers
    images = convert_from_bytes(contents)
    
    extracted_text = ""
    for img in images:
        extracted_text += pytesseract.image_to_string(img) + "\n"
        
    return {"text": extracted_text}
 
