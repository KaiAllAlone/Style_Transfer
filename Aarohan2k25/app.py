from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import numpy as np
import tempfile
import os

app = FastAPI()

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    # Create a temp file for uploaded image
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    contents = await file.read()
    input_temp.write(contents)
    input_temp.close()

    # Read image using OpenCV
    img = cv2.imread(input_temp.name)

    # Example operation — convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create another temp file for output
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(output_temp.name, gray)

    # Return the processed image
    return FileResponse(output_temp.name, media_type="image/png")
