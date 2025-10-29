from fastapi import FastAPI
from fastapi.responses import HTMLResponse,FileResponse 
from pathlib import Path
app= FastAPI()

@app.get('/')
def show():
    file_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(file_path)
