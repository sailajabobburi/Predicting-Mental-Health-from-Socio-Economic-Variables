from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn
import shutil
import src.main as main_module  # renamed to avoid conflict with main function
from typing import List

app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="src/templates")

@app.post("/upload/")
async def process_data(request: Request, file: UploadFile = File(...), model_type: str = Form(...), target_variable: List[str] = Form(...)):
    # Save the uploaded file temporarily (you can also process it directly from memory)
    with open("temp_data.csv", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Assuming 'main' function is already imported and adapted for this purpose
    features, accuracies = main_module.main("temp_data.csv", model_type, target_variable)
    features = features.to_dict(orient="records")
    accuracies = accuracies.to_dict(orient="records")

    return templates.TemplateResponse("features_table.html", {
        "request": request,
        "model_type": model_type,
        "features": features,
        "accuracies": accuracies
    })


@app.get("/")
async def main_page(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.get("/about")
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/faq")
async def faq_page(request: Request):
    return templates.TemplateResponse("faq.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
