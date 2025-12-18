import base64
from typing import Optional

import zipfile
import tempfile
import os
from PIL import Image
import json

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ml.ml_model import load_model, preprocess, calc_metrics

model = 1

app = FastAPI(
    title="ML Prediction Service",
    description="Сервис для обработки изображений ML-моделью",
    version="1.0.0",
)


class PredictionResponse(BaseModel):
    image_base64: str
    label: int
    score: float

class MetricsResponse(BaseModel):
    scores: dict


# create a route
@app.get("/")
def index():
    return {"text": "Sign Language Digits Classifier project"}


# Register the function to run during startup
@app.on_event("startup")
def startup_event():
    global model
    model = load_model(model)


@app.post("/forward")
async def forward(
    image: UploadFile = File(...),
    additional_param: Optional[str] = Header(None, alias="X-Additional-Param"),
):
    # Проверка формата
    if not image.content_type.startswith("image"):

        raise HTTPException(
            status_code=400, detail="Bad Request, you need get image file"
        )

        # Чтение изображения
    contents = await image.read()
    try:
        image = preprocess(contents)
        result = model.predict(image)

        # Конвертация в base64
        image_base64 = base64.b64encode(contents).decode("utf-8")

        return PredictionResponse(
            image_base64=image_base64, label=result.argmax(), score=result.max()
        )

    except Exception:
        raise HTTPException(
            status_code=403, detail="Модель не смогла обработать данные"
        )


@app.get("/forward-form", response_class=HTMLResponse)
async def forward_form():
    with open("app/temp/forward.html", "r") as f:
        return f.read()

@app.post("/forward_batch")
async def forward_batch(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    
    zip_path = os.path.join(temp_dir, "data.zip")
    with open(zip_path, "wb") as f:
        f.write(await file.read())
    
    results = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(temp_dir)
        
        for item in z.namelist():
            file_path = os.path.join(temp_dir, item)
            if os.path.isfile(file_path):
                ext = os.path.splitext(item)[1].lower()
                
                if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image = Image.open(file_path)
                    if image is not None:
                        try:
                            print(image)
                            image = preprocess(image)    
                            pred = model.predict(image)
                            image_base64 = base64.b64encode(image).decode("utf-8")

                            results.append({
                                "filename": item,
                                "prediction": PredictionResponse(
                                image_base64=image_base64, label=pred.argmax(), score=pred.max()
                                )  
                            })
                        except Exception:
                            raise HTTPException(
                                status_code=403, detail="Модель не смогла обработать данные"
                            )

    for item in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, item))
    os.rmdir(temp_dir)

    return results

@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    
    zip_path = os.path.join(temp_dir, "data.zip")
    with open(zip_path, "wb") as f:
        f.write(await file.read())
    
    results = []
    ground_truths = []
    y_pred = []
    y_true = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(temp_dir)
        
        # Ищем metadata.json
        metadata_path = os.path.join(temp_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                ground_truths = metadata.get("ground_truth", [])
        ground_truths_copy = ground_truths.copy()
        # Обрабатываем файлы
        for item in z.namelist():
            file_path = os.path.join(temp_dir, item)
            if os.path.isfile(file_path):
                ext = os.path.splitext(item)[1].lower()
                
                if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image = Image.open(file_path)
                    print(file_path)
                    if image is not None:
                        try:
                            image = preprocess(image)    
                            pred = model.predict(image)
                            y_pred.append(pred)
                            for file in ground_truths_copy: 
                                if file["file"] == item:
                                    y_true.append(file["label"])
                                    ground_truths_copy.remove(file)

                            image_base64 = base64.b64encode(image).decode("utf-8")

                            results.append({
                                "filename": item,
                                "prediction": PredictionResponse(
                                image_base64=image_base64, label=pred.argmax(), score=pred.max()
                            )
                            })
                        except Exception:
                            raise HTTPException(
                                status_code=403, detail="Модель не смогла обработать данные"
                            )

    for item in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, item))
    os.rmdir(temp_dir)

    return MetricsResponse(scores = calc_metrics(y_true, y_pred))