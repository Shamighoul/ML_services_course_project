import base64
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ml.ml_model import load_model, preprocess

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


# create a route
@app.get("/")
def index():
    return {"text": "Sign Language Digits Classifier project"}


# Register the function to run during startup
@app.on_event("startup")
def startup_event():
    global model
    model = load_model(model)


@app.get("/predict")
def predict_digit(image: str):
    image = preprocess(image)
    model_pred = model.predict(image, batch_size=32, verbose=0)
    print(model_pred)
    image_base64 = str(base64.b64encode(image).decode("utf-8"))

    response = PredictionResponse(
        image_base64=image_base64,
        label=int(model_pred.argmax()),
        score=float(model_pred.max()),
    )

    return response


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

    image = preprocess(contents)
    print(image.shape)
    result = model.predict(image)

    try:

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
