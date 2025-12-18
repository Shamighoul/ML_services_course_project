import base64
from PIL import Image

import numpy as np
import pytest

from app.app import PredictionResponse
from ml.ml_model import load_model, preprocess


def test_load_model():
    model = load_model(1)
    assert model is not None


def load_test_images():
    test_images = np.load("tests/test_images.npy")
    image_list = [(image, i) for i, image in enumerate(test_images)]

    return image_list


@pytest.fixture(scope="function")
def model():
    return load_model("test")


@pytest.mark.parametrize("image, expected_label", load_test_images())
def test_digit(model, image, expected_label: int):
    print(image)
    image = Image.fromarray(image.astype(np.float32))
    print(image)

    image = preprocess(image)
    print(image)

    model_pred = model.predict(image)
    print(model_pred)

    image_base64 = str(base64.b64encode(image).decode("utf-8"))

    result = PredictionResponse(
        image_base64=image_base64,
        label=int(model_pred.argmax()),
        score=float(model_pred.max()),
    )

    assert isinstance(result, PredictionResponse)
    assert result.label == expected_label
