import io

import numpy as np
import yaml
from keras import models
from PIL import Image


class Config:
    config_path = "ml/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]

    image_size = model_cfg["image_size"]
    img_channel = model_cfg["img_channel"]
    num_classes = model_cfg["num_classes"]
    input_shape = (image_size, image_size, img_channel)
    target_size = (image_size, image_size)

    path = model_cfg["path"]
    name = model_cfg["name"]
    model_path = path + name + ".keras"
    prod_model = model_cfg["prod"]


cfg = Config()


def preprocess(image, cfg=Config()):
    """Предобработка изображения"""

    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    if not isinstance(image, np.ndarray):
        image = image.resize(cfg.target_size, Image.Resampling.LANCZOS)
        if image.mode != "L":
            image = image.convert("L")
    image = np.array(image)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    if image.max() > 1:
        image = image / 255.0
    if image.ndim == 3:
        image = np.expand_dims(image, axis=-1)

    return image


def load_model(prod=None):
    """Load a pre-trained Sign Language Digits model."""

    if prod is None:
        model_hf = models.load_model(cfg.model_path)
    else:
        model_hf = models.load_model(cfg.prod_model)

    return model_hf
