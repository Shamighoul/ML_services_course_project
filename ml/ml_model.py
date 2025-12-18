import io

import numpy as np
import yaml
from keras import models
from PIL import Image

from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score,
)
from keras.utils import to_categorical



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
    if image.mode == 'RGB':
        image = image.convert('L')  
    image = image.resize((64, 64), Image.Resampling.BILINEAR)
    image = np.array(image, dtype=np.float32)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    image = image.astype(np.float32)
    if image.max() > 1:
        image = image / 255.0
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    return image


def load_model(prod=None):
    """Load a pre-trained Sign Language Digits model."""

    if prod is None:
        model_hf = models.load_model(cfg.model_path)
    else:
        model_hf = models.load_model(cfg.prod_model)

    return model_hf

def calc_metrics(y_true, y_pred):
    """Calculate metrics."""
    y_pred = np.array(y_pred)
    multy_y_true = to_categorical(y_true, num_classes=cfg.num_classes)

    y_true = np.array(list(map(int, y_true)))
    only_y_pred = y_pred.argmax(axis=2)

    return {
            "accuracy": accuracy_score(y_true, only_y_pred),
            "f1": f1_score(y_true, only_y_pred, average="macro"),
            "roc_auc": roc_auc_score(multy_y_true, y_pred[:,:,0], average="macro"),
        }

