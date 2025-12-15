import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_data(cfg):
    x_load = np.load("ml/training/data/X.npy")
    y_load = np.load("ml/training/data/Y.npy")

    y_labels = np.argmax(y_load, axis=1)

    lookup = np.array([9, 0, 7, 6, 1, 8, 4, 3, 2, 5])
    y_labels = lookup[y_labels]

    x_load = x_load.astype("float32")
    # if x_load.ndim == 3:
    #     x_load = np.expand_dims(x_load, -1)
    if x_load.max() > 1.0:
        x_load /= 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        x_load, y_labels, test_size=0.15, random_state=42, stratify=y_labels
    )

    NUM_CLASSES = cfg.num_classes
    y_train_categorical = to_categorical(y_train, NUM_CLASSES)
    y_test_categorical = to_categorical(y_test, NUM_CLASSES)

    return X_train, y_train_categorical, X_test, y_test_categorical
