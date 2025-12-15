import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)
from keras.models import Sequential
from tensorflow import keras

from ml.ml_model import Config
from ml.training.data_preprocessing import load_data
from ml.training.plots import plot_confusion_matrix, plot_history

# import onnx
# import tf2onnx


def train_model(cfg):

    augmentation = Sequential(
        [
            RandomRotation(0.10, fill_mode="nearest"),
            RandomTranslation(0.10, 0.10, fill_mode="nearest"),
            RandomZoom(0.10, fill_mode="nearest"),
            RandomContrast(0.15),
            RandomFlip("horizontal"),
        ],
        name="augmentation",
    )

    inputs = keras.Input(shape=cfg.input_shape)
    x = augmentation(inputs)

    for i in range(3):

        x = Conv2D(
            32 * 2**i,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
        )(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-3)(x)
        x = Conv2D(
            32 * 2**i,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
        )(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-3)(x)
        x = MaxPooling2D(2)(x)
        x = Dropout(0.10 * (i + 1))(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-3)(x)
    x = Dropout(0.40)(x)

    outputs = Dense(cfg.num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="digits_cnn")
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stoping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        restore_best_weights=True,
        verbose=0,
    )

    reduce_learning_rate = ReduceLROnPlateau(
        monitor="val_accuracy", patience=2, factor=0.5, verbose=1, min_lr=1e-5
    )

    checkpoint_filepath = cfg.model_path + "checkpoints/checkpoint{epoch:04d}.keras"

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        # Metric to monitor (e.g., validation accuracy)
        monitor="val_accuracy",
        save_best_only=True,  # Only save the model when it is the best so far
        # Save the entire model (False) or just weights (True)
        save_weights_only=False,
        save_freq=10 * cfg.model_cfg["training"]["batch_size"],
        verbose=1,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(val_x, val_y),
        epochs=cfg.model_cfg["training"]["epochs"],
        callbacks=[early_stoping, reduce_learning_rate, model_checkpoint_callback],
        verbose=1,
        batch_size=cfg.model_cfg["training"]["batch_size"],
    )

    pd.DataFrame(history.history).to_csv(
        cfg.model_path + "train_history.csv", index=False
    )
    model.save(cfg.model_path)

    # Evaluvate for train generator
    loss, acc = model.evaluate(x_train, y_train, verbose=0)

    print("The accuracy of the model for training data is:", acc * 100)
    print("The Loss of the model for training data is:", loss)

    # Evaluvate for validation generator
    loss, acc = model.evaluate(val_x, val_y, verbose=0)

    print("The accuracy of the model for validation data is:", acc * 100)
    print("The Loss of the model for validation data is:", loss)

    return history, model


def fine_tuting(base_model):
    for layer in base_model.layers:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stoping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        restore_best_weights=True,
        verbose=0,
    )

    reduce_learning_rate = ReduceLROnPlateau(
        monitor="val_accuracy", patience=2, factor=0.5, verbose=1, min_lr=1e-5
    )

    new_history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(val_x, val_y),
        epochs=cfg.model_cfg["fine_tuning"]["epochs"],
        callbacks=[early_stoping, reduce_learning_rate],
        verbose=1,
        batch_size=cfg.model_cfg["fine_tuning"]["batch_size"],
    )

    pd.DataFrame(new_history.history).to_csv(
        cfg.path + "train_history.csv", index=False
    )
    model.save(cfg.path + "fune_tining.keras")

    # Evaluvate for train generator
    loss, acc = model.evaluate(x_train, y_train, verbose=0)

    print("The accuracy of the model for fine_tuning data is:", acc * 100)
    print("The Loss of the model for fine_tuning data is:", loss)

    # Evaluvate for validation generator
    loss, acc = model.evaluate(val_x, val_y, verbose=0)

    print("The accuracy of the model for validation data is:", acc * 100)
    print("The Loss of the model for validation data is:", loss)


if __name__ == "__main__":

    cfg = Config()
    x_train, y_train, val_x, val_y = load_data(cfg)

    history, model = train_model(cfg)
    plot_history(history)
    model = keras.models.load_model(cfg.model_path)
    fine_tuting(model)

    plot_confusion_matrix(val_y, model.predict(val_x))
