from pathlib import Path

import numpy as np
import yaml
from tensorflow import keras


def main():
    with open("params.yaml", encoding="utf-8") as file:
        settings = yaml.safe_load(file)["train"]

    keras.utils.set_random_seed(settings["seed"])

    with np.load("data/processed/train.npz") as data:
        train_images = data["images"]
        train_labels = data["labels"]

    with np.load("data/processed/validation.npz") as data:
        val_images = data["images"]
        val_labels = data["labels"]

    model = keras.Sequential([
        keras.Input(shape=train_images.shape[1:]),
        keras.layers.Flatten(),
        keras.layers.Dense(settings["dense_units"], activation="relu"),
        keras.layers.Dropout(settings["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    output = Path("models")
    output.mkdir(exist_ok=True)

    # Write one row per epoch so we can compare training and validation later.
    history_file = output / "history.csv"
    model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        callbacks=[keras.callbacks.CSVLogger(str(history_file))],
        verbose=2,
    )

    model.save(output / "model.h5")
    print("Saved:", output / "model.h5")
    print("Saved:", history_file)


if __name__ == "__main__":
    main()