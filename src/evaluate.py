import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras


def main():
    with np.load("data/processed/test.npz") as data:
        images = data["images"]
        labels = data["labels"]

    model = keras.models.load_model("models/model.h5", compile=False)
    probabilities = model.predict(images, verbose=0)
    predictions = probabilities.argmax(axis=1)

    # Use the same loss function that was used during training.
    loss = keras.losses.SparseCategoricalCrossentropy()
    metrics = {
        "test_loss": float(loss(labels, probabilities)),
        "test_accuracy": float(np.mean(predictions == labels)),
    }

    with open("metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]

    output = Path("reports")
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay.from_predictions(
        labels, predictions, labels=range(10), display_labels=class_names,
        cmap="Blues", values_format="d", xticks_rotation=45,
        colorbar=False, ax=ax,
    )
    ax.set_title("Fashion-MNIST test set")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"Test images: {len(labels)}")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.2%}")
    print("Saved: metrics.json and reports/confusion_matrix.png")


if __name__ == "__main__":
    main()