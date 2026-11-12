from pathlib import Path

import numpy as np
from tensorflow.keras.datasets import fashion_mnist


def main():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    output = Path("data/raw")
    output.mkdir(parents=True, exist_ok=True)

    # Keep the original pixels and labels together in one compressed file.
    np.savez_compressed(
        output / "fashion_mnist.npz",
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

    print("Training images:", train_images.shape)
    print("Test images:", test_images.shape)
    print("Saved:", output / "fashion_mnist.npz")


if __name__ == "__main__":
    main()