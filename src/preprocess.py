from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def normalize(images):
    return images.astype("float32") / 255.0


def main():
    with open("params.yaml", encoding="utf-8") as file:
        settings = yaml.safe_load(file)["preprocess"]

    with np.load("data/raw/fashion_mnist.npz") as raw:
        images = raw["train_images"]
        labels = raw["train_labels"]
        test_images = raw["test_images"]
        test_labels = raw["test_labels"]

    # Keep the same class proportions in the training and validation sets.
    train_x, val_x, train_y, val_y = train_test_split(
        images,
        labels,
        test_size=settings["validation_size"],
        random_state=settings["seed"],
        stratify=labels,
    )

    output = Path("data/processed")
    output.mkdir(parents=True, exist_ok=True)

    for name, x, y in [
        ("train", train_x, train_y),
        ("validation", val_x, val_y),
        ("test", test_images, test_labels),
    ]:
        x = normalize(x)
        np.savez_compressed(output / f"{name}.npz", images=x, labels=y)
        print(f"{name}: {len(y)} images, pixel range {x.min():.1f} to {x.max():.1f}")


if __name__ == "__main__":
    main()