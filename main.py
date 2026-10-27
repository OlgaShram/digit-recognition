
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = Path("models/digit_model.keras")


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train():
    print("Loading MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    model = create_model()
    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Test loss: {loss:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def predict(image_path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python main.py --train"
        )

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    model = tf.keras.models.load_model(MODEL_PATH)

    with Image.open(path) as image:
        image = ImageOps.grayscale(image)
        image = ImageOps.fit(
            image,
            (28, 28),
            method=Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(image, dtype=np.float32)

    # MNIST images have a dark background and a light digit.
    if pixels.mean() > 127:
        pixels = 255.0 - pixels

    pixels /= 255.0
    pixels = pixels[np.newaxis, ..., np.newaxis]

    probabilities = model.predict(pixels, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])

    print(f"Predicted digit: {digit}")
    print(f"Confidence: {confidence:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Handwritten digit recognition"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--predict", metavar="IMAGE")

    args = parser.parse_args()

    if args.train:
        train()
    else:
        predict(args.predict)


if __name__ == "__main__":
    main()
