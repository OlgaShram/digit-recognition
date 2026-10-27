
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter

MODEL_PATH = Path("models/digit_model.keras")
IMAGE_SIZE = 28


def create_model():
    """Создает сверточную нейронную сеть."""
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
    """Обучает нейросеть на MNIST и сохраняет модель."""
    print("Загрузка датасета MNIST...")

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    model = create_model()
    model.summary()

    print("\nНачало обучения...")

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
    )

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    print(f"\nТочность на тестовых данных: {accuracy:.2%}")
    print(f"Функция потерь: {loss:.4f}")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(MODEL_PATH)

    print(f"Модель сохранена: {MODEL_PATH}")


def preprocess_image(image_path):
    """Подготавливает изображение для нейросети."""
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Изображение не найдено: {path}"
        )

    with Image.open(path) as original:
        image = ImageOps.grayscale(original)

        pixels = np.asarray(image, dtype=np.uint8)

        # MNIST использует черный фон и светлую цифру.
        if pixels.mean() > 127:
            image = ImageOps.invert(image)

        # Убираем слабые оттенки и делаем цифру четче.
        image = image.point(
            lambda pixel: 255 if pixel > 80 else 0
        )

        bbox = image.getbbox()

        if bbox is None:
            raise ValueError(
                "Цифра не найдена на изображении."
            )

        # Обрезаем пустые края.
        image = image.crop(bbox)

        # Немного утолщаем линии цифры.
        image = image.filter(
            ImageFilter.MaxFilter(3)
        )

        # Уменьшаем цифру, сохраняя пропорции.
        image.thumbnail(
            (20, 20),
            Image.Resampling.LANCZOS,
        )

        # Размещаем цифру по центру холста 28x28.
        result = Image.new(
            "L",
            (IMAGE_SIZE, IMAGE_SIZE),
            0,
        )

        x = (IMAGE_SIZE - image.width) // 2
        y = (IMAGE_SIZE - image.height) // 2

        result.paste(image, (x, y))

        # Сохраняем увеличенную копию для проверки.
        preview = result.resize(
            (280, 280),
            Image.Resampling.NEAREST,
        )

        preview.save("processed_digit.png")

        pixels = np.asarray(
            result,
            dtype=np.float32,
        ) / 255.0

    return pixels[np.newaxis, ..., np.newaxis]


def predict(image_path):
    """Распознает цифру на изображении."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Обученная модель не найдена. "
            "Сначала выполни: python main.py --train"
        )

    print("Загрузка модели...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Обработка изображения...")
    image = preprocess_image(image_path)

    print(
        "Обработанное изображение сохранено: "
        "processed_digit.png"
    )

    probabilities = model.predict(
        image,
        verbose=0,
    )[0]

    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])

    print(f"\nРаспознанная цифра: {digit}")
    print(f"Уверенность модели: {confidence:.2%}")

    print("\nВероятности по цифрам:")

    for number, probability in enumerate(probabilities):
        print(f"{number}: {probability:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Нейросеть для распознавания "
            "рукописных цифр"
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--train",
        action="store_true",
        help="Обучить нейросеть",
    )

    group.add_argument(
        "--predict",
        metavar="IMAGE",
        help="Распознать цифру на изображении",
    )

    args = parser.parse_args()

    try:
        if args.train:
            train()
        else:
            predict(args.predict)

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nОшибка: {error}")


if __name__ == "__main__":
    main()
