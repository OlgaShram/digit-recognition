
# Digit Recognition

A Python neural network for handwritten digit recognition.

## Features

- Train a convolutional neural network on MNIST.
- Recognize handwritten digits from 0 to 9.
- Display prediction confidence.
- Save the trained model.

## Installation

Install Python, then run:

```bash
python -m pip install -r requirements.txt
```

## Train

```bash
python main.py --train
```

## Predict

```bash
python main.py --predict digit.png
```

The image should contain one handwritten digit.
The model is trained on the MNIST dataset.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pillow
