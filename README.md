# MNIST Digit Recognition using CNN

A deep learning project that implements a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset.

## Overview

This project uses TensorFlow/Keras to build and train a CNN model that achieves high accuracy on the MNIST digit classification task. The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9), with 60,000 training images and 10,000 test images.

## Features

- **CNN Architecture**: Multi-layer convolutional neural network with pooling and dropout
- **Data Preprocessing**: Automatic normalization and reshaping of MNIST data
- **Training Visualization**: Plots showing training/validation accuracy and loss
- **Prediction Visualization**: Visual comparison of true labels vs predictions
- **Model Persistence**: Save and load trained models

## Requirements

- Python 3.7+
- TensorFlow 2.12.1+
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sushobhan04-Ai/AI_MINI_PROJECT_AICNN_MNIST.git
cd AI_MINI_PROJECT_AICNN_MNIST
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the training script:
```bash
python mnist_cnn.py
```

This will:
1. Load and preprocess the MNIST dataset
2. Create a CNN model
3. Train the model for 10 epochs
4. Evaluate the model on test data
5. Generate visualization plots
6. Save the trained model

## Model Architecture

The CNN model consists of:
- **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D Layer 1**: 2x2 pool size
- **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D Layer 2**: 2x2 pool size
- **Flatten Layer**: Converts 2D features to 1D
- **Dropout Layer 1**: 50% dropout rate
- **Dense Layer 1**: 128 units, ReLU activation
- **Dropout Layer 2**: 50% dropout rate
- **Dense Layer 2**: 10 units (output), Softmax activation

## Results

The model typically achieves:
- Training accuracy: ~99%
- Test accuracy: ~98-99%

## Output Files

After training, the following files are generated:
- `mnist_cnn_model.h5`: Trained model file
- `training_history.png`: Plot showing training/validation accuracy and loss
- `predictions.png`: Visualization of predictions on test samples

## Project Structure

```
AI_MINI_PROJECT_AICNN_MNIST/
├── mnist_cnn.py          # Main training script
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Git ignore file
```

## License

This project is open source and available for educational purposes.
