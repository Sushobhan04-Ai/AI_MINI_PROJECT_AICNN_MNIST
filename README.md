# MNIST Digit Recognition using CNN

A Convolutional Neural Network (CNN) implementation for recognizing handwritten digits from the MNIST dataset.

## Project Overview

This project implements a deep learning model using CNN architecture to classify handwritten digits (0-9) from the MNIST dataset. The model achieves high accuracy through multiple convolutional layers, pooling, and dropout regularization.

## Features

- **CNN Architecture**: Multi-layer convolutional neural network with dropout regularization
- **MNIST Dataset**: Automatic download and preprocessing of the MNIST dataset
- **Training & Evaluation**: Complete training pipeline with validation and testing
- **Visualization**: Training history plots, confusion matrix, and prediction visualizations
- **Model Persistence**: Save and load trained models
- **Prediction Interface**: Easy-to-use prediction functions for new images

## Model Architecture

```
Input (28x28x1)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Flatten
    ↓
Dropout (0.5)
    ↓
Dense (128 units) + ReLU
    ↓
Dropout (0.5)
    ↓
Dense (10 units) + Softmax
```

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- NumPy
- Matplotlib
- scikit-learn

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

### Training the Model

To train the CNN model on the MNIST dataset:

```bash
python mnist_cnn.py
```

This will:
- Download the MNIST dataset (if not already present)
- Build the CNN model
- Train for 10 epochs with validation
- Evaluate on test set
- Save the trained model as `mnist_cnn_model.h5`
- Generate visualization plots (training history, confusion matrix, predictions)

### Making Predictions

To make predictions using the trained model:

```bash
python predict.py
```

This will:
- Load the trained model
- Make predictions on 20 random test samples
- Display prediction results with confidence scores
- Visualize predictions with correct/incorrect classifications

### Using the Model Programmatically

```python
from mnist_cnn import MNISTDigitRecognizer
import numpy as np

# Create recognizer instance
recognizer = MNISTDigitRecognizer()

# Load trained model
recognizer.load_model('mnist_cnn_model.h5')

# Prepare your image (28x28 grayscale, normalized to 0-1)
image = your_image.reshape(1, 28, 28, 1).astype('float32') / 255.0

# Make prediction
predicted_class, predictions = recognizer.predict(image)
print(f"Predicted digit: {predicted_class[0]}")
```

## Project Structure

```
AI_MINI_PROJECT_AICNN_MNIST/
│
├── mnist_cnn.py          # Main CNN implementation and training
├── predict.py            # Prediction script for inference
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
├── README.md            # Project documentation
│
└── (Generated files after training)
    ├── mnist_cnn_model.h5         # Trained model
    ├── training_history.png       # Training/validation curves
    ├── confusion_matrix.png       # Confusion matrix heatmap
    └── predictions.png            # Sample predictions visualization
```

## Results

After training, the model typically achieves:
- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~98-99%
- **Test Accuracy**: ~98-99%

## Visualizations

The training process generates several visualizations:

1. **Training History**: Shows accuracy and loss curves over epochs
2. **Confusion Matrix**: Displays classification performance across all digits
3. **Predictions**: Shows sample predictions with true/predicted labels

## Dataset

The MNIST database contains:
- 60,000 training images
- 10,000 test images
- Each image is 28x28 pixels, grayscale
- 10 classes (digits 0-9)

## License

This project is open source and available for educational purposes.

## Author

Sushobhan04-Ai

## Acknowledgments

- MNIST dataset by Yann LeCun and Corinna Cortes
- TensorFlow and Keras frameworks
