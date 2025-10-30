"""
Prediction script for MNIST digit recognition

This script loads a trained model and makes predictions on test images.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from mnist_cnn import MNISTDigitRecognizer


def predict_sample_images(model_path='mnist_cnn_model.h5', num_samples=20):
    """
    Load model and predict on sample images from test set
    
    Args:
        model_path: Path to the saved model
        num_samples: Number of samples to predict
    """
    # Load test data
    print("Loading test data...")
    (_, _), (x_test, y_test) = mnist.load_data()
    
    # Preprocess test images
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Create recognizer and load model
    recognizer = MNISTDigitRecognizer()
    recognizer.load_model(model_path)
    
    # Select random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    sample_images = x_test[indices]
    sample_labels = y_test[indices]
    
    # Make predictions
    print(f"\nMaking predictions on {num_samples} samples...")
    predicted_classes, predictions = recognizer.predict(sample_images)
    
    # Display results
    print("\nPrediction Results:")
    print("-" * 50)
    for i in range(num_samples):
        confidence = predictions[i][predicted_classes[i]] * 100
        correct = "✓" if predicted_classes[i] == sample_labels[i] else "✗"
        print(f"Image {i+1}: True={sample_labels[i]}, "
              f"Predicted={predicted_classes[i]}, "
              f"Confidence={confidence:.2f}% {correct}")
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == sample_labels) * 100
    print("-" * 50)
    print(f"Accuracy on sample: {accuracy:.2f}%")
    
    # Visualize predictions
    recognizer.visualize_predictions(sample_images, sample_labels, 
                                    predicted_classes, num_samples=num_samples,
                                    save_path='sample_predictions.png')
    
    return predicted_classes, sample_labels


def predict_single_image(model_path='mnist_cnn_model.h5', image_index=0):
    """
    Predict a single image from test set
    
    Args:
        model_path: Path to the saved model
        image_index: Index of the image to predict
    """
    # Load test data
    (_, _), (x_test, y_test) = mnist.load_data()
    
    # Preprocess image
    image = x_test[image_index:image_index+1]
    image = image.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    true_label = y_test[image_index]
    
    # Create recognizer and load model
    recognizer = MNISTDigitRecognizer()
    recognizer.load_model(model_path)
    
    # Make prediction
    predicted_class, predictions = recognizer.predict(image)
    confidence = predictions[0][predicted_class[0]] * 100
    
    # Display result
    print(f"\nPrediction for image at index {image_index}:")
    print(f"True label: {true_label}")
    print(f"Predicted label: {predicted_class[0]}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nAll class probabilities:")
    for digit in range(10):
        prob = predictions[0][digit] * 100
        print(f"  Digit {digit}: {prob:.2f}%")
    
    # Visualize
    plt.figure(figsize=(6, 6))
    plt.imshow(x_test[image_index], cmap='gray')
    color = 'green' if predicted_class[0] == true_label else 'red'
    plt.title(f'True: {true_label}, Predicted: {predicted_class[0]} ({confidence:.1f}%)', 
             color=color, fontsize=14)
    plt.axis('off')
    plt.savefig('single_prediction.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to single_prediction.png")
    plt.close()
    
    return predicted_class[0], true_label


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("MNIST Digit Recognition - Prediction Script")
    print("="*60)
    
    # Check if model exists
    import os
    model_path = 'mnist_cnn_model.h5'
    
    if not os.path.exists(model_path):
        print(f"\nError: Model file '{model_path}' not found!")
        print("Please train the model first by running: python mnist_cnn.py")
        sys.exit(1)
    
    # Predict on multiple samples
    print("\n--- Predicting on sample images ---")
    predict_sample_images(model_path, num_samples=20)
    
    # Predict on a single image
    print("\n--- Predicting on a single image ---")
    predict_single_image(model_path, image_index=42)
    
    print("\n" + "="*60)
    print("Prediction completed successfully!")
    print("="*60)
