"""
MNIST Digit Recognition using Convolutional Neural Network (CNN)
This script implements a CNN model to classify handwritten digits from the MNIST dataset.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data to add channel dimension (required for CNN)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a CNN model for MNIST digit classification.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes (10 for digits 0-9)
    
    Returns:
        keras.Model: Compiled CNN model
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    """
    Train the CNN model on MNIST data.
    
    Args:
        model: Keras model to train
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    
    Returns:
        History: Training history object
    """
    print("\nTraining the model...")
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    
    print("\nEvaluating on test data...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return history


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss.
    
    Args:
        history: Training history object from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history plot saved as 'training_history.png'")


def predict_and_visualize(model, x_test, y_test, num_samples=10):
    """
    Make predictions and visualize results.
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels
        num_samples (int): Number of samples to visualize
    """
    # Make predictions
    predictions = model.predict(x_test[:num_samples])
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Visualize predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(x_test[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'True: {y_test[i]}\nPred: {predicted_labels[i]}')
        axes[i].axis('off')
        
        # Color the title based on correctness
        if y_test[i] == predicted_labels[i]:
            axes[i].title.set_color('green')
        else:
            axes[i].title.set_color('red')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
    print("\nPrediction visualization saved as 'predictions.png'")


def main():
    """
    Main function to run the complete MNIST CNN pipeline.
    """
    print("=" * 60)
    print("MNIST Digit Recognition using CNN")
    print("=" * 60)
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create model
    print("\nCreating CNN model...")
    model = create_cnn_model()
    model.summary()
    
    # Train model
    history = train_model(model, x_train, y_train, x_test, y_test, epochs=10)
    
    # Plot training history
    plot_training_history(history)
    
    # Make predictions and visualize
    predict_and_visualize(model, x_test, y_test)
    
    # Save the model
    model.save('mnist_cnn_model.h5')
    print("\nModel saved as 'mnist_cnn_model.h5'")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
