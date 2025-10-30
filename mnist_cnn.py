"""
MNIST Digit Recognition using Convolutional Neural Network (CNN)

This module implements a CNN model for recognizing handwritten digits from the MNIST dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class MNISTDigitRecognizer:
    """CNN-based digit recognizer for MNIST dataset"""
    
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        """
        Initialize the MNIST digit recognizer
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes (10 for digits 0-9)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the CNN architecture"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                         input_shape=self.input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Reshape data to include channel dimension
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=128):
        """
        Train the CNN model
        
        Args:
            x_train: Training images
            y_train: Training labels
            x_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("\nStarting training...")
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        print("\nTraining completed!")
        return self.history
    
    def evaluate(self, x_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            x_test: Test images
            y_test: Test labels (one-hot encoded)
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or trained.")
        
        print("\nEvaluating model...")
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        # Get predictions
        y_pred = self.model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes))
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': y_pred_classes,
            'true_labels': y_true_classes,
            'confusion_matrix': cm
        }
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return results
    
    def predict(self, images):
        """
        Make predictions on new images
        
        Args:
            images: Array of images to predict
        
        Returns:
            Predicted digit classes
        """
        if self.model is None:
            raise ValueError("Model not built or trained.")
        
        predictions = self.model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        return predicted_classes, predictions
    
    def save_model(self, filepath='mnist_cnn_model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not built or trained.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='mnist_cnn_model.h5'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, confusion_matrix, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
        plt.close()
    
    def visualize_predictions(self, images, true_labels, predicted_labels, 
                            num_samples=10, save_path='predictions.png'):
        """Visualize sample predictions"""
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            axes[i].imshow(images[i].reshape(28, 28), cmap='gray')
            
            true_label = true_labels[i] if len(true_labels.shape) == 1 else np.argmax(true_labels[i])
            pred_label = predicted_labels[i]
            
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
        plt.close()


def main():
    """Main function to train and evaluate the CNN model"""
    # Create recognizer instance
    recognizer = MNISTDigitRecognizer()
    
    # Build model
    print("Building CNN model...")
    model = recognizer.build_model()
    print(model.summary())
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = recognizer.load_and_preprocess_data()
    
    # Train model
    recognizer.train(x_train, y_train, x_test, y_test, epochs=10, batch_size=128)
    
    # Evaluate model
    results = recognizer.evaluate(x_test, y_test)
    
    # Save model
    recognizer.save_model('mnist_cnn_model.h5')
    
    # Plot training history
    recognizer.plot_training_history()
    
    # Plot confusion matrix
    recognizer.plot_confusion_matrix(results['confusion_matrix'])
    
    # Visualize some predictions
    recognizer.visualize_predictions(x_test[:10], y_test[:10], 
                                    results['predictions'][:10])
    
    print("\n" + "="*50)
    print("Training and evaluation completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
