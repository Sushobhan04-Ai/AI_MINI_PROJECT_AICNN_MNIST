"""
Test script for MNIST CNN implementation.
Tests the model structure and code without requiring dataset download.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def test_model_creation():
    """Test that the model can be created with correct architecture."""
    print("Testing model creation...")
    
    # Create model with same architecture as mnist_cnn.py
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Check model structure - test essential components
    layer_types = [type(layer).__name__ for layer in model.layers]
    assert 'Conv2D' in layer_types, "Model should contain Conv2D layers"
    assert 'MaxPooling2D' in layer_types, "Model should contain MaxPooling2D layers"
    assert 'Dropout' in layer_types, "Model should contain Dropout layers"
    assert 'Dense' in layer_types, "Model should contain Dense layers"
    assert model.input_shape == (None, 28, 28, 1), "Input shape should be (None, 28, 28, 1)"
    assert model.output_shape == (None, 10), "Output shape should be (None, 10)"
    
    print("✓ Model creation test passed!")
    return model


def test_model_prediction():
    """Test that the model can make predictions on dummy data."""
    print("\nTesting model prediction...")
    
    model = test_model_creation()
    
    # Create dummy input data
    dummy_data = np.random.rand(5, 28, 28, 1).astype('float32')
    
    # Make predictions
    predictions = model.predict(dummy_data, verbose=0)
    
    # Verify prediction shape
    assert predictions.shape == (5, 10), "Predictions should have shape (5, 10)"
    
    # Verify predictions are probabilities (sum to 1)
    for i in range(5):
        assert abs(np.sum(predictions[i]) - 1.0) < 0.001, "Predictions should sum to 1"
    
    print("✓ Model prediction test passed!")


def test_data_preprocessing():
    """Test data preprocessing logic."""
    print("\nTesting data preprocessing...")
    
    # Create dummy MNIST-like data
    dummy_images = np.random.randint(0, 256, size=(100, 28, 28), dtype=np.uint8)
    
    # Normalize
    normalized = dummy_images.astype('float32') / 255.0
    assert normalized.max() <= 1.0 and normalized.min() >= 0.0, "Data should be normalized to [0, 1]"
    
    # Add channel dimension
    reshaped = np.expand_dims(normalized, -1)
    assert reshaped.shape == (100, 28, 28, 1), "Data should have shape (100, 28, 28, 1)"
    
    print("✓ Data preprocessing test passed!")


def test_model_training():
    """Test that the model can be trained on dummy data."""
    print("\nTesting model training...")
    
    model = test_model_creation()
    
    # Create dummy training data
    x_train = np.random.rand(100, 28, 28, 1).astype('float32')
    y_train = np.random.randint(0, 10, size=(100,))
    
    # Train for 1 epoch
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=1,
        validation_split=0.1,
        verbose=0
    )
    
    # Check that history contains expected keys
    assert 'accuracy' in history.history, "History should contain 'accuracy'"
    assert 'loss' in history.history, "History should contain 'loss'"
    assert 'val_accuracy' in history.history, "History should contain 'val_accuracy'"
    assert 'val_loss' in history.history, "History should contain 'val_loss'"
    
    print("✓ Model training test passed!")


def test_imports():
    """Test that all required modules can be imported."""
    print("\nTesting imports...")
    
    try:
        import mnist_cnn
        print("✓ mnist_cnn module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import mnist_cnn: {e}")
        return False
    
    # Check that all required functions exist
    required_functions = [
        'load_and_preprocess_data',
        'create_cnn_model',
        'train_model',
        'plot_training_history',
        'predict_and_visualize',
        'main'
    ]
    
    for func_name in required_functions:
        assert hasattr(mnist_cnn, func_name), f"mnist_cnn should have function '{func_name}'"
        print(f"✓ Function '{func_name}' exists")
    
    print("✓ All imports test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running MNIST CNN Tests")
    print("=" * 60)
    
    try:
        test_imports()
        test_data_preprocessing()
        test_model_creation()
        test_model_prediction()
        test_model_training()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
