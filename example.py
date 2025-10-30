"""
Example script demonstrating the MNIST CNN digit recognizer usage

This is a simplified example showing how to use the MNISTDigitRecognizer class.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

from mnist_cnn import MNISTDigitRecognizer


def run_example():
    """Run a complete example of training and prediction"""
    
    print("="*70)
    print("MNIST Digit Recognition Example")
    print("="*70)
    
    # Step 1: Create recognizer instance
    print("\n[Step 1] Creating MNISTDigitRecognizer instance...")
    recognizer = MNISTDigitRecognizer(input_shape=(28, 28, 1), num_classes=10)
    print("✓ Recognizer created")
    
    # Step 2: Build the CNN model
    print("\n[Step 2] Building CNN model...")
    model = recognizer.build_model()
    print("✓ Model built with the following architecture:")
    print(f"  - Total parameters: {model.count_params():,}")
    print(f"  - Number of layers: {len(model.layers)}")
    
    # Step 3: Load and preprocess data
    print("\n[Step 3] Loading and preprocessing MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = recognizer.load_and_preprocess_data()
    print(f"✓ Dataset loaded:")
    print(f"  - Training samples: {x_train.shape[0]:,}")
    print(f"  - Test samples: {x_test.shape[0]:,}")
    print(f"  - Image shape: {x_train.shape[1:]}")
    
    # Step 4: Train the model
    print("\n[Step 4] Training the model...")
    print("(Training with 10 epochs - this may take a few minutes)")
    history = recognizer.train(
        x_train, y_train, 
        x_test, y_test,
        epochs=10,
        batch_size=128
    )
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\n✓ Training completed:")
    print(f"  - Final training accuracy: {final_train_acc*100:.2f}%")
    print(f"  - Final validation accuracy: {final_val_acc*100:.2f}%")
    
    # Step 5: Evaluate on test set
    print("\n[Step 5] Evaluating model on test set...")
    results = recognizer.evaluate(x_test, y_test)
    print(f"✓ Evaluation completed:")
    print(f"  - Test accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"  - Test loss: {results['test_loss']:.4f}")
    
    # Step 6: Make predictions on sample images
    print("\n[Step 6] Making predictions on sample images...")
    sample_images = x_test[:5]
    predicted_classes, predictions = recognizer.predict(sample_images)
    
    print("✓ Predictions for first 5 test images:")
    for i, (pred, probs) in enumerate(zip(predicted_classes, predictions)):
        true_label = y_test[i].argmax() if len(y_test[i].shape) > 0 else y_test[i]
        confidence = probs[pred] * 100
        status = "✓" if pred == true_label else "✗"
        print(f"  Image {i+1}: Predicted={pred}, True={true_label}, "
              f"Confidence={confidence:.1f}% {status}")
    
    # Step 7: Save the model
    print("\n[Step 7] Saving the trained model...")
    model_path = 'mnist_cnn_model.h5'
    recognizer.save_model(model_path)
    print(f"✓ Model saved to '{model_path}'")
    
    # Step 8: Generate visualizations
    print("\n[Step 8] Generating visualizations...")
    recognizer.plot_training_history('training_history.png')
    recognizer.plot_confusion_matrix(results['confusion_matrix'], 'confusion_matrix.png')
    recognizer.visualize_predictions(
        x_test[:10], y_test[:10], 
        results['predictions'][:10],
        save_path='predictions.png'
    )
    print("✓ Visualizations saved:")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("  - predictions.png")
    
    # Step 9: Demonstrate loading a saved model
    print("\n[Step 9] Testing model loading...")
    new_recognizer = MNISTDigitRecognizer()
    new_recognizer.load_model(model_path)
    
    # Make a prediction with loaded model
    test_pred, _ = new_recognizer.predict(x_test[:1])
    print(f"✓ Loaded model prediction: {test_pred[0]}")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Check the generated visualization files")
    print("  2. Use 'predict.py' to make predictions on new images")
    print("  3. Customize the model architecture in 'mnist_cnn.py'")
    print("="*70)


if __name__ == "__main__":
    run_example()
