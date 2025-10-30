# MNIST CNN Usage Examples

## Basic Usage

### 1. Train the Model

```bash
python mnist_cnn.py
```

This will:
- Download and preprocess the MNIST dataset
- Create and compile the CNN model
- Train for 10 epochs
- Save the trained model as `mnist_cnn_model.h5`
- Generate visualization plots

### 2. Expected Output

```
============================================================
MNIST Digit Recognition using CNN
============================================================
Training data shape: (60000, 28, 28, 1)
Training labels shape: (60000,)
Test data shape: (10000, 28, 28, 1)
Test labels shape: (10000,)

Creating CNN model...
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 26, 26, 32)        320       
max_pooling2d (MaxPooling2D)(None, 13, 13, 32)       0         
conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496     
max_pooling2d_1 (MaxPooling2D)(None, 5, 5, 64)        0         
flatten (Flatten)           (None, 1600)              0         
dropout (Dropout)           (None, 1600)              0         
dense (Dense)               (None, 128)               204928    
dropout_1 (Dropout)         (None, 128)               0         
dense_1 (Dense)             (None, 10)                1290      
=================================================================
Total params: 225034 (879.04 KB)
Trainable params: 225034 (879.04 KB)
Non-trainable params: 0 (0.00 Byte)

Training the model...
Epoch 1/10
422/422 [==============================] - 15s 34ms/step
...
Test accuracy: 0.9876
```

### 3. Using the Trained Model

```python
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Prepare your image (28x28 grayscale, normalized to [0,1])
image = np.random.rand(1, 28, 28, 1)  # Replace with your actual image

# Make prediction
prediction = model.predict(image)
predicted_digit = np.argmax(prediction)

print(f"Predicted digit: {predicted_digit}")
```

### 4. Custom Training

```python
from mnist_cnn import load_and_preprocess_data, create_cnn_model, train_model

# Load data
(x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

# Create model
model = create_cnn_model()

# Train with custom parameters
history = train_model(
    model, 
    x_train, y_train, 
    x_test, y_test, 
    epochs=5,  # Fewer epochs
    batch_size=256  # Larger batch size
)
```

## Testing

Run the test suite to verify the implementation:

```bash
python test_mnist_cnn.py
```

Expected output:
```
============================================================
Running MNIST CNN Tests
============================================================

Testing imports...
✓ All imports test passed!

Testing data preprocessing...
✓ Data preprocessing test passed!

Testing model creation...
✓ Model creation test passed!

Testing model prediction...
✓ Model prediction test passed!

Testing model training...
✓ Model training test passed!

============================================================
ALL TESTS PASSED!
============================================================
```

## Tips for Best Results

1. **GPU Acceleration**: If you have a CUDA-compatible GPU, TensorFlow will automatically use it for faster training.

2. **Adjusting Hyperparameters**: You can modify the model architecture or training parameters in `mnist_cnn.py`:
   - Change the number of filters in Conv2D layers
   - Adjust dropout rates
   - Modify the learning rate by changing the optimizer
   - Increase/decrease the number of epochs

3. **Early Stopping**: Add early stopping to prevent overfitting:
   ```python
   from tensorflow.keras.callbacks import EarlyStopping
   
   early_stop = EarlyStopping(monitor='val_loss', patience=3)
   history = model.fit(..., callbacks=[early_stop])
   ```

4. **Data Augmentation**: Enhance training with data augmentation:
   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   
   datagen = ImageDataGenerator(
       rotation_range=10,
       zoom_range=0.1,
       width_shift_range=0.1,
       height_shift_range=0.1
   )
   ```

## Troubleshooting

### Out of Memory Error
- Reduce batch size in `train_model()` function
- Use fewer filters in convolutional layers

### Low Accuracy
- Train for more epochs
- Add data augmentation
- Reduce dropout rates
- Increase model capacity (more filters/layers)

### Slow Training
- Use GPU if available
- Increase batch size
- Reduce model complexity
