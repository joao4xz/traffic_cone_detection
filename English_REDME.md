# Traffic Cone Detection

## Credits

This project uses a dataset from [ikatsamenis](https://github.com/ikatsamenis). You can find the original dataset in their repository: [Cone-Detection](https://github.com/ikatsamenis/Cone-Detection).

## Project Description

This project aims to create a machine learning model capable of detecting traffic cones in images. The model is built using TensorFlow and the VGG16 architecture, with additional custom layers for classification. The project includes scripts for training the model and for making predictions on new images.

## How to Run and Train

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- NumPy

You can install the required packages using pip:

```bash
pip install tensorflow keras matplotlib numpy
```

### Training the Model

1. Clone this repository:

   ```bash
   git clone https://github.com/joao4xz/traffic_cone_detection.git
   cd traffic-cone-detection
   ```

2. Run the script

   ```bash
   python train_model.py
   ```

3. The trained model will be saved as 'traffic_cone_detector.h5'.

## Making predictions

1. Use the `predict_image.py` script to make predictions on new images:

```python
# predict_image.py

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def load_and_preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

def predict_image(img_path, model):
    img = load_and_preprocess_image(img_path)
    prediction = model.predict(img)
    if prediction[0] > 0.5:
        print("Traffic Cone")
    else:
        print("Not a Traffic Cone")

# Load the model
model = load_model('traffic_cone_detector.h5')

# Path to the image you want to test
test_image_path = './test/ok10.png'  # Update with your test image path

# Predict the image
predict_image(test_image_path, model)
```

## Made by

This project was created by [João Henrique](https://github.com/joao4xz) and [Marcelle Andrade](https://github.com/Marcelleap)
