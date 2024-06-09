import tensorflow as tf
import numpy as np
import os

# Load and test the model on new images
def load_and_preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

def predict_image(img_path, model_path='traffic_cone_detector.h5'):
    model = tf.keras.models.load_model(model_path)
    img = load_and_preprocess_image(img_path)
    prediction = model.predict(img)
    if prediction[0] > 0.5:
        print(f"Traffic Cone {prediction[0]}")
    else:
        print(f"Not a Traffic Cone {prediction[0]}")

# Test the model
test_image_path = './test/1498292.jpg'  # Update with your test image path
predict_image(test_image_path)