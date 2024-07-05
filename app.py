from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import base64
import keras
from keras import layers
import io
from PIL import Image

app = Flask(__name__)

# Load the CIFAR-10 dataset
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the pixel values to the range [0, 1]
x_test = x_test / 255.0

# One-hot encode the labels
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define the CNN architecture
model=keras.Sequential([
        keras.Input(shape=(32,32,3)),#32 for height 32 for width and 3 for rgb colors
        layers.Conv2D(32,(3,3),padding='valid',activation='relu'), #32 filter of size 3x3 for each layer 
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(128,(3,3),padding='valid',activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(10,activation="softmax")
    ]
)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load the trained model weights
model.load_weights(r"C:\Users\samee\machine learning\image prediction\modelcnn.weights.h5")
  # Ensure the path is correct

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    class_name = class_names[np.argmax(predictions[0])]
    
    return jsonify({'class_name': class_name})

if __name__ == '__main__':
    app.run(debug=True)
