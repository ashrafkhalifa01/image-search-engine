from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import csv

# Get the current working directory
current_dir = os.getcwd()

# Set the path to your train, test, and predict directories relative to the current directory
train_dir = os.path.join(current_dir, "train")
test_dir = os.path.join(current_dir, "test")
predict_dir = os.path.join(current_dir, "predict")

# Define the number of classes and input shape
num_classes = 10
input_shape = (224, 224, 3)

# Load the pre-trained ResNet50 model without the top (classification) layer
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Freeze the pre-trained layers
base_model.trainable = False

# Create the classification model
model = keras.Sequential([
    base_model,
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load image features from the CSV file
csv_file_path = "image_features_mod.csv"
image_features = {}
with open(csv_file_path, "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        image_path = row[0]
        features = np.array(row[1:], dtype=np.float32)
        image_features[image_path] = features

# Function to extract features from an image
def extract_features(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=input_shape[:2])
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array)
    return features.reshape((-1,))

# Function to find similar images based on cosine similarity
def find_similar_images(query_image_path):
    query_features = extract_features(query_image_path)
    similar_images = []

    with open(csv_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            image_path = row[0]
            features = np.array(row[1:], dtype=np.float32)
            similarity = cosine_similarity([query_features], [features])[0][0]
            image_path = image_path[30:]
            similar_images.append((image_path, similarity))

    similar_images.sort(key=lambda x: x[1], reverse=True)
    return similar_images


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        query_image = request.files['query_image']
        query_image_path = "static/" + query_image.filename
        query_image.save(query_image_path)

        # Find similar images
        similar_images = find_similar_images(query_image_path)
        print(similar_images[:3])
        return render_template('result.html', query_image=query_image.filename, similar_images=similar_images[:4])

if __name__ == '__main__':
    app.run(debug=True)
