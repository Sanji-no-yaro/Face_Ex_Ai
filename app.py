from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Model and label encoder initialization
img_size = (128, 128)
categories = ['sad', 'surprise', 'happy', 'neutral', 'angry', 'ahegao']
label_encoder = LabelEncoder().fit(categories)

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0
        return np.expand_dims(img, axis=0)
    return None

def load_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    for layer in base_model.layers[:-8]:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(categories), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('face_ex_model.h5')
    return model

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        img = load_and_preprocess_image(file_path)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        return jsonify({'prediction': predicted_label})

    return jsonify({'error': 'File processing error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
