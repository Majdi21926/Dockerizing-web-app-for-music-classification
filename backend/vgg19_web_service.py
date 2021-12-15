import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import flask
import io
import string
import time
import os
from PIL import Image
from flask import Flask, jsonify, request
from PIL import Image, ImageOps
import librosa
import librosa.display
import wave

## Loading the Pickled svm_model
file_path = 'C:/Users/majdi/OneDrive/Bureau/MAJDI/PFA2/teachable_machine_model_tf/keras_model.h5'
model = tf.keras.models.load_model(file_path)


def mel_spectrogram(song_path):
    song = wave.open(song_path, "rb")
    y, sr = librosa.load(song)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    img = librosa.power_to_db(mels, ref=np.max)
    return type(img)


def prepare_image(imgpath):
    # img = Image.open(imgpath)
    img = Image.open(io.BytesIO(imgpath))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


def predict(img):
    prediction = model.predict(img)
    label = np.argmax(prediction)
    switcher = {
        0: "blues",
        1: "classical",
        2: "country",
        3: "disco",
        4: "hiphop",
        5: "jazz",
        6: "metal",
        7: "pop",
        8: "reggae",
        9: "rock",
    }
    genre = switcher.get(label, lambda: "Invalid gender")
    return genre


# music = 'C:/Users/majdi/OneDrive/Bureau/Docker/mini-projet-docker/Data/genres_original/blues/blues.00000.wav'
# type = mel_spectrogram(music)
# print(type)

# img = prepare_image(image)
# res = predict(img)
# print(res)

app = Flask(__name__)


@app.route('/vgg19', methods=['POST'])
def infer_image():
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    file = request.files.get('file')

    if not file:
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    # Return on a JSON format
    return jsonify(prediction=predict(img))


@app.route('/', methods=['GET'])
def index():
    return 'Majdi Habibi web service'




if __name__ == '__main__':
    app.run(debug=True)
