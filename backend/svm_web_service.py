import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
import sklearn
import joblib
from PIL import Image
from flask import Flask, jsonify, request
import pickle
import base64
import json
from io import BytesIO
import requests
from flask import Flask, request, jsonify, redirect, url_for
from keras.preprocessing import image
import random
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import sys
import os
import pickle
import librosa
import librosa.display
from IPython.display import Audio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# print('tensorflow version: {}.'.format(tf.__version__))
# print('The scikit-learn version is {}.'.format(sklearn.__version__))
# print('The scikit-learn version is {}.'.format(flask.__version__))

print('Hi, im creating my first flask web service')


## Loading the Pickled svm_model
def model():
    file_name = 'C:/Users/majdi/OneDrive/Bureau/Docker/mini-projet-docker/SVM_models/best_svm.sav'
    model_file = open(file_name, 'rb')
    loaded_model = pickle.load(model_file)
    return loaded_model


## preprocessing the data
def preprocess_data():
    df = pd.read_csv('C:/Users/majdi/OneDrive/Bureau/Docker/mini-projet-docker/Data/features_30_sec.csv')
    df = df.drop(labels='filename', axis=1)
    standardizer = StandardScaler()
    music_data = standardizer.fit_transform(np.array(df.iloc[:, :-1], dtype=float))
    return music_data


def dict_names():
    d = dict()
    df = pd.read_csv('C:/Users/majdi/OneDrive/Bureau/Docker/mini-projet-docker/Data/features_30_sec.csv')
    filenames = df.iloc[:, 0]
    for i in range(len(filenames)):
        d[filenames[i]] = i
    return d


# musics = preprocess_data()
# model = model()
# file_names = dict_names()


def predict(filename, model, file_names, musics):
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
    if filename in file_names.keys():
        num_file = file_names[filename]
        music = np.array([musics[num_file]])
        result = model.predict(music)
        genre = switcher.get(result[0], lambda: "Invalid gender")
        return genre
    else:
        print('there is no file with this name')


svm = model()
musics = preprocess_data()
files_names = dict_names()
# genres = predict('pop.00056.wav', svm, files_names, musics)
# print(genres)

app = Flask(__name__)


@app.route('/svm', methods=['POST'])
def sv_service():
    ##json_data = request.get_json(force=True)
    ##wav_music = json_data['wav_music']
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    file = request.files.get('file')
    wav_music = file.filename
    genre = predict(wav_music, svm, files_names, musics)
    return jsonify({"genre": genre})


@app.route('/', methods=['GET'])
def index():
    return 'Majdi Habibi svm web service'


if __name__ == '__main__':
    app.run(debug=True)

## Preprocessing audios
# def encode_audio(wav_music):
#    music_file = open(wav_music, "rb")
#    music_content = music_file.read()
#    music_binary = base64.b64encode(music_content)
#    return music_binary


## Predicting the genre of the music
# def predict(model, base64_music):
#    genre = model.predict(base64_music)
#    return genre


print('execution done')
