import numpy as np
import pandas as pd

import tensorflow as tf

import io

from PIL import Image

from flask import Flask, request, jsonify, redirect, url_for

import pickle

from sklearn.preprocessing import StandardScaler
from tensorflow import keras

## Loading the vgg19
##file_path = "C:/Users/majdi/OneDrive/Bureau/mini-projet-docker/vgg19.h5"
vgg19 = tf.keras.models.load_model("vgg19.h5")


## Loading the Pickled svm_model
def model():
    #file_name = "C:/Users/majdi/OneDrive/Bureau/Docker/mini-projet-docker/SVM_models/best_svm.pkl"
    model_file = open("best_svm.sav", 'rb')
    loaded_model = pickle.load(model_file)
    return loaded_model


## preprocessing the data
def preprocess_data():
    df = pd.read_csv("features_30_sec.csv")
    df = df.drop(labels='filename', axis=1)
    standardizer = StandardScaler()
    music_data = standardizer.fit_transform(np.array(df.iloc[:, :-1], dtype=float))
    return music_data


def prepare_image(imgpath):
    # img = Image.open(imgpath)
    img = Image.open(io.BytesIO(imgpath))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


def dict_names():
    d = dict()
    df = pd.read_csv("features_30_sec.csv")
    filenames = df.iloc[:, 0]
    for i in range(len(filenames)):
        d[filenames[i]] = i
    return d


def predict_svm(filename, model, file_names, musics):
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


def predict_vgg19(img):
    prediction = vgg19.predict(img)
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


## For SVM
svm = model()
musics = preprocess_data()
files_names = dict_names()

app = Flask(__name__)


@app.route('/svm', methods=['POST'])
def sv_service():
    ##json_data = request.get_json(force=True)
    ##wav_music = json_data['wav_music']
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    file = request.files.get('file')
    wav_music = file.filename
    genre = predict_svm(wav_music, svm, files_names, musics)
    return jsonify({"genre": genre})


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
    return jsonify(prediction=predict_vgg19(img))


@app.route('/', methods=['GET'])
def index():
    return 'Majdi Habibi web service'


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5000)
