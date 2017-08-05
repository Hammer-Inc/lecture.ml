from __future__ import print_function
from flask import Flask, request
import json
import csv
import os, sys
import numpy
import pandas
import requests as ajax
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


@app.route('/')
def get():
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    # load dataset
    dataframe = pandas.read_csv("file.csv", header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8].astype(float)
    Y = dataset[:, 8]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    model = Sequential()
    model.add(Dense(16, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))  # replace with relu
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, encoded_Y, epochs=100, batch_size=1)
    return "{\nSuccess\n}"


@app.route('/post', methods=["POST"])
def post():
    response = "{"
    if request.method != 'POST':
        return
    response += run_ml_on_json()
    response += "}"
    return response


def run_ml_on_json():
    # call microsoft api
    upload_images = request.files[""].read()

    print(request.files[""], file=sys.stderr)
    auth_header = {"Ocp-Apim-Subscription-Key": "04d57b905eee48e980fcecd95007e0a7",
                   "content-type": "application/octet-stream"}

    data_json = ajax.post("https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect", headers=auth_header,
                          data=upload_images)

    data_emotion = json.loads(data_json.text)
    print(data_emotion, file=sys.stderr)
    anger = data_emotion["scores"]["anger"]
    contempt = data_emotion["scores"]["contempt"]
    disgust = data_emotion["scores"]["disgust"]
    fear = data_emotion["scores"]["fear"]
    happiness = data_emotion["scores"]["happiness"]
    neutral = data_emotion["scores"]["neutral"]
    sadness = data_emotion["scores"]["sadness"]
    surprise = data_emotion["scores"]["surprise"]

    return json.dumps(
        model.predict(numpy.array([[anger, contempt, disgust, fear, happiness, neutral, sadness, surprise]])))
