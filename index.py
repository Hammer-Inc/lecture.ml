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
    response += run_ml_on_json(request.json)
    response += "}"
    return response


def run_ml_on_json(data):
    # call microsoft api

    array = []

    for x in range(0, len(data) - 1):
        tmp = json.load(data[x])
        array.append(tmp)

    arrayEmotion = []

    for z in array:
        arrayemotion.append([z])
        arrayEmotion[z].append(x["faceAttributes"]["emotion"]["anger"])
        arrayEmotion[z].append(x["faceAttributes"]["emotion"]["contempt"])
        arrayEmotion[z].append(x["faceAttributes"]["emotion"]["disgust"])
        arrayEmotion[z].append(x["faceAttributes"]["emotion"]["fear"])
        arrayEmotion[z].append(x["faceAttributes"]["emotion"]["happiness"])
        arrayEmotion[z].append(x["faceAttributes"]["emotion"]["neutral"])
        arrayEmotion[z].append(x["faceAttributes"]["emotion"]["sadness"])
        arrayEmotion[z].append(x["faceAttributes"]["emotion"]["surprise"])

    confArray = []

    for y in range(0, 7):
        confidence = model.predict(
            numpy.array([[arrayEmotion[y][0], arrayEmotion[y][1], arrayEmotion[y][2], arrayEmotion[y][3], arrayEmotion[y][4], arrayEmotion[y][5], arrayEmotion[y][6], arrayEmotion[y][7]]]))
        confArray.append(confidence)

    return confArray
