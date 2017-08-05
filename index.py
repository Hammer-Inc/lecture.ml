from flask import Flask, request
import json
import csv
import os, sys
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


@app.route('/')
def get():
    return "hello world"


@app.route('/post')
def post():
    response = "{"
    if request.method != 'POST':
        return
    uploads = request.files
    for file in uploads:
        response += run_ml_on_json(''.join(file.readlines()))
    response += "}"
    return response


def run_ml_on_json(data_json):
    # data = json.loads(data_json)
    #
    # with open("data.csv", "w") as file:
    #     csv_file = csv.writer(file)
    #     csv_file.writerow(["Emotion", "Score", "classification"])
    #     for item in data:
    #         csv_file.writerow(["Anger", item['scores']["anger"], item['classification']])

    # load dataset
    dataframe = pandas.read_csv("file.csv", header=None)
    dataset = dataframe.values

    # split into input (X) and output (Y) variables
    X = dataset[:, 0:3].astype(float)
    Y = dataset[:, 3]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    model = Sequential()
    model.add(Dense(6, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))  # replace with relu
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, encoded_Y, epochs=100, batch_size=1)

    return json.dumps(model.predict(numpy.array([[0.01, 0.05, 0.01]])))
