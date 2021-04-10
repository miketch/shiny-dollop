import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from flask import Flask, request, jsonify
app = Flask(__name__)


@app.route("/api", methods=['GET'])
def hello():

    data = request.get_json()

    train_data = np.empty((0,12), float)
    test_data = np.empty((0,11), float)
    

    for i in data["Training"]:
        current = data["Training"][i].split(",")
        current = [float(item) for item in current]
        current = np.array(current).T
        current.shape = (1, 12)
        train_data = np.append(train_data, current, axis=0)
        
    train_labels = train_data[:, 11]
    train_data = np.delete(train_data, -1, axis=1)



    for i in data["Testing"]:
        current = data["Testing"][i].split(",")
        current = [float(item) for item in current]
        current = np.array(current).T
        current.shape = (1, 11)
        test_data = np.append(test_data, current, axis=0)
        


    # Build the model.
    model = Sequential([
    Dense(128, activation='relu', input_shape=(11,)),
    Dense(128, activation='sigmoid'),
    Dense(1, activation='softmax'),
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
    optimizer=opt,
    loss='mean_squared_error',
    metrics=['accuracy'],
    )

    model.fit(
    train_data, # training data
    to_categorical(train_labels), # training targets
    epochs=5,
    batch_size=1,
    )

    for i in test_data:
        y = model(test_data)
        result = y.numpy()
        list = result.tolist()
        print(result)


    return jsonify(list)


@app.route("/", methods=['GET'])
def welcome():
    return "Welcome!"