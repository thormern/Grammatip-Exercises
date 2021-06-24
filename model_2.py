import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import feature_column
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

data = pd.read_csv("user_data_small.csv")
data.head()

labelEngcoder = preprocessing.LabelEncoder()

data['user_id'] = labelEngcoder.fit_transform(data['user_id'])
data['teacher_id'] = labelEngcoder.fit_transform(data['teacher_id'])
data['date_completed'] = labelEngcoder.fit_transform(data['date_completed'])

print(data)

Y = data.exercise_id
X = data.drop('exercise_id', axis=1)

X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

model = keras.Sequential()
model.add(layers.Dense(300, input_dim=X_train.shape[1], activation="relu", name="layer1"))
model.add(layers.Dense(100, activation="relu", name="layer2"))
model.add(layers.Dense(1, activation="softmax", name="layer3"))

model.summary()

history = model.compile(optimizer="Nadam", loss="mean_squared_error", metrics=["mean_absolute_error", "mean_squared_error"])
earlystopping = EarlyStopping(monitor="mean_absolute_error", patience=10, verbose=1, mode="auto")
result = model.fit(X_train, Y_train, batch_size=16, epochs=5, validation_data=(X_val, Y_val), callbacks=[earlystopping])

Y_pred = model.predict(X_test)

model.evaluate(X_test, Y_test)[1]
