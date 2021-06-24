import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# Input files
# train = pd.read_csv("user_data_train.csv")
# test = pd.read_csv("user_data_test.csv")


data = pd.read_csv("user_data.csv")
data.head()

labelEngcoder = preprocessing.LabelEncoder()

data['user_id'] = labelEngcoder.fit_transform(data['user_id'])
data['teacher_id'] = labelEngcoder.fit_transform(data['teacher_id'])
data['date_completed'] = labelEngcoder.fit_transform(data['date_completed'])

print(data)
y = data.exercise_id
x = data.drop('exercise_id', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = keras.Sequential()
model.add(layers.Dense(100, input_dim=4, activation="relu", name="layer1"))
model.add(layers.Dense(50, activation="relu", name="layer2"))
model.add(layers.Dense(1, name="layer3"))

model.summary()

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)


