import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
# from tensorflow.keras.layers.experimental import preprocessing

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import feature_column
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
# from tensorflow.python.keras.utils import plot_model
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from TrainingPlot import TrainingPlot


#######################################

# Provide the output path and name for the plot
dataset_number = sys.argv[1]
dataset_chunk = sys.argv[2]
loss = 'output/loss_model_8.jpg'
acc = 'output/acc_model_8.jpg'

# Create an instance of the TrainingPlot class with the filename.
plot_losses = TrainingPlot(loss, True)
plot_accuracy = TrainingPlot(acc, False)
data = pd.read_csv("datasets/reduced_datasets/full_length/" + dataset_number + ".csv")

scalar = MinMaxScaler()

scaled_data = pd.DataFrame(scalar.fit_transform(data))
scaled_data.columns = data.columns

x_data = scaled_data.drop("target_exercise_id", axis=1)
x_data.drop(x_data.columns[[0]], axis=1, inplace = True)
y_data = pd.get_dummies(scaled_data["target_exercise_id"])

x, x_vt, y, y_vt = train_test_split(x_data, y_data, test_size=0.3)
x_v, x_t, y_v, y_t = train_test_split(x_vt, y_vt, test_size=0.5)


model = keras.Sequential()
model.add(layers.Dense(100, input_dim=x.shape[1], activation="relu", name="input_layer"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(100, activation="relu", name="layer2"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(100, activation="relu", name="layer3"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(100, activation="relu", name="layer4"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(y.shape[1], activation="softmax", name="output_layer"))

summery = model.summary()
print(summery)

adam = keras.optimizers.Adam(lr=0.00001)
history = model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
earlystopping = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")

result = model.fit(x, y, batch_size=16, epochs=200, validation_data=(x_v, y_v), callbacks=[earlystopping, plot_losses, plot_accuracy])

# Generate generalization metrics
score = model.evaluate(x_t, y_t, verbose=1)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

save_path = "saved_models/" + dataset_number + "/dateset_" + dataset_chunk + "_model"
keras.models.save_model(model, save_path)

#
predictions = model.predict(x_t)
print(predictions)


pred_df = pd.DataFrame(predictions)
pred_df.to_csv("output/" + dataset_number + "_" + dataset_chunk + "model_8_predictions.csv")
test_labels = pd.DataFrame(y_t)
test_labels.to_csv("output/" + dataset_number + "_" + dataset_chunk + "model_8_predictions_labels.csv")



