import glob
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping
from TrainingPlot import TrainingPlot
from sklearn import preprocessing
from DataGenerator import DataGenerator

dataset_number = sys.argv[1]

out = 'output/datasets_' + dataset_number + '/training_plot_model_6.jpg'
plot_losses = TrainingPlot(out)

train_path = "datasets/not_encoded/train/"
x_train_file = "x_data/x_train_dataset_" + dataset_number + ".csv"
y_train_file = "y_data/y_train_dataset_" + dataset_number + ".csv"

val_path = "datasets/not_encoded/val/"
x_val_file = "x_data/x_val_dataset_" + dataset_number + ".csv"
y_val_file = "y_data/y_val_dataset_" + dataset_number + ".csv"

x_train = pd.read_csv(train_path + x_train_file) #Dataframe
y_train = pd.read_csv(train_path + y_train_file) #Dataframe
x_columns = pd.read_csv("datasets/columns/data/dataset_" + dataset_number + ".csv")
y_columns = pd.read_csv("datasets/columns/labels/dataset_" + dataset_number + ".csv")
train_indices = list(range(sum(1 for line in open(train_path + x_train_file))))  # correct index, 0 index is header always.
train_indices.pop(0)
batch_size = 8192

train_generator = DataGenerator(x_train, y_train, x_columns, y_columns, train_indices, batch_size)


x_val = pd.read_csv(val_path + x_val_file) #Dataframe
y_val = pd.read_csv(val_path + y_val_file) #Dataframe
val_indices = list(range(sum(1 for line in open(val_path + x_val_file))))  # correct index, 0 index is header always.
val_indices.pop(0)
batch_size = 8192

val_generator = DataGenerator(x_val, y_val, x_columns, y_columns, val_indices, batch_size)


model = keras.Sequential()
model.add(layers.Dense(300, activation="relu", input_shape=(len(x_columns.axes[1]), ), name="input_layer"))
model.add(layers.Dense(100, activation="relu", name="layer2"))
model.add(layers.Dense(len(y_columns.axes[1]), activation="sigmoid", name="output_layer"))

summery = model.summary()

print(summery)

history = model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error", "mean_squared_error", "accuracy"])
earlystopping = EarlyStopping(monitor="mean_absolute_error", patience=10, verbose=1, mode="auto")

result = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[earlystopping, plot_losses])
