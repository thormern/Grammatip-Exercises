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

from TrainingPlot import TrainingPlot

dataframe_0 = pd.read_csv("user_data.csv")

dataframe_0.head()


def make_target_column(dataframe):
    taget_column = []

    prev_row = None

    for row in dataframe['exercise_id']:
        if prev_row is not None:
             prev_row = row
             taget_column.append(prev_row)
        else:
            prev_row = row
    taget_column.append(0)

    prev_row = None
    row_count = 0
    index = 0
    for row in dataframe['teacher_id']:
        if prev_row is not None:
            if prev_row != row:
                taget_column[row_count] = 0
            else:
                prev_row = row
        else:
            prev_row = row
        index += 1


    return taget_column


dataframe_exercise_id = pd.get_dummies(dataframe_0.exercise_id, prefix='exercise_id')
# print(dataframe_exercise_id)

dataframe_0['target_exercise_id'] = make_target_column(dataframe_0)
# print(dataframe_0)


dataframe_target_exercise_id = pd.get_dummies(dataframe_0.target_exercise_id, prefix='target_exercise_id')
# print(dataframe_target_exercise_id)


tmp = [dataframe_exercise_id, dataframe_target_exercise_id]

dataframe = dataframe_exercise_id.join(dataframe_target_exercise_id)
dataframe = dataframe.join(dataframe_0)
# print(dataframe)


index = dataframe[dataframe['target_exercise_id'] == 0].index
dataframe.drop(index, inplace=True)
# pd.set_option('display.max_rows', None)
# print(dataframe)


 #########################################


labelEngcoder = preprocessing.LabelEncoder()

dataframe['user_id'] = labelEngcoder.fit_transform(dataframe['user_id'])
dataframe['teacher_id'] = labelEngcoder.fit_transform(dataframe['teacher_id'])
dataframe['date_completed'] = labelEngcoder.fit_transform(dataframe['date_completed'])

# print(dataframe)




#######################################




# Provide the output path and name for the plot
out = 'output/training_plot_model_3.jpg'

# Create an instance of the TrainingPlot class with the filename.
plot_losses = TrainingPlot(out)



# Y = dataframe.target_exercise_id
# # X = dataframe.drop('target_exercise_id', axis=1)
Y = dataframe_target_exercise_id
X = dataframe_exercise_id


X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

# Y = dataframe
# X = dataframe
#
# X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=0.3)
# X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)
#
# print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)



model = keras.Sequential()
model.add(layers.Dense(300, input_dim=X_train.shape[1], activation="relu", name="layer1"))
model.add(layers.Dense(100, activation="relu", name="layer2"))
model.add(layers.Dense(57, activation="sigmoid", name="layer3"))

summery = model.summary()
# plot_model(model, to_file='model_3_plot.png') #, show_shapes=True, show_layer_names=True
# print(summery)


history = model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error", "mean_squared_error", "accuracy"])
earlystopping = EarlyStopping(monitor="mean_absolute_error", patience=10, verbose=1, mode="auto")

result = model.fit(X_train, Y_train, batch_size=16, epochs=5, validation_data=(X_val, Y_val), callbacks=[earlystopping, plot_losses])


Y_pred = model.predict(X_test)

model.evaluate(X_test, Y_test)[1]





