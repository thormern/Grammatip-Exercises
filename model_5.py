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

# x_train_path = glob.glob("datasets/train/x_data/" + "*csv")
# x_train_files = [i.split('/')[-1] for i in x_train_path if dataset_number in i]
#
# y_train_path = glob.glob("datasets/train/y_data/" + "*csv")
# y_train_files = [i.split('/')[-1] for i in y_train_path if dataset_number in i]
#
# x_val_path = glob.glob("datasets/val/x_data/" + "*csv")
# x_val_files = [i.split('/')[-1] for i in x_val_path if dataset_number in i]
#
# y_val_path = glob.glob("datasets/val/y_data/" + "*csv")
# y_val_files = [i.split('/')[-1] for i in y_val_path if dataset_number in i]

out = 'output/datasets_' + dataset_number + '/training_plot_model_5.jpg'
plot_losses = TrainingPlot(out)

# def generate_batches(path, x_file, y_file, chunksize):
#     while True:
#         labelEngcoder = preprocessing.LabelEncoder()
#         x_file = glob.glob(path + "x_data/" + x_file)
#         y_file = glob.glob(path + "y_data/" + y_file)
#
#         x_chunks = pd.read_csv(x_file[0], chunksize=chunksize)
#         y_chunks = pd.read_csv(y_file[0], chunksize=chunksize)
#
#         for x_chunk, y_chunk in x_chunks, y_chunks:
#             x_chunk['user_id'] = labelEngcoder.fit_transform(x_chunk['user_id'])
#             x_chunk['teacher_id'] = labelEngcoder.fit_transform(x_chunk['teacher_id'])
#             x_chunk['date_completed'] = labelEngcoder.fit_transform(x_chunk['date_completed'])
#             yield x_chunk, y_chunk

# def generate_batches(path, file, isX, chunksize):
#     while True:
#         if isX:
#             labelEngcoder = preprocessing.LabelEncoder()
#         file = glob.glob(path + "x_data/" + file)
#
#         chunks = pd.read_csv(file[0], chunksize=chunksize)
#
#         for chunk in chunks:
#             if isX:
#                 chunk['user_id'] = labelEngcoder.fit_transform(chunk['user_id'])
#                 chunk['teacher_id'] = labelEngcoder.fit_transform(chunk['teacher_id'])
#                 chunk['date_completed'] = labelEngcoder.fit_transform(chunk['date_completed'])
#             yield chunk

# def generate_batches(path, x, y, chunksize):
#     while True:
#         labelEngcoder = preprocessing.LabelEncoder()
#         x_file = glob.glob(path + "x_data/" + x)
#         y_file = glob.glob(path + "y_data/" + y)
#         print(x_file)
#         print("------------")
#         print(y_file)
#         x_chunks = pd.read_csv(x_file[0], chunksize=chunksize)
#         y_chunks = pd.read_csv(y_file[0], chunksize=chunksize)
#
#         for x_chunk in x_chunks:
#             print(x_chunk)
#             x_chunk['user_id'] = labelEngcoder.fit_transform(x_chunk['user_id'])
#             x_chunk['teacher_id'] = labelEngcoder.fit_transform(x_chunk['teacher_id'])
#             x_chunk['date_completed'] = labelEngcoder.fit_transform(x_chunk['date_completed'])
#             print("----------------")
#             print(x_chunk)
#             yield x_chunk
#
#         print("Reach here?")
#         for y_chunk in y_chunks:
#             print(y_chunk)
#
#             yield y_chunk


# def generate_batches(data_folder):
#     cnt = 0
#     x_path = glob.glob(data_folder + "x_data/" + "*csv")
#     x_files = [i.split('/')[-1] for i in x_path if dataset_number in i]
#
#     y_path = glob.glob(data_folder + "y_data/" + "*csv")
#     y_files = [i.split('/')[-1] for i in y_path if dataset_number in i]
#     while True:
#         x_data = x_files[cnt]
#         y_data = y_files[cnt]
#
#         x_dataframe = pd.read_csv(data_folder + "x_data/" + x_data)
#         y_dataframe = pd.read_csv(data_folder + "y_data/" + y_data)
#         cnt += 1
#
#         print(x_dataframe)
#         print("------------------------------")
#         print(y_dataframe)
#
#         yield x_dataframe, y_dataframe

train_path = "datasets/not_encoded/train/"
x_train_file = "x_data/x_train_dataset_" + dataset_number + ".csv"
y_train_file = "y_data/y_train_dataset_" + dataset_number + ".csv"

val_path = "datasets/not_encoded/val/"
x_val_file = "x_data/x_val_dataset_" + dataset_number + ".csv"
y_val_file = "y_data/y_val_dataset_" + dataset_number + ".csv"


def preprocess_dataframes(path, file, isX):
    if isX:
        init_dataframe = pd.read_csv(path + file)

        onehot_dataframe = pd.get_dummies(init_dataframe.exercise_id, prefix="exercise_id")
        init_dataframe = init_dataframe.drop(init_dataframe.columns[[0, 2]], axis=1, inplace=True)
        merged_dataframe = onehot_dataframe.join(init_dataframe)

        all_columns = pd.DataFrame(columns=merged_dataframe.columns)

        return merged_dataframe, all_columns
    else:
        init_dataframe = pd.read_csv(path + file)
        onehot_dataframe = pd.get_dummies(init_dataframe.target_exercise_id, prefix="target_exercise_id")
        all_columns = pd.DataFrame(columns=onehot_dataframe.columns)

        return onehot_dataframe, all_columns


# x_train, x_columns = preprocess_dataframes(train_path, x_train_file, True)
# y_train, y_columns = preprocess_dataframes(train_path, y_train_file, False)
# x_val, x_val_columns = preprocess_dataframes(val_path, x_val_file, True)
# y_val, y_val_columns = preprocess_dataframes(val_path, y_val_file, False)


x_train = pd.read_csv(train_path + x_train_file) #Dataframe
y_train = pd.read_csv(train_path + y_train_file) #Dataframe
x_columns = pd.read_csv("datasets/columns/data/dataset_" + dataset_number + ".csv")
y_columns = pd.read_csv("datasets/columns/labels/dataset_" + dataset_number + ".csv")
train_indices = list(range(sum(1 for line in open(train_path + x_train_file))))  # correct index, 0 index is header always.
train_indices.pop(0)
batch_size = 2048

x_onehot_dataframe = pd.get_dummies(x_train.exercise_id, prefix="exercise_id")
x_train.drop(x_train.columns[[0, 2]], axis=1, inplace=True)
x_joined_dataframe = x_onehot_dataframe.join(x_train)
x_dataframe_to_return = x_joined_dataframe.combine_first(x_columns)
x_dataframe_to_return = x_dataframe_to_return.fillna(0)

y_onehot_dataframe = pd.get_dummies(y_train.target_exercise_id, prefix="target_exercise_id")
y_joined_dataframe = y_onehot_dataframe.join(y_train)
y_dataframe_to_return = y_joined_dataframe.combine_first(y_columns)
y_dataframe_to_return = y_dataframe_to_return.fillna(0)


# train_generator = DataGenerator(x_dataframe_to_return, y_dataframe_to_return, x_columns, y_columns, train_indices, batch_size)


x_val = pd.read_csv(val_path + x_val_file) #Dataframe
y_val = pd.read_csv(val_path + y_val_file) #Dataframe
val_indices = list(range(sum(1 for line in open(val_path + x_val_file))))  # correct index, 0 index is header always.
val_indices.pop(0)
batch_size = 2048

xv_onehot_dataframe = pd.get_dummies(x_val.exercise_id, prefix="exercise_id")
x_val.drop(x_val.columns[[0, 2]], axis=1, inplace=True)
xv_joined_dataframe = xv_onehot_dataframe.join(x_train)
xv_dataframe_to_return = xv_joined_dataframe.combine_first(x_columns)
xv_dataframe_to_return = xv_dataframe_to_return.fillna(0)

yv_onehot_dataframe = pd.get_dummies(y_val.target_exercise_id, prefix="target_exercise_id")
yv_joined_dataframe = yv_onehot_dataframe.join(y_val)
yv_dataframe_to_return = yv_joined_dataframe.combine_first(y_columns)
yv_dataframe_to_return = yv_dataframe_to_return.fillna(0)


# val_generator = DataGenerator(x_val, y_val, x_columns, y_columns, val_indices, batch_size)

# train_generator = DataGenerator(train_path + "x_data/",
#                                 train_path + "y_data/",
#                                 x_train_file,
#                                 y_train_file,
#                                 1024)
# val_generator = DataGenerator(val_path + "x_data/",
#                               val_path + "y_data/",
#                               x_val_file,
#                               y_val_file,
#                               1024)

model = keras.Sequential()
model.add(layers.Dense(300, activation="relu", input_shape=(len(x_columns.axes[1]), ), name="input_layer"))
model.add(layers.Dense(100, activation="relu", name="layer2"))
model.add(layers.Dense(len(y_columns.axes[1]), activation="sigmoid", name="output_layer"))

summery = model.summary()

print(summery)

history = model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error", "mean_squared_error", "accuracy"])
earlystopping = EarlyStopping(monitor="mean_absolute_error", patience=10, verbose=1, mode="auto")


# result = model.fit(train_generator, batch_size=16, epochs=10, validation_data=val_generator, callbacks=[earlystopping, plot_losses])
result = model.fit(x_dataframe_to_return, y_dataframe_to_return, batch_size=32, epochs=10, validation_data=(xv_dataframe_to_return, yv_dataframe_to_return), callbacks=[earlystopping, plot_losses])

