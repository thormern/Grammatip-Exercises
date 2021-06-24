import tensorflow.keras as keras
import numpy as np
import os
import glob
import pandas as pd

from sklearn import preprocessing


class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_dataframe, y_dataframe, data_columns, label_columns, indices, batch_size, shuffle=True):
        self.x_dataframe = x_dataframe
        self.y_dataframe = y_dataframe
        self.data_columns = data_columns
        self.label_columns = label_columns
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        stop = (index + 1) * self.batch_size
        indices_to_use = self.indices[start:stop]
        data, labels = self.process_data(indices_to_use)

        data = data.to_numpy()
        labels = labels.to_numpy()

        return data, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def process_data(self, indices_to_use):
        print(indices_to_use)
        print("--------------------")
        print(len(self.x_dataframe))
        x_dataframe = self.x_dataframe.iloc[indices_to_use, :].copy()
        x_onehot_dataframe = pd.get_dummies(x_dataframe.exercise_id, prefix="exercise_id")
        x_dataframe.drop(x_dataframe.columns[[0, 2]], axis=1, inplace=True)
        x_joined_dataframe = x_onehot_dataframe.join(x_dataframe)
        x_dataframe_to_return = x_joined_dataframe.combine_first(self.data_columns)
        x_dataframe_to_return = x_dataframe_to_return.fillna(0)

        y_dataframe = self.y_dataframe.iloc[indices_to_use, :].copy()
        y_onehot_dataframe = pd.get_dummies(y_dataframe.target_exercise_id, prefix="target_exercise_id")
        y_joined_dataframe = y_onehot_dataframe.join(y_dataframe)
        y_dataframe_to_return = y_joined_dataframe.combine_first(self.label_columns)
        y_dataframe_to_return = y_dataframe_to_return.fillna(0)
        return x_dataframe_to_return, y_dataframe_to_return
        # return x_dataframe, y_dataframe

# class DataGenerator(keras.utils.Sequence):
#
#     def __init__(self, x_data_path, y_data_path, x_file, y_file, batch_size, shuffle=True):
#         self.x_data_path = x_data_path
#         self.y_data_path = y_data_path
#         self.x_file = x_file
#         self.y_file = y_file
#         self.batch_size = batch_size
#
#         self.indices = list(range(sum(1 for line in open(x_data_path + self.x_file))))  # correct index, 0 index is header always.
#         self.indices.pop(0)
#
#         tmp = pd.read_csv(self.x_data_path + self.x_file)
#         tmp["user_id"] = preprocessing.LabelEncoder().fit_transform(tmp["user_id"])
#         tmp["teacher_id"] = preprocessing.LabelEncoder().fit_transform(tmp["teacher_id"])
#         tmp["date_completed"] = preprocessing.LabelEncoder().fit_transform(tmp["date_completed"])
#
#         tmp2 = pd.get_dummies(tmp.exercise_id, prefix="exercise_id")
#
#         tmp.drop(tmp.columns[[0, 2]], axis=1, inplace=True)
#         self.data_columns = tmp2.join(tmp)
#         self.label_columns = pd.get_dummies(pd.read_csv(self.y_data_path + self.y_file).target_exercise_id, prefix="target_exercise_id")
#         self.shuffle = shuffle
#
#     def __len__(self):
#         return int(np.floor(len(self.indices) / self.batch_size))
#
#     def __getitem__(self, index):
#         start = index * self.batch_size
#         stop = (index + 1) * self.batch_size
#         indices_to_use = self.indices[start:stop]
#         # print(indices_to_use)
#         data, labels = self.process_files(indices_to_use)
#
#         data = data.to_numpy()
#         labels = labels.to_numpy()
#
#         return data, labels
#
#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)
#
#     def process_files(self, indices):
#         # print("Do process_files start?")
#         # labelEngcoder = preprocessing.LabelEncoder()
#         # rows_to_skip = self.rows_to_skip(indices)
#         # x_dataframe = pd.read_csv(self.x_data_path + self.x_file, skiprows=rows_to_skip)
#
#         x_dataframe = self.data_columns.iloc[indices, :]
#
#         # lambda x: x not in indices
#         # print("No label encoding")
#         #
#         # x_dataframe['user_id'] = labelEngcoder.fit_transform(x_dataframe['user_id'])
#         # x_dataframe['teacher_id'] = labelEngcoder.fit_transform(x_dataframe['teacher_id'])
#         # x_dataframe['date_completed'] = labelEngcoder.fit_transform(x_dataframe['date_completed'])
#         # print("label encoded, but not onehot encoded")
#         # print(x_dataframe)
#         # print("onehot encoded")
#         # x_dataframe = pd.get_dummies(x_dataframe.exercise_id, prefix="exercise_id")
#         # print(x_dataframe)
#         # y_dataframe = pd.read_csv(self.y_data_path + self.y_file, skiprows=rows_to_skip)
#
#         # y_dataframe = pd.get_dummies(y_dataframe.target_exercise_id, prefix="target_exercise_id")
#
#         y_dataframe = self.label_columns.iloc[indices, :]
#
#         return x_dataframe, y_dataframe
#
#     def rows_to_skip(self, rows_to_keep):
#         rows_to_skip = [x for x in self.indices if x not in rows_to_keep]
#         return rows_to_skip
