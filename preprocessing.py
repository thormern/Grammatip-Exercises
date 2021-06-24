import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np

def make_target_column(current_dataframe):
    taget_column = []

    prev_row = None

    for row in current_dataframe['exercise_id']:
        if prev_row is not None:
            prev_row = row
            taget_column.append(prev_row)
        else:
            prev_row = row
    taget_column.append(0)

    prev_row = None
    row_count = 0
    i = 0
    for row in current_dataframe['teacher_id']:
        if prev_row is not None:
            if prev_row != row:
                taget_column[row_count] = 0
            else:
                prev_row = row
        else:
            prev_row = row
        i += 1

    return taget_column

initdataframe = pd.read_csv("user_data.csv", chunksize=458402, skiprows=range(1,4125618))#2800000)
# initdataframe.head()
cnt = 9
for chunk in initdataframe:
    print("chunk " + str(cnt))
    # print("Initial Dataframe from CSV")
    # print(chunk)

    dataframe_exercise_id = pd.get_dummies(chunk.exercise_id, prefix='exercise_id')
    # print("OneHot Encoding for exercise_id")
    # print(dataframe_exercise_id)

    chunk["target_exercise_id"] = make_target_column(chunk)
    # print("Dataframe with targets")
    # print(chunk)

    dataframe_target_exercise_id = pd.get_dummies(chunk.target_exercise_id, prefix='target_exercise_id')
    # print("OneHot Encoding for target_exercise_id")
    # print(dataframe_target_exercise_id)

    dataframe = dataframe_exercise_id.join(dataframe_target_exercise_id)
    mergeddataframe = dataframe.join(chunk)
    # print("Merged Dataframe")
    # print(mergeddataframe)

    ydataframe = dataframe_target_exercise_id.join(chunk["target_exercise_id"])
    # print("ydataframe")
    # print(ydataframe)  # -> Y values

    xdataframe = dataframe_exercise_id.join(chunk)
    # print("xdataframe")
    # print(xdataframe)  # -> X values

    index = mergeddataframe[mergeddataframe['target_exercise_id'] == 0].index
    mergeddataframe.drop(index, inplace=True)
    ydataframe.drop(index, inplace=True)
    # ydataframe.drop("target_exercise_id_0", axis=1)
    xdataframe.drop(index, inplace=True)
    # print("Cleaned Dataframes")
    # print(mergeddataframe)
    # print(ydataframe)
    # print(xdataframe)

    # dataframe.to_csv("datasets/cleaned_dataset_" + str(cnt) + ".csv")

    X_train, X_val_test, Y_train, Y_val_test = train_test_split(xdataframe, ydataframe, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

    X_train.to_csv("datasets/train/x_data/x_train_dataset_" + str(cnt) + ".csv")
    Y_train.to_csv("datasets/train/y_data/y_train_dataset_" + str(cnt) + ".csv")
    X_test.to_csv("datasets/test/x_data/x_test_dataset_" + str(cnt) + ".csv")
    Y_test.to_csv("datasets/test/y_data/y_test_dataset_" + str(cnt) + ".csv")
    X_val.to_csv("datasets/val/x_data/x_val_dataset_" + str(cnt) + ".csv")
    Y_val.to_csv("datasets/val/y_data/y_val_dataset_" + str(cnt) + ".csv")

    cnt += 1


    # X_train_chunks = split(X_train, chunksize=46)
    # Y_train_chunks = split(Y_train, chunksize=46)
    # X_val_chunks = split(X_val, chunksize=46)
    # Y_val_chunks = split(Y_val, chunksize=46)
    # X_test_chunks = split(X_test, chunksize=46)
    # Y_test_chunks = split(Y_test, chunksize=46)
    #
    #
    # # X_train_chunks = pd.read_table(X_train, chunksize=7000)
    # # Y_train_chunks = pd.read_table(Y_train, chunksize=7000)
    # # X_val_chunks = pd.read_table(X_val, chunksize=1500)
    # # Y_val_chunks = pd.read_table(Y_val, chunksize=1500)
    # # X_test_chunks = pd.read_table(X_test, chunksize=1500)
    # # Y_test_chunks = pd.read_table(Y_test, chunksize=1500)
    #
    # i = 0
    #
    # for x_chunk in X_train_chunks:
    #     if i < 10:
    #         x_chunk.to_csv("datasets/train/x_data/x_train_dataset_" + str(cnt) + "_chunk_0" + str(i) + ".csv")
    #     else:
    #         x_chunk.to_csv("datasets/train/x_data/x_train_dataset_" + str(cnt) + "_chunk_" + str(i) + ".csv")
    #     i += 1
    #
    # i = 0
    #
    # for y_chunk in Y_train_chunks:
    #     if i < 10:
    #         y_chunk.to_csv("datasets/train/y_data/y_train_dataset_" + str(cnt) + "_chunk_0" + str(i) + ".csv")
    #     else:
    #         y_chunk.to_csv("datasets/train/y_data/y_train_dataset_" + str(cnt) + "_chunk_" + str(i) + ".csv")
    #     i += 1
    #
    # i = 0
    #
    # for x_chunk in X_val_chunks:
    #     if i < 10:
    #         x_chunk.to_csv("datasets/val/x_data/x_val_dataset_" + str(cnt) + "_chunk_0" + str(i) + ".csv")
    #     else:
    #         x_chunk.to_csv("datasets/val/x_data/x_val_dataset_" + str(cnt) + "_chunk_" + str(i) + ".csv")
    #     i += 1
    #
    # i = 0
    #
    # for y_chunk in Y_val_chunks:
    #     if i < 10:
    #         y_chunk.to_csv("datasets/val/y_data/x_val_dataset_" + str(cnt) + "_chunk_0" + str(i) + ".csv")
    #     else:
    #         y_chunk.to_csv("datasets/val/y_data/x_val_dataset_" + str(cnt) + "_chunk_" + str(i) + ".csv")
    #     i += 1
    #
    # i = 0
    #
    # for x_chunk in X_test_chunks:
    #     if i < 10:
    #         x_chunk.to_csv("datasets/test/x_data/x_test_dataset_" + str(cnt) + "_chunk_0" + str(i) + ".csv")
    #     else:
    #         x_chunk.to_csv("datasets/test/x_data/x_test_dataset_" + str(cnt) + "_chunk_" + str(i) + ".csv")
    #     i += 1
    #
    # i = 0
    #
    # for y_chunk in Y_test_chunks:
    #     if i < 10:
    #         y_chunk.to_csv("datasets/test/y_data/y_test_dataset_" + str(cnt) + "_chunk_0" + str(i) + ".csv")
    #     else:
    #         y_chunk.to_csv("datasets/test/y_data/y_test_dataset_" + str(cnt) + "_chunk_" + str(i) + ".csv")
    #     i += 1



