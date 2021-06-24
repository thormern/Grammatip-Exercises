import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


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
    i = 0
    for row in current_dataframe['teacher_id']:
        if prev_row is not None:
            if prev_row != row:
                taget_column[i] = 0
                prev_row = row
            else:
                prev_row = row
        else:
            prev_row = row
        i += 1

    return taget_column


user_data = "user_data.csv"

dataframe = pd.read_csv(user_data)

counted_df = dataframe.groupby("exercise_id").count()
#
# print(dataframe.groupby("exercise_id").count())
# print(dataframe.groupby("teacher_id").count())
# print(dataframe.groupby("user_id").count())
# print(counted_df)

#
# print("over 50")
# print(counted_df[counted_df > 50].count())
#
# print("over 100")
# print(counted_df[counted_df > 100].count())
#
# print("over 200")
# print(counted_df[counted_df > 200].count())

# print("over 500")
# print(counted_df[counted_df > 500].count())
# print("over 1000")
# print(counted_df[counted_df > 1000].count())
# print("over 2000")
# print(counted_df[counted_df > 2000].count())
#
# print("over 5000")
# print(counted_df[counted_df > 5000].count())


# print("over 10000")
# print(counted_df[counted_df > 10000].count())


dataframe["target_exercise_id"] = make_target_column(dataframe)
index = dataframe[dataframe["target_exercise_id"] == 0].index
dataframe.drop(index, inplace=True)
# print(dataframe)


dataframe["user_id"] = preprocessing.LabelEncoder().fit_transform(dataframe["user_id"])
dataframe["teacher_id"] = preprocessing.LabelEncoder().fit_transform(dataframe["teacher_id"])
dataframe["date_completed"] = preprocessing.LabelEncoder().fit_transform(dataframe["date_completed"])
# dataframe["excercise_id"] = preprocessing.LabelEncoder().fit_transform(dataframe["excercise_id"])
# dataframe["resuæt"] = preprocessing.LabelEncoder().fit_transform(dataframe["result"])
# print(dataframe)
# dataframe.to_csv("datasets/user_data_cleaned.csv")

size = [10000, 5000, 1000, 500]


for i in size:
    reduced_dataframe = dataframe[dataframe.groupby('exercise_id')['exercise_id'].transform('size') > i].copy()
    print(reduced_dataframe)

    ex_id = reduced_dataframe["exercise_id"].tolist()
    target = reduced_dataframe["target_exercise_id"].tolist()
    to_exclude = (set(ex_id) ^ set(target)) & set(target)
    print(len(to_exclude))

    reduced_dataframe = reduced_dataframe[~reduced_dataframe.target_exercise_id.isin(list(to_exclude))]

    index = reduced_dataframe[reduced_dataframe["target_exercise_id"] == 0].index
    reduced_dataframe.drop(index, inplace=True)
    print(reduced_dataframe)

    x_onehot = pd.get_dummies(reduced_dataframe.exercise_id, prefix="exercise_id")
    y_onehot = pd.get_dummies(reduced_dataframe.target_exercise_id, prefix="target_exercise_id")

    reduced_dataframe.to_csv("datasets/reduced_datasets/full_length/over" + str(i) + ".csv")
    chunksize = int(np.floor(reduced_dataframe.shape[0] / 10))
    tmp_dataframe = reduced_dataframe.drop(reduced_dataframe.columns[[1, 5]], axis=1)
    x_joined = x_onehot.join(tmp_dataframe)

    x_columns = pd.DataFrame(columns=x_joined.columns)
    y_columns = pd.DataFrame(columns=y_onehot.columns)

    print(x_columns.shape)
    print(y_columns.shape)

    x_columns.to_csv("datasets/reduced_datasets/columns/data/over" + str(i) + ".csv")
    y_columns.to_csv("datasets/reduced_datasets/columns/labels/over" + str(i) + ".csv")
    if i == 10000 or i == 5000:
        reduced_dataframe.drop(reduced_dataframe.columns[[5]], axis=1, inplace=True)
        X_train, X_val_test, Y_train, Y_val_test = train_test_split(x_joined, y_onehot, test_size=0.3)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

        X_train.to_csv("datasets/reduced_datasets/encoded/over" + str(i) + "/train/x_dataset_all.csv")
        Y_train.to_csv("datasets/reduced_datasets/encoded/over" + str(i) + "/train/y_dataset_all.csv")
        X_test.to_csv("datasets/reduced_datasets/encoded/over" + str(i) + "/test/x_dataset_all.csv")
        Y_test.to_csv("datasets/reduced_datasets/encoded/over" + str(i) + "/test/y_dataset_all.csv")
        X_val.to_csv("datasets/reduced_datasets/encoded/over" + str(i) + "/val/x_dataset_all.csv")
        Y_val.to_csv("datasets/reduced_datasets/encoded/over" + str(i) + "/val/y_dataset_all.csv")

        X_train, X_val_test, Y_train, Y_val_test = train_test_split(reduced_dataframe, y_onehot, test_size=0.3)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

        X_train.to_csv("datasets/reduced_datasets/not_encoded/over" + str(i) + "/train/x_dataset_all.csv")
        Y_train.to_csv("datasets/reduced_datasets/not_encoded/over" + str(i) + "/train/y_dataset_all.csv")
        X_test.to_csv("datasets/reduced_datasets/not_encoded/over" + str(i) + "/test/x_dataset_all.csv")
        Y_test.to_csv("datasets/reduced_datasets/not_encoded/over" + str(i) + "/test/y_dataset_all.csv")
        X_val.to_csv("datasets/reduced_datasets/not_encoded/over" + str(i) + "/val/x_dataset_all.csv")
        Y_val.to_csv("datasets/reduced_datasets/not_encoded/over" + str(i) + "/val/y_dataset_all.csv")

    dataframe_to_chunk = pd.read_csv("datasets/reduced_datasets/full_length/over" + str(i) + ".csv", chunksize=chunksize)

    chunk_number = 0
    # for chunk in dataframe_to_chunk:
    #     if chunk_number == 10: continue
    #     print("over" + str(i) + ", chunk: " + str(chunk_number))
    #
    #
    #     y_data_onehot = pd.get_dummies(chunk.target_exercise_id, prefix="target_exercise_id")
    #     y_data_final = y_data_onehot.combine_first(y_columns)
    #     y_data_final = y_data_final.fillna(int(0))
    #
    #     x_data = chunk.drop("target_exercise_id", axis=1)
    #     x_data_onehot = pd.get_dummies(x_data.exercise_id, prefix="exercise_id")
    #     x_date_dropped = x_data.drop("exercise_id", axis=1)
    #     x_data_final = x_data_onehot.combine_first(x_date_dropped)
    #     x_data_final = x_data_final.fillna(int(0))
    #
    #     X_train, X_val_test, Y_train, Y_val_test = train_test_split(x_data_final, y_data_final, test_size=0.3)
    #     X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)
    #
    #     X_train.to_csv("datasets/reduced_datasets/encoded/over" + str(i) + "/train/x_dataset_" + str(chunk_number) + ".csv")
    #     Y_train.to_csv("datasets/reduced_datasets/encoded/over" + str(i) + "/train/y_dataset_" + str(chunk_number) + ".csv")
    #     X_test.to_csv("datasets/reduced_datasets/encoded/over" + str(i) + "/test/x_dataset_" + str(chunk_number) + ".csv")
    #     Y_test.to_csv("datasets/reduced_datasets/encoded/over" + str(i) + "/test/y_dataset_" + str(chunk_number) + ".csv")
    #     X_val.to_csv("datasets/reduced_datasets/encoded/over" + str(i) + "/val/x_dataset_" + str(chunk_number) + ".csv")
    #     Y_val.to_csv("datasets/reduced_datasets/encoded/over" + str(i) + "/val/y_dataset_" + str(chunk_number) + ".csv")
    #
    #     X_train, X_val_test, Y_train, Y_val_test = train_test_split(x_data, y_data_final, test_size=0.3)
    #     X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)
    #
    #     X_train.to_csv("datasets/reduced_datasets/not_encoded/over" + str(i) + "/train/x_dataset_" + str(chunk_number) + ".csv")
    #     Y_train.to_csv("datasets/reduced_datasets/not_encoded/over" + str(i) + "/train/y_dataset_" + str(chunk_number) + ".csv")
    #     X_test.to_csv("datasets/reduced_datasets/not_encoded/over" + str(i) + "/test/x_dataset_" + str(chunk_number) + ".csv")
    #     Y_test.to_csv("datasets/reduced_datasets/not_encoded/over" + str(i) + "/test/y_dataset_" + str(chunk_number) + ".csv")
    #     X_val.to_csv("datasets/reduced_datasets/not_encoded/over" + str(i) + "/val/x_dataset_" + str(chunk_number) + ".csv")
    #     Y_val.to_csv("datasets/reduced_datasets/not_encoded/over" + str(i) + "/val/y_dataset_" + str(chunk_number) + ".csv")
    #
    #
    #     chunk_number += 1



# def preprocess_dataframes(train_path, train_file, val_path, val_file, isX):
#     if isX:
#         train_dataframe = pd.read_csv(train_path + train_file) # load training file
#         train_onehot_dataframe = pd.get_dummies(train_dataframe.exercise_id, prefix="exercise_id") # onehot encoding
#         train_dataframe.drop(train_dataframe.columns[[0, 2]], axis=1, inplace=True) # dropping random index column and exercise column
#         train_merged_dataframe = train_onehot_dataframe.join(train_dataframe) # merging onehot with user_id, teacher_id etc.
#         train_all_columns = pd.DataFrame(columns=train_merged_dataframe.columns) # all columns in training data
#
#         val_dataframe = pd.read_csv(val_path + val_file) # load val file
#         val_onehot_dataframe = pd.get_dummies(val_dataframe.exercise_id, prefix="exercise_id") # onehot encoding val data
#         val_dataframe.drop(val_dataframe.columns[[0, 2]], axis=1, inplace=True)
#         val_merged_dataframe = val_onehot_dataframe.join(val_dataframe)
#         val_all_columns = pd.DataFrame(columns=val_onehot_dataframe.columns) # all columns in val data
#
#         all_columns = train_all_columns.merge(val_all_columns) # all columns from training and val
#
#         train_final_dataframe = train_merged_dataframe.combine_first(all_columns) # inserting all the data with the correct number of columns
#         train_final_dataframe = train_final_dataframe.fillna(0) # replace NaN values with 0
#
#         val_final_dataframe = val_merged_dataframe.combine_first(all_columns)
#         val_final_dataframe = val_final_dataframe.fillna(0)
#
#         return train_final_dataframe, val_final_dataframe
#     else:
#         train_dataframe = pd.read_csv(train_path + train_file)
#         train_onehot_dataframe = pd.get_dummies(train_dataframe.target_exercise_id, prefix="target_exercise_id")
#         train_all_columns = pd.DataFrame(columns=train_onehot_dataframe.columns)
#
#         val_dataframe = pd.read_csv(train_path + train_file)
#         val_onehot_dataframe = pd.get_dummies(val_dataframe.target_exercise_id, prefix="target_exercise_id")
#         val_all_columns = pd.DataFrame(columns=val_onehot_dataframe.columns)
#
#         all_columns = train_all_columns.merge(val_all_columns)
#
#         train_final_dataframe = train_onehot_dataframe.combine_first(all_columns)  # inserting all the data with the correct number of columns
#         train_final_dataframe = train_final_dataframe.fillna(0)  # replace NaN values with 0
#
#         val_final_dataframe = val_onehot_dataframe.combine_first(all_columns)
#         val_final_dataframe = val_final_dataframe.fillna(0)
#
#         return train_final_dataframe, val_final_dataframe


# x_train, x_val = preprocess_dataframes(xdata, xfile, xvaldata, xvalfile, True)
# y_train, y_val = preprocess_dataframes(ydata, yfile, yvaldata, yvalfile, False)
# print(len(x_train.axes[1]))
# print(len(x_val.axes[1]))
# print("----------------")
# print(len(y_train.axes[1]))
# print(len(y_val.axes[1]))


# tmp = pd.read_csv(xdata + xfile)
#
# index = list(tmp.index.values)
#
# print(len(index))
#
# for i in range(10):
#     print(index[i])


# x_train = pd.read_csv(xdata + xfile) #Dataframe
# y_train = pd.read_csv(ydata + yfile) #Dataframe
# x_columns = pd.read_csv("datasets/columns/data/dataset_" + dataset_number + ".csv")
# y_columns = pd.read_csv("datasets/columns/labels/dataset_" + dataset_number + ".csv")
# train_indices = list(range(sum(1 for line in open(xdata + xfile))))  # correct index, 0 index is header always.
# train_indices.pop(0)
# batch_size = 1024
#
# train_generator = DataGenerator(x_train, y_train, x_columns, y_columns, train_indices, batch_size)
# cnt = 0
# for x, y in train_generator:
#     cnt += 1
