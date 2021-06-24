import pandas as pd
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

file = "over10000"
path = "datasets/reduced_datasets/"
datafile = "full_length/" + file + ".csv"
initdataframe = pd.read_csv(path+datafile, chunksize=37750)
# initdataframe.head()
cnt = 0
for chunk in initdataframe:
    if cnt == 10: continue
    print("chunk " + str(cnt))
    # print("Initial Dataframe from CSV")
    # print(chunk)
    chunk["user_id"] = preprocessing.LabelEncoder().fit_transform(chunk["user_id"])
    chunk["teacher_id"] = preprocessing.LabelEncoder().fit_transform(chunk["teacher_id"])
    chunk["date_completed"] = preprocessing.LabelEncoder().fit_transform(chunk["date_completed"])

    # chunk["target_exercise_id"] = make_target_column(chunk)
    # print("Dataframe with targets")
    # print(chunk)


    # index = chunk[chunk['target_exercise_id'] == 0].index
    # chunk.drop(index, inplace=True)

    # dataframe.to_csv("datasets/cleaned_dataset_" + str(cnt) + ".csv")

    y_data = chunk.target_exercise_id
    x_data = chunk.drop("target_exercise_id", axis=1)

    X_train, X_val_test, Y_train, Y_val_test = train_test_split(x_data, y_data, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

    chunk.to_csv("datasets/not_encoded/user_data_" + str(cnt) + ".csv")


    x_data.to_csv("datasets/reduced_datasets/not_encoded/over10000/x_data_" + str(cnt) + ".csv")
    y_data.to_csv("datasets/reduced_datasets/not_encoded/over10000/y_data_" + str(cnt) + ".csv")

    X_train.to_csv(path + "/not_encoded/" + file + "/train/x_train_dataset_" + str(cnt) + ".csv")
    Y_train.to_csv(path + "/not_encoded/" + file + "/train/y_train_dataset_" + str(cnt) + ".csv")
    X_test.to_csv(path + "/not_encoded/" + file + "/test/x_test_dataset_" + str(cnt) + ".csv")
    Y_test.to_csv(path + "/not_encoded/" + file + "/test/y_test_dataset_" + str(cnt) + ".csv")
    X_val.to_csv(path + "/not_encoded/" + file + "/val/x_val_dataset_" + str(cnt) + ".csv")
    Y_val.to_csv(path + "/not_encoded/" + file + "/val/y_val_dataset_" + str(cnt) + ".csv")

    cnt += 1




