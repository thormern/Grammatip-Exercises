import pandas as pd
import sys
from sklearn import preprocessing

dataset_number = sys.argv[1]
dataset_chunk = sys.argv[2]

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


dataframe = pd.read_csv("datasets/reduced_datasets/full_length/" + dataset_number + ".csv")

print(dataframe)
print(dataframe.groupby("exercise_id").count())

full = pd.read_csv("user_data.csv")
print(full)
# counted_df = full.groupby("exercise_id").count()
# print("over 10000")
# print(counted_df[counted_df > 10000].count())


full["target_exercise_id"] = make_target_column(full)
index = full[full["target_exercise_id"] == 0].index
full.drop(index, inplace=True)
# print(dataframe)


full["user_id"] = preprocessing.LabelEncoder().fit_transform(full["user_id"])
full["teacher_id"] = preprocessing.LabelEncoder().fit_transform(full["teacher_id"])
full["date_completed"] = preprocessing.LabelEncoder().fit_transform(full["date_completed"])




# reduced10000 = full[full.groupby('exercise_id')['exercise_id'].transform('size') > 10000].copy()
#
# print(reduced10000)
# print(reduced10000.groupby("exercise_id").count())
#
# reduced10000.to_csv("datasets/reduced_datasets/full_length/over10000_new.csv")

reduced5000 = full[full.groupby('exercise_id')['exercise_id'].transform('size') > 5000].copy()

print(reduced5000)
print(reduced5000.groupby("exercise_id").count())

reduced5000.to_csv("datasets/reduced_datasets/full_length/over5000_new.csv")

# reduced1000 = full[full.groupby('exercise_id')['exercise_id'].transform('size') > 10000].copy()
#
# print(reduced1000)
# print(reduced1000.groupby("exercise_id").count())
#
# reduced1000.to_csv("datasets/reduced_datasets/full_length/over1000_new.csv")
#
#
# reduced500 = full[full.groupby('exercise_id')['exercise_id'].transform('size') > 5000].copy()
#
# print(reduced500)
# print(reduced500.groupby("exercise_id").count())
#
# reduced500.to_csv("datasets/reduced_datasets/full_length/over500_new.csv")