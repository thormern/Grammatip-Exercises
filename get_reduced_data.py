import pandas as pd


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


dataframe = pd.read_csv("user_data.csv")


dataframe["target_exercise_id"] = make_target_column(dataframe)
index = dataframe[dataframe['target_exercise_id'] == 0].index
dataframe.drop(index, inplace=True)
over10000 = dataframe[dataframe.groupby('exercise_id')['exercise_id'].transform('size') > 10000]
over5000 = dataframe[dataframe.groupby('exercise_id')['exercise_id'].transform('size') > 5000]
over1000 = dataframe[dataframe.groupby('exercise_id')['exercise_id'].transform('size') > 1000]
over500 = dataframe[dataframe.groupby('exercise_id')['exercise_id'].transform('size') > 500]
print(dataframe)

print("--------------")
print("reduced")
print(over10000)
print(over5000)
print(over1000)
print(over500)
counted_df10000 = over10000.groupby("exercise_id").count()
counted_df5000 = over5000.groupby("exercise_id").count()
counted_df1000 = over1000.groupby("exercise_id").count()
counted_df500 = over500.groupby("exercise_id").count()
print(counted_df10000)
print(counted_df5000)
print(over1000)
print(over500)


over10000.to_csv("datasets/reduced_datasets/full_length/over10000.csv")
over5000.to_csv("datasets/reduced_datasets/full_length/over5000.csv")
over1000.to_csv("datasets/reduced_datasets/full_length/over1000.csv")
over500.to_csv("datasets/reduced_datasets/full_length/over500.csv")