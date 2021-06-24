import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


t_over10000 = "datasets/reduced_datasets/not_encoded/over10000/train/x_train_dataset_0.csv"
t_over5000 = "datasets/reduced_datasets/not_encoded/train/over5000/"
t_over1000 = "datasets/reduced_datasets/not_encoded/train/over1000/"
t_over500 = "datasets/reduced_datasets/not_encoded/train/over500/"

v_over10000 = "datasets/reduced_datasets/not_encoded/val/over10000/"
v_over5000 = "datasets/reduced_datasets/not_encoded/val/over5000/"
v_over1000 = "datasets/reduced_datasets/not_encoded/val/over1000/"
v_over500 = "datasets/reduced_datasets/not_encoded/val/over500/"

te_over10000 = "datasets/reduced_datasets/not_encoded/test/over10000/"
te_over5000 = "datasets/reduced_datasets/not_encoded/test/over5000/"
te_over1000 = "datasets/reduced_datasets/not_encoded/test/over1000/"
te_over500 = "datasets/reduced_datasets/not_encoded/test/over500/"

full_over10000 = "datasets/reduced_datasets/full_length/over10000.csv"
full_over5000 = "datasets/reduced_datasets/full_length/over5000.csv"
full_over1000 = "datasets/reduced_datasets/full_length/over1000.csv"
full_over500 = "datasets/reduced_datasets/full_length/over500.csv"

over10000 = pd.read_csv(t_over10000)
# over5000 = pd.read_csv(full_over5000)
# over1000 = pd.read_csv(full_over1000)
# over500 = pd.read_csv(full_over500)

print(over10000.shape)
# print(over5000.shape)
# print(over1000.shape)
# print(over500.shape)

over10000_x_onehot = pd.get_dummies(over10000.exercise_id, prefix="exercise_id")
over10000_y_onehot = pd.get_dummies(over10000.target_exercise_id, prefix="target_exercise_id")

print(over10000_x_onehot.shape)
print(over10000_y_onehot.shape)



# over10000_onehot = pd.get_dummies(over10000.target_exercise_id, prefix="target_exercise_id")
#
#
# over10000_columns = pd.DataFrame(columns=over10000_onehot.columns)
# print(over10000_columns.shape)
#
#
#
# over5000_onehot = pd.get_dummies(over5000.target_exercise_id, prefix="target_exercise_id")
# print(over5000_onehot.shape)
#
# over5000_columns = pd.DataFrame(columns=over5000_onehot.columns)
# print(over5000_columns.shape)
#
#
# over1000_onehot = pd.get_dummies(over1000.target_exercise_id, prefix="target_exercise_id")
# print(over1000_onehot.shape)
#
# over1000_columns = pd.DataFrame(columns=over1000_onehot.columns)
# print(over1000_columns.shape)
#
# over500_onehot = pd.get_dummies(over500.target_exercise_id, prefix="target_exercise_id")
# print(over500_onehot.shape)
#
# over500_columns = pd.DataFrame(columns=over500_onehot.columns)
# print(over500_columns.shape)
#
#
# print("over10000: ", len(over10000_columns.axes[1]))
# print("over5000: ", len(over5000_columns.axes[1]))
# print("over1000: ", len(over1000_columns.axes[1]))
# print("over500: ", len(over500_columns.axes[1]))
#
#
# over10000_onehot_ex = pd.get_dummies(over10000.exercise_id, prefix="exercise_id")
# print(over10000_onehot_ex.shape)
# over10000.drop(over10000.columns[[0, 2, 6]], axis=1, inplace=True)
# over10000_joined = over10000_onehot_ex.join(over10000)
#
# over10000_columns = pd.DataFrame(columns=over10000_joined.columns)
# print(over10000_columns.shape)
#
#
# over5000 = pd.read_csv(full_over5000)
# over5000_onehot_ex = pd.get_dummies(over5000.exercise_id, prefix="exercise_id")
# print(over5000_onehot_ex.shape)
# over5000.drop(over5000.columns[[0, 2, 6]], axis=1, inplace=True)
# over5000_joined = over5000_onehot_ex.join(over5000)
#
# over5000_columns = pd.DataFrame(columns=over5000_joined.columns)
# print(over5000_columns.shape)
# over1000 = pd.read_csv(full_over1000)
# over1000_onehot_ex = pd.get_dummies(over1000.exercise_id, prefix="exercise_id")
# print(over1000_onehot_ex.shape)
# over1000.drop(over1000.columns[[0, 2, 6]], axis=1, inplace=True)
# over1000_joined = over1000_onehot_ex.join(over1000)
#
# over1000_columns = pd.DataFrame(columns=over1000_joined.columns)
# print(over1000_columns.shape)
# over500 = pd.read_csv(full_over500)
# over500_onehot_ex = pd.get_dummies(over500.exercise_id, prefix="exercise_id")
# print(over500_onehot_ex.shape)
# over500.drop(over500.columns[[0, 2, 6]], axis=1, inplace=True)
# over500_joined = over500_onehot_ex.join(over500)
#
# over500_columns = pd.DataFrame(columns=over500_joined.columns)
# print(over500_columns.shape)
#
#
# print("over10000: ", len(over10000_columns.axes[1]))
# print("over5000: ", len(over5000_columns.axes[1]))
# print("over1000: ", len(over1000_columns.axes[1]))
# print("over500: ", len(over500_columns.axes[1]))






# for i in range(10):





# for i in range(10):
#     x_train_file = "x_train_dataset_" + str(i) + ".csv"
#     y_train_file = "y_train_dataset_" + str(i) + ".csv"
#     x_val_file = "x_val_dataset_" + str(i) + ".csv"
#     y_val_file = "y_val_dataset_" + str(i) + ".csv"
#
#     dataset = "over10000"
#
#     x_train = "datasets/reduced_datasets/not_encoded/"+ dataset + "/train/" + x_train_file
#     y_train = "datasets/reduced_datasets/not_encoded/"+ dataset +"/train/" + y_train_file
#     x_val = "datasets/reduced_datasets/not_encoded/"+ dataset +"/val/" + x_val_file
#     y_val = "datasets/reduced_datasets/not_encoded/"+ dataset +"/val/" + y_val_file
#
#     x_train_dataframe = pd.read_csv(x_train)
#     x_train_onehot_dataframe = pd.get_dummies(x_train_dataframe.exercise_id, prefix="exercise_id")
#     x_train_dataframe.drop(x_train_dataframe.columns[[0, 2]], axis=1, inplace=True)
#     x_train_merged = x_train_onehot_dataframe.join(x_train_dataframe)
#     x_train_columns = pd.DataFrame(columns=x_train_merged.columns)
#     print("x_train_columns:" + str(len(x_train_columns.axes[1])))
#
#     x_val_dataframe = pd.read_csv(x_val)
#     x_val_onehot_dataframe = pd.get_dummies(x_val_dataframe.exercise_id, prefix="exercise_id")
#     x_val_dataframe.drop(x_val_dataframe.columns[[0, 2]], axis=1, inplace=True)
#     x_val_merged = x_val_onehot_dataframe.join(x_val_dataframe)
#     x_val_columns = pd.DataFrame(columns=x_val_merged.columns)
#     print("x_val_columns:" + str(len(x_val_columns.axes[1])))
#
#     all_x_columns = x_train_columns.merge(x_val_columns)
#     print("all_x_columns:" + str(len(all_x_columns.axes[1])))
#
#     y_train_dataframe = pd.read_csv(y_train)
#     y_train_onehot_dataframe = pd.get_dummies(y_train_dataframe.target_exercise_id, prefix="target_exercise_id")
#     y_train_merged = y_train_onehot_dataframe.join(y_train_dataframe)
#     y_train_columns = pd.DataFrame(columns=y_train_merged.columns)
#     print("y_train_columns:" + str(len(y_train_columns.axes[1])))
#
#     y_val_dataframe = pd.read_csv(y_val)
#     y_val_onehot_dataframe = pd.get_dummies(y_val_dataframe.target_exercise_id, prefix="target_exercise_id")
#     y_val_merged = y_val_onehot_dataframe.join(y_val_dataframe)
#     y_val_columns = pd.DataFrame(columns=y_val_merged.columns)
#     print("y_val_columns:" + str(len(y_val_columns.axes[1])))
#
#     all_y_columns = y_train_columns.merge(y_val_columns)
#     print("all_y_columns:" + str(len(all_y_columns.axes[1])))
#
#     all_x_columns.to_csv("datasets/reduced_datasets/columns/data/dataset_" + dataset + "_" + str(i) + ".csv")
#     all_y_columns.to_csv("datasets/reduced_datasets/columns/labels/dataset_" + dataset + "_" + str(i) + ".csv")
