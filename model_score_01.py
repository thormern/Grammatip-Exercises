import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, save_model, load_model
from TrainingPlot import TrainingPlot
from DataGenerator import DataGenerator
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
dataset_number = sys.argv[1]
dataset_chunk = sys.argv[2]

path = "datasets/reduced_datasets/full_length/" + dataset_number + ".csv"


init_df = pd.read_csv(path)

print(init_df)

score = init_df["result"].nunique()
n_users = init_df['user_id'].nunique()
print(score)
print(init_df["result"].describe())

ratings = init_df
ratings["user_id"] = ratings["user_id"].fillna(0)
ratings["exercise_id"] = ratings["exercise_id"].fillna(0)
ratings["teacher_id"] = ratings["teacher_id"].fillna(0)

ratings["result"] = ratings["result"].fillna(ratings["result"].mean())

ratings.drop(ratings.columns[[0,4,6]], axis=1, inplace=True)

print(ratings)

train_data, test_data = train_test_split(ratings, test_size=0.2)

train_data_matrix = train_data.to_numpy()
test_data_matrix = test_data.to_numpy()

print(train_data_matrix.shape)
print(test_data_matrix.shape)

user_correlation = 1 - pairwise_distances(train_data, metric="correlation")
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation[:5, :5])

item_correlation = 1 - pairwise_distances(train_data_matrix.T, metric="correlation")
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation[:5, :5])

print(user_correlation.shape, item_correlation.shape)

def predict(ratings, similarity, type="user"):
    if type == "user":
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == "item":
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

def rmse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))


user_prediction = predict(train_data_matrix, user_correlation, type="user")
item_prediction = predict(train_data_matrix, item_correlation, type="item")

print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

print('User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)))