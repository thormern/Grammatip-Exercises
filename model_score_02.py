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

n_score = init_df["result"].nunique()
n_users = init_df['user_id'].nunique()
n_exercises = init_df["exercise_id"].nunique()
n_teachers = init_df["teacher_id"].nunique()


print(n_score)
print(init_df["result"].describe())

ratings = init_df
ratings["user_id"] = ratings["user_id"].fillna(0)
ratings["exercise_id"] = ratings["exercise_id"].fillna(0)
ratings["teacher_id"] = ratings["teacher_id"].fillna(0)


min_score = min(ratings["result"])
max_score = max(ratings["result"])


ratings.drop(ratings.columns[[0,4,6]], axis=1, inplace=True)
scaler = MinMaxScaler()

scaled_ratings = pd.DataFrame(scaler.fit_transform(ratings))
scaled_ratings.columns = init_df.columns
print(ratings)
print(scaled_ratings)


X = scaled_ratings.drop("result", axis=1)
y = scaled_ratings["result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train_array = [X_train["user_id"], X_train["exercise_id"], X_train["teacher_id"]]
X_test_array = [X_test["user_id"], X_test["exercise_id"], X_test["teacher_id"]]



print(X)


n_factors = 20
class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors

    def __call__(self, x):
        x = keras.layers.Embedding(self.n_items, self.n_factors, embeddings_initializer="he_normal", embeddings_regularizer=keras.regularizers.l2(1e-6))(x)
        x = keras.layers.Reshape((self.n_factors,))(x)

        return x

def recommenderNet(n_users, n_ex, n_teachers, n_factors, min_score, max_score):
    user = keras.layers.Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors)(user)

    ex = keras.layers.Input(shape=(1,))
    e = EmbeddingLayer(n_ex, n_factors)(ex)

    teacher = keras.layers.Input(shape=(1,))
    t = EmbeddingLayer(n_teachers, n_factors)(teacher)

    x = keras.layers.Concatenate()([u, e])
    x = keras.layers.Concatenate()([x, t])
    x = keras.layers.Dropout(0.05)(x)
    x = keras.layers.Dense(10, kernel_initializer="he_normal")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(15, kernel_initializer="he_normal")(x)
    x = keras.layers.Activation("sigmoid")(x)
    x = keras.layers.Lambda(lambda x: x * (max_score - min_score) + min_score)(x)

    model = keras.models.Model(inputs=[user, ex, teacher], outputs=x)# , teacher
    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(loss="mean_squared_error", optimizer=adam, metrics=["accuracy"])

    return model

earlystopping = EarlyStopping(monitor="loss", patience=20, verbose=1, mode="auto")
loss = "output/model_score_2" + dataset_chunk + "_loss.jpg"
accuracy = "output/model_score_2" + dataset_chunk + "_accuracy.jpg"

plot_losses = TrainingPlot(loss, True)
plot_accuracy = TrainingPlot(accuracy, False)


model = recommenderNet(n_users, n_exercises, n_teachers, n_factors, min_score, max_score)
summery = model.summary()
print(summery)

history = model.fit(X_train_array, y_train, batch_size=64, epochs=200, verbose=1, validation_data=(X_test_array, y_test), callbacks=[earlystopping, plot_losses, plot_accuracy])