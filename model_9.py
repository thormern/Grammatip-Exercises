import glob
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

# from keras.models import Model
# from keras.layers import Input, Reshape, Dot
# from keras.layers.embeddings import Embedding
# from keras.optimizers import Adam
# from keras.regularizers import l2


dataset_number = sys.argv[1]
dataset_chunk = sys.argv[2]

loss = "output/reduced_datasets/" + dataset_number + "/model_9_" + dataset_chunk + "_loss.jpg"
accuracy = "output/reduced_datasets/" + dataset_number + "/model_9_" + dataset_chunk + "_accuracy.jpg"
plot_losses = TrainingPlot(loss, True)

plot_accuracy = TrainingPlot(accuracy, False)


#using over10000 set
dataframe = pd.read_csv("datasets/reduced_datasets/full_length/" + dataset_number + ".csv")


scalar = MinMaxScaler()

y_data = pd.get_dummies(dataframe.target_exercise_id, prefix="target_exercise_id")
# normalized_df = scalar.fit_transform(dataframe)
# normalized_df.columns = dataframe.columns
# y_data = normalized_df["target_exercise_id"]

x_data = dataframe.drop("target_exercise_id", axis=1)
x_data.drop(x_data.columns[[0]], axis=1, inplace=True)
print(x_data)

normalized_x = scalar.fit_transform(x_data)
normalized_x_df = pd.DataFrame(normalized_x)
normalized_x_df.columns = x_data.columns

n_users = dataframe['user_id'].nunique()
n_ex = dataframe['exercise_id'].nunique()
n_teachers = dataframe['teacher_id'].nunique()
n_factors = 50

min_score = min(dataframe["result"])
max_score = max(dataframe["result"])
print("number of users: ", n_users)
print("number of exercises: ", n_ex)
print("number of teachers: ", n_teachers)
print("lowest score: ", min_score)
print("highest score: ", max_score)


X_train, X_val_test, Y_train, Y_val_test = train_test_split(normalized_x_df, y_data, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

X_train_array = [X_train["user_id"], X_train["exercise_id"], X_train["teacher_id"]]
X_val_array = [X_val["user_id"], X_val["exercise_id"], X_val["teacher_id"]]
X_test_array = [X_test["user_id"], X_test["exercise_id"], X_test["teacher_id"]]

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
#, embeddings_regularizer=12(1e-6)

class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors

    def __call__(self, x):
        x = keras.layers.Embedding(self.n_items, self.n_factors, embeddings_initializer="he_normal")(x)
        x = keras.layers.Reshape((self.n_factors,))(x)

        return x


def recommenderV1(n_users, n_ex, n_teachers, n_factors, min_score, max_score):
    user = keras.layers.Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors)(user)
    ub = EmbeddingLayer(n_users, 1)(user)

    ex = keras.layers.Input(shape=(1,))
    e = EmbeddingLayer(n_ex, n_factors)(ex)
    eb = EmbeddingLayer(n_ex, 1)(ex)

    teacher = keras.layers.Input(shape=(1,))
    t = EmbeddingLayer(n_teachers, n_factors)(teacher)
    tb = EmbeddingLayer(n_teachers, 1)(teacher)

    # teacher = keras.layers.Input(shape=(1,))
    # t = keras.layers.Embedding(n_teachers, n_factors, embeddings_initializer="he_normal")(ex)
    # t = keras.layers.Reshape((n_factors,))(t)

    x1 = keras.layers.Dot(axes=1)([u, e])
    x2 = keras.layers.Dot(axes=1)([u, t])
    x = keras.layers.Dot(axes=1)([x1, x2])
    x = keras.layers.Add()([x, ub, eb, tb])
    x = keras.layers.Lambda(lambda x: x * (max_score- min_score) + min_score)(x)

    model = keras.models.Model(inputs=[user, ex, teacher], outputs=x)
    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(loss="mean_squared_error", optimizer=adam, metrics=["accuracy"])

    return model

def recommenderV2(n_users, n_ex, n_teachers, n_factors, min_score, max_score):
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

    model = keras.models.Model(inputs=[user, ex, teacher], outputs=x)
    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(loss="mean_squared_error", optimizer=adam, metrics=["accuracy"])

    return model

model = recommenderV2(n_users, n_ex, n_teachers, n_factors, min_score, max_score)
summary = model.summary()
print(summary)


earlystopping = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
for x in X_test_array:
    print(x.shape)
print(Y_train.shape)
history = model.fit(x=X_train_array, y=Y_train, batch_size=64, epochs=5,
                    verbose=1, validation_data=(X_val_array, Y_val), callbacks=[plot_losses, plot_accuracy, earlystopping])



# Generate generalization metrics
score = model.evaluate(X_test_array, Y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

save_path = "saved_models/" + dataset_number + "/dateset_" + dataset_chunk + "_model"
save_model(model, save_path)


samples_to_predict = X_test_array
#
predictions = model.predict(samples_to_predict)
print(predictions)


pred_df = pd.DataFrame(predictions)
pred_df.to_csv("output/" + dataset_number + "_" + dataset_chunk + "model_9_predictions.csv")
test_labels = pd.DataFrame(Y_test)
test_labels.to_csv("output/" + dataset_number + "_" + dataset_chunk + "model_9_predictions_labels.csv")