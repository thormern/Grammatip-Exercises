import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping

from TrainingPlot import TrainingPlot








# Get Dataset

dataset_number = sys.argv[1]
dataset_chunk = sys.argv[2]

dataframe = pd.read_csv("datasets/reduced_datasets/full_length/" + dataset_number + ".csv")

scalar = MinMaxScaler()

normalized_df = pd.DataFrame(scalar.fit_transform(dataframe))
normalized_df.columns = dataframe.columns

y_data = normalized_df["target_exercise_id"]
print(y_data)
x_data = normalized_df.drop("target_exercise_id", axis=1)
x_data.drop(x_data.columns[[0]], axis=1, inplace=True)
print(x_data)

X_train, X_val_test, Y_train, Y_val_test = train_test_split(x_data, y_data, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

X_train_array = [X_train["user_id"], X_train["exercise_id"], X_train["teacher_id"], X_train["result"]]
X_val_array = [X_val["user_id"], X_val["exercise_id"], X_val["teacher_id"], X_val["result"]]
X_test_array = [X_test["user_id"], X_test["exercise_id"], X_test["teacher_id"], X_test["result"]]

#
n_users = dataframe['user_id'].nunique()
n_exercises = dataframe['exercise_id'].nunique()
n_teachers = dataframe['teacher_id'].nunique()
n_results = dataframe['result'].nunique()
n_factors = 50


class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors

    def __call__(self, x):
        x = keras.layers.Embedding(self.n_items, self.n_factors, embeddings_initializer="he_normal")(x)
        x = keras.layers.Reshape((self.n_factors,))(x)

        return x

# The Model

user_input = keras.layers.Input(shape=(1,))
user_side = EmbeddingLayer(n_users, n_factors)(user_input)


exercise_input = keras.layers.Input(shape=(1,))
exercise_side = EmbeddingLayer(n_exercises, n_factors)(exercise_input)

teacher_input = keras.layers.Input(shape=(1,))
teacher_side = EmbeddingLayer(n_teachers, n_factors)(teacher_input)

result_input = keras.layers.Input(shape=(1,))
result_side = EmbeddingLayer(n_results, n_factors)(result_input)

dot_user_exercise = keras.layers.dot([user_side, exercise_side], axes=1)
dot_teacher_result = keras.layers.dot([teacher_side, result_side], axes=1)

dot_all = keras.layers.dot([dot_user_exercise, dot_teacher_result], axes=1)

output = keras.layers.Dense(1, activation="sigmoid")(dot_all)

model = keras.models.Model(inputs=[user_input, exercise_input, teacher_input, result_input], outputs=output)

adam = keras.optimizers.Adam(lr=0.001)

model.compile(loss="mse", optimizer=adam, metrics=["accuracy"])

summary = model.summary()
print(summary)

loss = "output/reduced_datasets/" + dataset_number + "/model_10_" + dataset_chunk + "_loss.jpg"
accuracy = "output/reduced_datasets/" + dataset_number + "/model_10_" + dataset_chunk + "_accuracy.jpg"
plot_losses = TrainingPlot(loss, True)

plot_accuracy = TrainingPlot(accuracy, False)

earlystopping = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")

history = model.fit(X_train_array, Y_train, batch_size=64, epochs=5, verbose=1, validation_data=(X_val_array, Y_val), callbacks=[plot_losses, plot_accuracy, earlystopping])

# Generate generalization metrics
score = model.evaluate(X_test_array, Y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

save_path = "saved_models/" + dataset_number + "/dateset_" + dataset_chunk + "_model"
keras.models.save_model(model, save_path)


samples_to_predict = X_test_array
#
predictions = model.predict(samples_to_predict)
print(predictions)


pred_df = pd.DataFrame(predictions)
pred_df.to_csv("output/" + dataset_number + "_" + dataset_chunk + "model_10_predictions.csv")
test_labels = pd.DataFrame(Y_test)
test_labels.to_csv("output/" + dataset_number + "_" + dataset_chunk + "model_10_predictions_labels.csv")
