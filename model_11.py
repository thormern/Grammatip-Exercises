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



def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        tmp = end_ix + 3
        if end_ix > len(sequence)-1:
            break
        seq_x = sequence[i:end_ix]

        seq_y = sequence[end_ix][1]

        X.append(seq_x)
        y.append(seq_y)


    # # print(X)
    # # print(y)
    # new_y = []
    # for i in range(len(y)):
    #     new_element = []
    #     for yidx in y[i]:
    #         new_element.append(yidx[1])
    #     if len(new_element) == 1:
    #         new_element.append(0)
    #         new_element.append(0)
    #     if len(new_element) == 2:
    #         new_element.append(0)
    #     new_y.append(new_element)

    return np.array(X), np.array(y)


raw = [[1,34,2,1,50],[1,35,2,2,40],[1,36,2,2,60],[1,37,2,1,50],[1,38,2,2,70],[1,39,2,2,65]]
x_data = normalized_df.drop("target_exercise_id", axis=1)
x_data.drop(x_data.columns[[0]], axis=1, inplace=True)
x_data = x_data.to_numpy()
y_data = pd.get_dummies(normalized_df["target_exercise_id"])
print(y_data)
y_data = y_data.to_numpy()
n_steps = 1

X, y = split_sequence(x_data, n_steps)

n_classes = len(np.unique(y))
# print(n_classes)
print(X.shape)
print(y.shape)

print(len(X))
print(len(y))
# print(y)

print(y_data)
print(y_data.shape)
y = pd.DataFrame(y)
print(y)
y_onehot_0 = pd.get_dummies(y[0]) #    1, 0, 0
                                  #    0, 1, 0
                                  #    0, 0, 1
# y_onehot_1 = pd.get_dummies(y[1]) # 0,    1, 0
#                                   # 0,    0, 1
#                                   # 1,    0, 1
# y_onehot_2 = pd.get_dummies(y[2]) # 0,       1
                                  # 1,       0
                                  # 1,       0

                                  # 0, 1, 1, 1
                                  # 1, 0, 1, 1
                                  # 1, 0, 0, 1
print(y_onehot_0)

# final_y = y_onehot_0.add(y_onehot_1, fill_value=0)
# final_y = final_y.add(y_onehot_2, fill_value=0)

# final_y[final_y > 1] = 1
#
# final_y = final_y.astype(int)

# print(final_y)
# y = pd.concat([pd.get_dummies(y[col]) for col in y], axis=1)

print(y)
y = y_onehot_0.to_numpy()
print(y)
print(y_data.shape)
if len(X) != len(y_data):
    tmp = abs(len(X) - len(y_data))
    y_data = y_data[:-tmp]


print(y.shape)
print(y_data.shape)

X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, y_data, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)


# X_train = np.dstack(X_train)
# Y_train = keras.utils.to_categorical(Y_train)
# # X_val = np.dstack(X_val)
# Y_val = keras.utils.to_categorical(Y_val)
# # X_test = np.dstack(X_test)
# Y_test = keras.utils.to_categorical(Y_test)

print(X_train.shape, X_val.shape, X_test.shape)
print(Y_train.shape, Y_val.shape, Y_test.shape)
print(X_train)
print(Y_train)

n_outputs = Y_train.shape[1]
n_features = X_train.shape[2]
n_timesteps = X_train.shape[1]
# The Model


model = keras.models.Sequential()
model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(n_steps, n_features), strides=2, padding="same"))
model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", strides=2, padding="same"))
model.add(keras.layers.MaxPooling1D(strides=2, padding="same"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(n_outputs, activation="softmax"))

adam = keras.optimizers.Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

earlystopping = EarlyStopping(monitor="loss", patience=20, verbose=1, mode="auto")
loss = "output/model_11_" + dataset_chunk + "_loss.jpg"
accuracy = "output/model_11_" + dataset_chunk + "_accuracy.jpg"

plot_losses = TrainingPlot(loss, True)
plot_accuracy = TrainingPlot(accuracy, False)

summery = model.summary()
print(summery)


result = model.fit(X_train, Y_train, epochs=500, batch_size=32, verbose=1, validation_data=(X_val, Y_val), callbacks=[plot_losses, plot_accuracy, earlystopping])

model.evaluate(X_test, Y_test, batch_size=64, verbose=1)

# Generate generalization metrics
score = model.evaluate(X_test, Y_test, verbose=1)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# save_path = "saved_models/" + dataset_number + "/dateset_" + dataset_chunk + "_model"
# keras.models.save_model(model, save_path)

#
predictions = model.predict(X_test)
print(predictions)


pred_df = pd.DataFrame(predictions)
pred_df.to_csv("output/" + dataset_number + "_" + dataset_chunk + "model_11_predictions.csv")
test_labels = pd.DataFrame(Y_test)
test_labels.to_csv("output/" + dataset_number + "_" + dataset_chunk + "model_11_predictions_labels.csv")
