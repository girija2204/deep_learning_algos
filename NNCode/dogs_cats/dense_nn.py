import tensorflow as tf
import numpy as np
import os

training_data = np.load('train_data.npy',allow_pickle=True)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation=tf.nn.relu),
    tf.keras.layers.Dense(512,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

X_train = np.array([i[0] for i in training_data])
Y_train = np.array([i[1] for i in training_data])

model.fit(X_train, Y_train, epochs=1, batch_size=64)

testing_data = np.load('test_data.npy',allow_pickle=True)

X_test = np.array([i for i in testing_data])

Y_test = model.predict(X_test)
print(Y_test)