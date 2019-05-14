import tensorflow as tf
import numpy as np

training_data = np.load('train_data.npy',allow_pickle=True)
testing_data = np.load('test_data.npy',allow_pickle=True)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',input_shape=((60,60,1))),
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),

    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),

    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),

    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),

    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),

    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(activation='relu',units=4096),
    tf.keras.layers.Dense(activation='relu',units=4096),
    tf.keras.layers.Dense(activation='softmax',units=1000)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

X_train = np.array([i[0] for i in training_data])
Y_train = np.array([i[1] for i in training_data])
X_train = X_train[:,:,:,np.newaxis]

model.fit(X_train,Y_train,epochs=5,batch_size=64)

X_test = np.array([i for i in testing_data])

Y_test = model.predict(X_test)
print(Y_test)