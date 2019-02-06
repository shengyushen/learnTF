import tensorflow as tf
from tensorflow.keras import layers


print(tf.VERSION)
print(tf.keras.__version__)

# acutally we add all these layers in one list
# strange , I have not specified the shape og input data
model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
# dense is just another name of fully connected with activation 
layers.Dense(64, activation='relu', input_shape=(32,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

# of course we can add them one by one with model.add
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# print out the layers of this model
model.summary()


import numpy as np

# first 1000 is m, the numbe rof example
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))

data_evl = np.random.random((1000, 32))
labels_evl = np.random.random((1000, 10))

# this have input of both data and label
model.evaluate(data_evl, labels_evl, batch_size=32)
# while predict input only data, but output label
result = model.predict(data, batch_size=32)
print(result.shape)



