import numpy as np
import tensorflow as tf
import os


INPUT_HEIGHT=32
INPUT_WIDTH=16
INPUT_CHAN=1
OUT='./out'

LEARN_RATE = 0.001
DECAY_RATE = 1e-6

x_data = np.load('x.npy')
y_data = np.load('y.npy')

l = int(len(x_data) * 0.8)

# Split the data
train_x = np.asarray(x_data[:l])
train_y = np.asarray(y_data[:l])
test_x = np.asarray(x_data[l:])
test_y = np.asarray(y_data[l:])

# Reshape the data to input shape
train_x = train_x.reshape(-1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHAN)
test_x = test_x.reshape(-1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHAN)

num_classes = 2

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(train_x.shape[1:]),
    tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes)
])

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)
model.summary()

model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(lr=LEARN_RATE, decay=DECAY_RATE),
              metrics=['accuracy']
              )
model.fit(train_x, train_y, epochs=10)

test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

print('\nTest accuracy:', test_acc)

model.save(os.path.join(OUT,'keras_out.h5'))


