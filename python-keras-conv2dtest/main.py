from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf
import keras
import json
import logging
import os
from PIL import Image
import numpy as np

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

I = np.asarray(Image.open('randomimage1.png'))
I2 = np.asarray(Image.open('randomimage2.png'))
batch = np.asarray([I])

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same', input_shape=batch[0].shape,
                 activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=1),
                 bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=1)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(units=8, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=1)))
model.summary()

intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer('dense').output)
intermediate_output = intermediate_layer_model.predict(batch)
print(intermediate_output)

json_string = []

for i in range(len(model.layers)):
    weights_biases = []
    for k in range(len(model.layers[i].get_weights())):
        weights_biases.append(model.layers[i].get_weights()[k].tolist())
    json_string.append({'config': model.layers[i].get_config(), 'weights_biases': weights_biases})

with open('model-2.json', 'w', encoding='utf-8') as f:
    json.dump(json_string, f, ensure_ascii=False, indent=3)