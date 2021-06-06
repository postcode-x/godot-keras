from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.callbacks import TensorBoard
import numpy as np
from time import time
import json
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# run this to see tensorboard charts & stats:
# python <path to your python installation folder>\site-packages\tensorboard\main.py --logdir=logs/


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float)
y = np.array([[0], [1], [1], [0]], dtype=np.float)

tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
model_name = 'XOR'
restore_last_model = True

if not restore_last_model:

    # New sequential model
    model = Sequential(name=model_name)
    model.add(Dense(16, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(4, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
    model.fit(X, y, batch_size=1, epochs=500, callbacks=[tensorboard], verbose=2)
    model.save('xor-last-model.h5')

    # Dump weights & biases to a json file which we can load inside Godot
    json_string = []

    for i in range(len(model.layers)):
        weights_biases = []
        for k in range(len(model.layers[i].get_weights())):
            weights_biases.append(model.layers[i].get_weights()[k].tolist())
        json_string.append({'config': model.layers[i].get_config(), 'weights_biases': weights_biases})

    with open('model-1.json', 'w', encoding='utf-8') as f:
        json.dump(json_string, f, ensure_ascii=False, indent=3)

else:

    # Restore last saved sequential model
    model = load_model('xor-last-model.h5')

model.summary()
print('Prediction:', y.tolist(), ' -> ', model.predict(X).tolist())
