# import numpy as np
# import os
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

from tensorflow import keras
from tensorflow.keras import layers



input_shape = (8, 900, 1)
inputs = keras.Input(shape=(8, 900))
x = layers.Reshape((8,900,1))(inputs)

x = layers.BatchNormalization()(x)

x1 = layers.Conv2D( 10, (2, 10), padding="same", activation='relu' )(x)

x2 = layers.Conv2D( 10, (2, 10), padding="same", activation='relu' )(x1)

print(x1)
print(x2)
#x = layers.Add()([x1,x2])