import os
import sys
import json
import time
import tensorflow as tf


per_worker_batch_size = 256
hosts = sys.argv[1].split(',')
task_id = int(sys.argv[2])
num_workers = len(hosts)
tab = []
for host in hosts:
    tab.append(str(host)+":2222")
cluster = tf.train.ClusterSpec({'worker': tab})
cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(cluster,
                                         task_type='worker', task_id=task_id,
                                         num_accelerators={'GPU': 0})
data_options = tf.data.Options()
data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
options = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver, communication_options=options)       
global_batch_size = per_worker_batch_size * num_workers

from Controller import *
from Utils import *
import numpy as np

def main():

    with strategy.scope():
        model = get_compiled_cnn_model()


    X = np.random.rand(10000,8,900)
    y = np.random.choice(2, 10000)
    y = keras.utils.to_categorical(y, num_classes=2)
    
    
    epochs_=5
    history = train_child(X, y, model, global_batch_size, epochs_, data_options, callbacks=[])



    
def get_compiled_cnn_model():

    outputs = []
    input_shape = (8, 900, 1)
    inputs = keras.Input(shape=(8, 900))
    x = layers.Reshape((8,900,1))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(10, (2, 10), padding="same", activation='relu')(x)
    outputs.append(x)

    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(10)(x)
    x = layers.Dropout(.25)(x)

    outputss = layers.Dense(2, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputss)
    loss = keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])

    return model



def train_child(X, y, model, batch_size, epochs_child, options, callbacks):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True)
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)

    call = None

    if(len(callbacks)>0):
        call = callbacks
    history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs_child,
            batch_size=batch_size,
            callbacks=call,
            verbose=1,
        )
    return history     




    
if __name__ == "__main__":

    main()