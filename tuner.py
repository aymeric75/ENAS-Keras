import scipy.io as sio
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
import kerastuner
from sklearn.model_selection import train_test_split


def load_data(trainingPath, validationPath, clip_value=300):

    # Loading the training data
    matTR = sio.loadmat(trainingPath)
    eeg = np.transpose(matTR['data']['x'][0][0],(2,1,0))
    Y = matTR['data']['y'][0][0]
    Y = np.reshape(Y,newshape=(Y.shape[0],))
    del matTR

    eeg = np.clip(eeg, a_min=-clip_value, a_max=clip_value)

    # Loading validation data
    matTR = sio.loadmat(validationPath)
    eeg_val = np.transpose(matTR['data']['x'][0][0],(2,1,0))
    Y_val = matTR['data']['y'][0][0]
    Y_val = np.reshape(Y_val,newshape=(Y_val.shape[0]))
    del matTR

    eeg_val = np.clip(eeg_val, a_min=-clip_value, a_max=clip_value)

    return eeg,Y,eeg_val,Y_val



loaded = load_data("../../../data_conv_training.mat", "../../../data_conv_testing.mat",clip_value=300)



X = loaded[0].transpose((0,2,1)) # eeg training
y = loaded[1]
y = keras.utils.to_categorical(y, num_classes=2)

X_test = loaded[2].transpose((0,2,1))
y_test = loaded[3]
y_test = keras.utils.to_categorical(y_test, num_classes=2)





def mvaverage(vals, window):
    tmp = np.copy(vals)
    for i, val in enumerate(vals):
        if(i >= window):
            # calcul de la moyenne
            average = np.average(vals[i-window: i])
            if(average < 0.5):
                tmp[i] = 0
            else:
                tmp[i] = 1
    return tmp




class MyTuner(kerastuner.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        # You can add additional HyperParameters for preprocessing and custom training loops
        # via overriding `run_trial`
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32)
        kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
        super(MyTuner, self).run_trial(trial, *args, **kwargs)





def build_model(hp):


    input_shape = (8, 900, 1)

    inputs = keras.Input(shape=(8, 900))

    x = layers.Reshape((8,900,1))(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D( filters=hp.Choice('num_filters_1', values=[10, 20, 30, 40]), kernel_size=(2, 10), dilation_rate = ( hp.Choice('dilation_rate_1', values=[1,2,3,4]), 1) , padding="same", activation='relu', input_shape=input_shape[:-1] ) (x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    #x = layers.MaxPooling2D(pool_size=hp.Int('pool_size', min_value=2, max_value=4, step=1))

    #x = layers.Conv2D(20, (1, 5), padding="same", activation='relu')(x)
    x = layers.Conv2D(filters=hp.Choice('num_filters_2', values=[10, 20, 30, 40]), kernel_size=(1, 5), padding="same", activation='relu')(x)

    #x = layers.Conv2D(20, (4, 1), padding="valid", activation='relu')(x)
    x = layers.Conv2D(filters=hp.Choice('num_filters_3', values=[10, 20, 30, 40]), kernel_size=(4, 1), padding="valid", activation='relu')(x)

    x = layers.MaxPooling2D(pool_size=(1, 6))(x)

    x = layers.Flatten()(x)

    x = layers.BatchNormalization()(x)

    #x = layers.Dropout(0.25)(x)
    x = layers.Dropout(
        rate=hp.Float(
            'dropout_1',
            min_value=0.20,
            max_value=0.5,
            step=0.05,
        )
        )(x)

    #x = layers.Dense(units=10)(x)
    x = layers.Dense(units=hp.Int('units', min_value=10, max_value=30, step=5))(x)

    #x = layers.Dropout(0.25)(x)
    x = layers.Dropout(
        rate=hp.Float(
            'dropout_2',
            min_value=0.20,
            max_value=0.5,
            step=0.05,
        )
        )(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="ansari_model")
    

    loss = keras.losses.CategoricalCrossentropy()
    #loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3])), metrics=['accuracy'])

    #print(model.summary())


    return model








##################################
#  Parameter Search (Keras Tuner)
##################################




strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


tuner = kerastuner.tuners.BayesianOptimization(
  build_model,
  objective='val_accuracy',
  #executions_per_trial=3, # Number of time we train an exact same model (with exact same parameters)
  max_trials=50,
  project_name='bayesearch',
  distribution_strategy=strategy
  )


tuner.search_space_summary()



my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=3),
]


# Init data option & disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle= True)

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))

val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))

batch_size = 30

train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)

train_data = train_data.with_options(options)
val_data = val_data.with_options(options)


batch_size = 30
num_epochs = 50
tuner.search(
    train_data,
    validation_data=val_data,
    batch_size=batch_size,
    epochs=num_epochs,
    callbacks=my_callbacks,
    verbose=1
)


models = tuner.get_best_models(num_models=4)

tuner.results_summary()