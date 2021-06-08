import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from keras import initializers, regularizers
from keras import backend as K
import numpy as np
from tensorflow.keras.models import Sequential
import math
from Cell import Cell
import scipy.io as sio
import time
import tracemalloc
import os
from sklearn.model_selection import train_test_split
import decimal
from statistics import mean
import h5py


class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):

        self.add_loss(inputs)
        #self.add_loss(1e-2 * tf.reduce_sum(inputs))
        # proba ET 

        return inputs


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



# new 'sampling' function, only return an integer (to feed the Embedding layer)
def sampling(x, depth):
    #x = np.array([[0.1, 0.9]])
    zeros = x*0. ### useless but important to produce gradient
    samples = tf.random.categorical(tf.math.log(x[0]), 1)
    return samples


class Controller():

    def __init__(self, num_block_conv=2 , num_block_reduc=2, num_op_conv=5, num_op_reduc=2, num_alt=5):


        self.num_block_conv = num_block_conv      # Number of blocks (one block = 2 inputs/2 operators) per conv Cell

        self.num_block_reduc = num_block_reduc     # Number of blocks per reduc Cell

        self.num_op_conv = num_op_conv    # Number of operations to choose from when building a conv Cell

        self.num_op_reduc = num_op_reduc    # Number of operations to choose from when building a reduc Cell

        self.num_units = 100    # Number of units in the LSTM layer(s)

        self.num_alt = num_alt # Number of times alterning a conv/reduc cell


    # Generate an Embedding(Sampling/Lambda(Dense/Softmax  layer
    def LSTM_softmax(self, inputs, num_classes, initial, count, type_cell):

        if initial: # First LSTM layer
            x = layers.LSTM(self.num_units, return_state=True, return_sequences=True)(inputs)
        else:
            #x = layers.LSTM(self.num_units, return_state=True)(reshaped_inputs, initial_state=inputs[1:])
            x = layers.LSTM(self.num_units, return_state=True, return_sequences=True)(inputs)

        # Name for Dense = size of the softmax + i
        x = layers.Dense(num_classes, activation="softmax", name=str(count)+"-"+type_cell+"-"+str(num_classes))(x[0])

        x = ActivityRegularizationLayer()(x)

        x = layers.Lambda(lambda t: sampling(t, depth=num_classes))(x)

        x = layers.Embedding(num_classes, 100, trainable=False)(x)

        return x # x = LSTM output, rx = reshaped, emb = embedding


    # Self explainatory...
    def generateController(self):

        y_soft, r_y = None, None

        controller_input = keras.Input(shape=(1,1,))

        outputs = []

        input_cell = controller_input


        count = 0

        for i in range(0, self.num_alt):
        

            # Generate the conv blocks
            for j in range(0, self.num_block_conv):

                # for each block/node, we choose 2 inputs and 2 operations

                for o in ["inputL", "inputR", "operL", "operR"]:

                    if i == 0 and j == 0 and o == "inputL":
                        _x, _initial = input_cell, True
                    else:
                        #_x, _rx, _initial = emb, rx, False # output of previous LSTM_softmax
                        _x, _initial = x, False # output of previous LSTM_softmax

                    if o == "inputL" or o == "inputR" :
                        # 1 softmax de taille 2 pour inputL et inputR
                        _num_classes = j+1
                    else:
                        # 1 softmax de taille #num_op
                        _num_classes = self.num_op_conv


                    count+=1
                    x = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="conv")
                    #x, rx, emb = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, reshaped_inputs=_rx, initial=_initial)

                    outputs.append(x)


            # Generate the reduc blocks
            for j in range(0, self.num_block_reduc):

                _x, _initial = x, False # output of previous LSTM_softmax

                _num_classes = self.num_op_reduc

                count+=1
                x = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="reduc")
                #x, rx, emb = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, reshaped_inputs=_rx, initial=_initial)

                outputs.append(x)

        model = Model(inputs=controller_input, outputs=outputs)


        return model


def get_compiled_cnn_model(cells_array, block_names):


    # Init of the child net
    outputs = []
    input_shape = (8, 900, 1)
    inputs = keras.Input(shape=(8, 900), name="outside_input")
    x = layers.Reshape((8,900,1), name="outside_reshape")(inputs)
    #outputs.append(x)
    x = layers.BatchNormalization(name="outside_batchnorm")(x)
    #outputs.append(x)
    x = layers.Conv2D(10, (1, 1), padding="same", activation='relu', name="outside_conv2")(x)
    
    outputs.append(x)
    
    # loop over the cells and construct the model
    for i, arr in enumerate(cells_array):

        if(i>0):
            outputs = [x]

        if( i%2==0 ):

            # Conv Cell, cell_inputs must ALWAYS be an array with the possible DIRECT inputs for the cell
            cell = Cell('conv', cell_inputs=outputs)
            blocks_array = np.array( arr )
            cell.generateCell( blocks_array, 'conv', num_cell=i )
            x = cell.cell_output

        else:

            cell = Cell('reduc', cell_inputs=outputs)
            blocks_array = np.array( arr )
            cell.generateCell(blocks_array, 'reduc', num_cell=i)
            x = cell.cell_output


    x = layers.Flatten(name="outside_flatten")(x)
    x = layers.BatchNormalization(name="outside_batchnorm2")(x)
    x = layers.Dropout(0.25, name="outside_droupout1")(x)
    x = layers.Dense(10, name="outside_dense1")(x)
    x = layers.Dropout(.25, name="outside_droupout2")(x)

    # Create the model
    outputss = layers.Dense(2, activation='softmax', name="outside_dense2")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputss, name="ansari_model")
    utils.plot_model(model, to_file="child_ENAS5.png")


    # Compute the accuracy
    loss = keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(1e-3), metrics=['accuracy'])
    
    return model



def moving_average(x, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return mean(a)



def truncate_float(number, length):
    """Truncate float numbers, up to the number specified
    in length that must be an integer"""

    number = number * pow(10, length)
    number = int(number)
    number = float(number)
    number /= pow(10, length)
    return number


def duplicate_in_array(array, decimals):
    decimal.getcontext().rounding = decimal.ROUND_DOWN
    for i in range(len(array)):
        if ( i < (len(array)-1) ):
            if(truncate_float(array[i], decimals) == truncate_float(array[i+1], decimals)):
                return True
    return False



def main():

    tracemalloc.start()

    loaded = load_data("../../../../data_conv_training.mat", "../../../../data_conv_testing.mat",clip_value=300)

    # Here we take only the 50 1st examples
    X = loaded[0].transpose((0,2,1)) # eeg training
    y = loaded[1]
    y = keras.utils.to_categorical(y, num_classes=2)

    
    # Init data option & disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    ##########################
    # Controller Instanciation
    ##########################

    num_block_conv=3
    num_block_reduc=1

    num_alt=2


    controllerInst = Controller(num_block_conv=num_block_conv, num_block_reduc=num_block_reduc, num_op_conv=7, num_op_reduc=2, num_alt=num_alt)
    controller = controllerInst.generateController()
    #utils.plot_model(controller, to_file="controller_example2.png")

    
    ###############
    # TRAINING LOOP
    ###############


    epochs = 9999999

    sampling_number = 1 # number of arch child to sample at each epoch

    sum_over_choices = 0 # outer sum of the policy gradient


    optimizer = keras.optimizers.SGD(learning_rate=1e-3)

    accuracies = []
    mean_acc = []
    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    all_ones = 0
    all_convs = 0
    block_names = []
    
    total_weights = h5py.File("total_weights.h5", "w")
    
    
    start = time.time()

    # Loop over the epochs
    for epoch in range(epochs):

        print("Epoch: ",epoch, " time: ", time.time() - start)

        with tf.GradientTape(persistent=False) as tape:

            total_sum = 0

            # Loop over the number of samplings
            for s in range(sampling_number):

                val_acc = tf.Variable(0)

                # FEEDFORWARD
                inp = controller.input  # input placeholder
                outputs = [layer.output for layer in controller.layers if ( layer.__class__.__name__ == "Dense" or layer.__class__.__name__ == "Lambda" )]  # all layer outputs
                functor = K.function([inp], outputs )   # evaluation function
                test = np.random.random((1,1))[np.newaxis,...]
                layer_outs = functor([test]) # Here are all the outputs of all the layers

                # sum over the hyperparameters (i.e, over the choices made my the RNN)

                # final array of all blocks/cells
                cells_array = []

                cell1 = [] # tmp array for conv cells
                cell2 = [] # tmp array for reduc cells

                k=0
                u=0
                count=0


                ###################
                # MAKING CHILD ARCH
                ###################

                # loop over each layer (choice)
                for i in range(0, len(layer_outs), 2):

                    classe = layer_outs[i+1][0][0]
                    proba = controller.losses[int(i/2)][0][0][classe]
                    total_sum -= tf.math.log(proba)

                    # when layer is for a convCell choice
                    if ( k < num_block_conv*4 ):
                        cell1.append(classe)                    
                        
                        #print("k: "+str(k)+ " "+str(cell1))
                        if(k==(num_block_conv*4-1)):
                            cells_array.append(cell1)
                            cell1=[]
                        k+=1

                    # when layer is for a reducCell choice
                    else:
                        cell1=[]
                        if(u<num_block_reduc):
                            #print("u: "+str(u))
                            cell2.append(classe)
                            if(u==(num_block_reduc-1)):
                                cells_array.append(cell2)
                                cell2=[]
                            u+=1
                        else:
                            k=0
                            u=0
                        if(u==num_block_reduc):
                            k=0
                            u=0
                    count+=1

                X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.03, test_size=0.003, shuffle= True)

                train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                
                val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))

                batch_size = 30

                train_data = train_data.batch(batch_size)
                val_data = val_data.batch(batch_size)

                train_data = train_data.with_options(options)
                val_data = val_data.with_options(options)

                #########

                with strategy.scope():

                    model = get_compiled_cnn_model(cells_array, block_names)

                callback = tf.keras.callbacks.EarlyStopping(monitor='loss')
             
                model.load_weights("total_weights.h5", by_name=True)
                    
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=10,
                    batch_size=30,
                    callbacks=[callback],
                    #validation_split=0.1
                )

                model.save_weights("tmp_weights.h5")

                tmp_weights =  h5py.File('tmp_weights.h5', 'r')
                keys_tmp_weights = list(tmp_weights.keys())
                keys_total_weights = list(total_weights.keys())

                for key in keys_tmp_weights:
                    if key not in keys_total_weights:
                        tmp_weights.copy(tmp_weights[key], total_weights)
                    else:
                        del total_weights[key]
                        tmp_weights.copy(tmp_weights[key], total_weights)
                              
                tmp_weights.close()
                
                val_acc = history.history['val_accuracy'][-1]                
                accuracies.append(val_acc)
                
                ema = 0
                if (len(accuracies) > 10):
                    mean_acc.append(mean(accuracies[-10:]))
                    ema = moving_average(accuracies, 10, type='exponential')
                    f = open("log_Controller_ENAS5.txt", "a")
                    f.write("ema: "+str(ema)+" mean_acc = "+str(mean_acc[-1])+" Epoch: "+str(epoch)+ " time: "+ str(time.time() - start)+"\n")
                    f.close()
                    
                    
                    
                
                total_sum *= ( val_acc - ema )
                model.save('ENAS5_last_child')
                del model

                
            total_sum/=sampling_number
            
        
        grads = tape.gradient(total_sum, controller.trainable_weights)
        optimizer.apply_gradients(zip(grads, controller.trainable_weights))

        if(len(mean_acc)>0):
            controller.save_weights("ENAS5_weights_.h5")
            
    total_weights.close()
    print("total allocated memory")
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()


if __name__ == "__main__":

    main()