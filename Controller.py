import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from keras import initializers, regularizers
from keras import backend as K
import numpy as np
from tensorflow.keras.models import Sequential
from sklearn import metrics
import math
from Cell import Cell
import scipy.io as sio
import time
import tracemalloc
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import decimal
from statistics import mean
import h5py
import tensorflow_addons as tfa
#from Utils import ActivityRegularizationLayer
#from Utils import moving_average, mvaverage, recall_m, precision_m, f1_m, kappa, truncate_float, duplicate_in_array
from Utils import *
import matplotlib.pyplot as plt
import random
import json
import datetime

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


class Controller():

    def __init__(self, num_block_conv=2 , num_block_reduc=2, num_op_conv=5, num_op_reduc=2, num_alt=5, scheme=1, path_train="./data_conv_training.mat", path_test="./data_conv_testing.mat"):


        self.num_block_conv = num_block_conv      # Number of blocks per conv Cell

        self.num_block_reduc = num_block_reduc     # Number of blocks per reduc Cell

        self.num_op_conv = num_op_conv    # Number of operations to choose from when building a conv Cell

        self.num_op_reduc = num_op_reduc    # Number of operations to choose from when building a reduc Cell

        self.num_units = 100    # Number of units in the LSTM layer(s)

        self.num_alt = num_alt # Number of times alterning of conv/reduc cell

        self.scheme = scheme

        self.path_train = path_train

        self.path_test = path_test

    # new 'sampling' function, only return an integer (to feed the Embedding layer)
    # return the 'choice' made by sampling
    def sampling(self, x, depth):
        #x = np.array([[0.1, 0.9]])
        zeros = x*0. ### useless but important to produce gradient
        samples = tf.random.categorical(tf.math.log(x[0]), 1)
        return samples



    # Generate an Embedding(Sampling/Lambda(Dense/Softmax  layer
    def LSTM_softmax(self, inputs, num_classes, initial, count, type_cell):

        if initial: # First LSTM layer
            x = layers.LSTM(self.num_units, return_state=True, return_sequences=True)(inputs)
        else:
            #x = layers.LSTM(self.num_units, return_state=True)(reshaped_inputs, initial_state=inputs[1:])
            x = layers.LSTM(self.num_units, return_state=True, return_sequences=True)(inputs)

        # Name for Dense = size of the softmax + i
        x = layers.Dense(num_classes, activation="softmax", name=str(count)+"-"+type_cell+"-"+str(num_classes))(x[0])

        x = ActivityRegularizationLayer()(x) # useful for retrieving Softmaxs outputs during feedforward (see sample_arch function)

        x = layers.Lambda(lambda t: self.sampling(t, depth=num_classes))(x) # sample from the softmax and return an integer (choice)

        x = layers.Embedding(num_classes, 100, trainable=False)(x)

        return x # x = LSTM output, rx = reshaped, emb = embedding


        # Generate the Mode
    def generateController(self):

        y_soft, r_y = None, None
        controller_input = keras.Input(shape=(1,1,))
        outputs = []
        input_cell = controller_input
        array_normal = []
        array_reduc = []
        count = 0

        # loop over the number of pairs (Normal_cell/Reduc_cell)
        for i in range(0, self.num_alt):
        
            # Generate the conv(normal) blocks (of the Normal cell)
            for j in range(0, self.num_block_conv):

                if ( self.scheme == 1 ): # 1 choice made here (one operation)
                    
                    if i == 0 and j == 0:
                        _x, _initial = input_cell, True
                    else:
                        _x, _initial = xx, False

                    _num_classes = self.num_op_conv

                    array_normal.append(count)
                    count+=1
                    xx = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="conv")
                    outputs.append(xx)

                elif ( self.scheme == 2 ): # 2 + 3 x num_block_conv

                    if ( j < 2 ): # 1st two blocks are just with 1 op / 1 input

                        if i == 0 and j == 0:
                            _x, _initial = input_cell, True
                        else:
                            _x, _initial = xx, False

                        _num_classes = self.num_op_conv

                        array_normal.append(count)
                        count+=1
                        xx = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="conv")
                        outputs.append(xx)

                    else: # next blocks have at most two inputs (one from previous output, one from ONE of the previous inputS)

                        for o in ["inputL", "operL"]:
                            
                            if i == 0 and j == 0 and o == "inputL":
                                _x, _initial = input_cell, True
                            else:
                                _x, _initial = xx, False

                            if o == "inputL" :
                                _num_classes = j
                            else:
                                _num_classes = self.num_op_conv

                            array_normal.append(count)
                            count+=1
                            xx = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="conv")

                            outputs.append(xx)

                elif ( self.scheme == 3 ): # blocks with at most 2 operations (but one input)

                    for o in ["operL", "operR"]:

                        if i == 0 and j == 0 and o == "operL":
                            _x, _initial = input_cell, True
                        else:
                            _x = xx
               
                        _num_classes = self.num_op_conv
                 
                        array_normal.append(count)
                        count+=1
                        xx = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="conv")
                        outputs.append(xx)

                else: # scheme == 4, blocks with at most 2 inputs and 2 operations

                    
                    if ( j < 2 ): # 1st two blocks with 1 op, 1 input only

                        for o in ["operL", "operR"]:

                            if i == 0 and j == 0 and o == "operL":
                                _x, _initial = input_cell, True
                            else:
                                _x = xx

                            _num_classes = self.num_op_conv

                            array_normal.append(count)
                            count+=1
                            xx = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="conv")
                            outputs.append(xx)
                        
                    else: #  next blocks with at most 2 inputs and 2 ops

                        for o in ["inputL", "operL", "operR"]:

                            if i == 0 and j == 0 and o == "inputL":
                                _x, _initial = input_cell, True
                            else:
                                _x, _initial = xx, False # output of previous LSTM_softmax

                            if o == "inputL" :
                                _num_classes = j
                            else:
                                _num_classes = self.num_op_conv

                            array_normal.append(count)
                            count+=1
                            xx = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="conv")
                            outputs.append(xx)


            # Generate the reduc blocks
            for j in range(0, self.num_block_reduc):

                _x, _initial = xx, False

                _num_classes = self.num_op_reduc

                array_reduc.append(count)
                count+=1
                xx = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="reduc")
                outputs.append(xx)

        model = Model(inputs=controller_input, outputs=outputs)
        #utils.plot_model(model, to_file="Controller.png")

        return model, array_normal, array_reduc


    
    
    
    def get_compiled_cnn_model(self, cells_array):


        # Init of the child net
        outputs = []
        input_shape = (8, 900, 1)
        inputs = keras.Input(shape=(8, 900), name="outside_input")
        x = layers.Reshape((8,900,1), name="outside_reshape")(inputs)
        #outputs.append(x)
        x = layers.BatchNormalization(name="outside_batchnorm")(x)
        #outputs.append(x)
        #x = layers.Conv2D(10, (2, 10), padding="same", activation='relu', name="outside_conv2")(x)
        outputs.append(x)
        
        
        # loop over the cells and construct the model
        for i, arr in enumerate(cells_array):


            if(i>0):
                outputs = [x]

            if( i%2==0 ):

                # Conv Cell, cell_inputs must ALWAYS be an array with the possible DIRECT inputs for the cell
                cell = Cell('conv', cell_inputs=outputs, scheme=self.scheme)
                blocks_array = np.array( arr )
                cell.generateCell( blocks_array, 'conv', num_cell=i )
                x = cell.cell_output

            else:

                cell = Cell('reduc', cell_inputs=outputs, scheme=self.scheme)
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
        utils.plot_model(model, to_file="child.png")

        # metrics=['acc',f1_m,precision_m, recall_m]
        # Compute the accuracy
        #acc = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.3)
        loss = keras.losses.CategoricalCrossentropy()
        # f1_m, precision_m, recall_m,tfa.metrics.CohenKappa(num_classes=2)
        model.compile(loss=loss, optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
        
        return model



    def load_shaped_data_train(self, random=0, seed=0):

        if(seed==1):
            np.random.seed(10)

        if(random==1):

            X = np.random.rand(10000,8,900)
            y = np.random.choice(2, 10000)
            y = keras.utils.to_categorical(y, num_classes=2)
            
        else:

            loaded = load_data(self.path_train, self.path_test, clip_value=300)
            X = loaded[0].transpose((0,2,1)) # eeg training
            y = loaded[1] 
            y = keras.utils.to_categorical(y, num_classes=2)
            
        return X, y        





    def load_shaped_data_test(self):

        loaded = load_data("./data_conv_training.mat", "./data_conv_testing.mat",clip_value=300)

        X_test = loaded[2].transpose((0,2,1))
        y_test = loaded[3]
            
        return X_test, y_test        



    def sample_arch(self, controller, array_normal, array_reduc):

        len_normale = int(len(array_normal)/self.num_alt)
        len_reduc = int(len(array_reduc)/self.num_alt)

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

        sum_proba = 0

        ###################
        # MAKING CHILD ARCH
        ###################

        # loop over each layer (choice)
        for i in range(0, len(layer_outs), 2): # 2 because there is this Lambda layer also

            classe = layer_outs[i+1][0][0]

            proba = controller.losses[int(i/2)][0][0][classe]
            sum_proba -= tf.math.log(proba)

            if( int(i/2) in array_normal):

                
                if ( len(cell1) < len_normale ):  # 
                    cell1.append(classe)

                if ( len(cell1) == len_normale ):
                    cells_array.append(cell1)
                    cell1 = []

            else:

                if ( len(cell2) < len_reduc ):  # 
                    cell2.append(classe)

                if ( len(cell2) == len_reduc ):
                    cells_array.append(cell2)
                    cell2 = []


        return cells_array, sum_proba




    def train_child(self, X, y, model, batch_size, epochs_child, options, callbacks, class_w):
                    

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True)
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
        train_data = train_data.batch(batch_size)
        val_data = val_data.batch(batch_size)
        train_data = train_data.with_options(options)
        val_data = val_data.with_options(options)
        
        y = out_categorical(y_train)
        class_w = class_weights(y)
        
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
                class_weight=class_w,
                #validation_split=0.1
            )
       

        return history     


    def train(self, strategy, global_batch_size, callbacks, rew_func, data_options, nb_child_epochs=2, mva1=10, mva2=500, epochs=5):
      
        
        X, y = self.load_shaped_data_train(random=0)

        controller, array_normal, array_reduc = self.generateController()

        sampling_number = 1 # number of arch child to sample at each epoch

        optimizer = keras.optimizers.SGD(learning_rate=1e-2)

        accuracies = []
        all_accs_rews = []
        means_mva1 = []
        means_mva2 = []
        
        
        dico_archs = {}  # each key is a hash of the array representing the arch
                         # each value is the corresponding accuracy (max acc so far...)
        
        #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        
        if os.path.exists("total_weights.h5"):
            os.remove("total_weights.h5")
        if os.path.exists("tmp_weights.h5"):
            os.remove("tmp_weights.h5")
        if os.path.exists("log_Controller_ema.txt"):
            os.remove("log_Controller_ema.txt")
        if os.path.exists("Controller_weights.h5"):
            os.remove("Controller_weights.h5")
       
            
        if not (os.path.isfile('./total_weights.h5')):
            total_weights = h5py.File("total_weights.h5", "w")
        
        start = time.time()

        

        # Loop over the epochs
        for epoch in range(epochs):

            #print("Epoch: ",epoch, " time: ", time.time() - start)

            with tf.GradientTape(persistent=False) as tape:

                total_sum = 0

                # Loop over the number of samplings
                for s in range(sampling_number):

                    val_acc = tf.Variable(0)
                    
                    

                    # Tries to generate a valid ARCH
                    no_valid = True
                    while no_valid:
                        cells_array, sum_proba = self.sample_arch(controller, array_normal, array_reduc)
                        try:
                            with strategy.scope():
                                model = self.get_compiled_cnn_model(cells_array)
                            no_valid = False
                        except ValueError:
                            print("Oops! not valid arch sampled, trying again...")
  
                    hash_ = hash(str(cells_array))
                    
                    total_sum += sum_proba
                        
                    # load existing weights ENAS !!!!!!!!!!!
                    model.load_weights("total_weights.h5", by_name=True)
                    
        
                    # !!!!!!!!!!!!! BEST OPTION !!!!!!!!!!!
                    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath='tmp_weights.h5',
                        save_weights_only=True,
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True)

                    callbacks = [model_checkpoint_callback]
                    
                    # train child      
                    history = self.train_child(X, y, model, global_batch_size, nb_child_epochs, data_options, callbacks, [])
                    
                    # !!!!!!!!!!!!! WORST OPTION !!!!!!!!!!!!! save model's weight
                    #model.save_weights("tmp_weights.h5")
                    
                    # # !!!!!!!!!!!!! WORST OPTION !!!!!!!!!!!!! update total_weights.h
                    """
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
                    """
                    
                    val_acc_max = max(history.history['val_accuracy'])
                    
                    # !!!!!!!!!!!!! BEST OPTION !!!!!!!!!!!
                    
                    if hash_ not in dico_archs:
                        dico_archs[hash_] = val_acc_max
                        # update total_weights.h
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
                        
                    else:
                        if( dico_archs[hash_] < val_acc_max ):
                            dico_archs[hash_] = val_acc_max
                        
                            # update total_weights.h
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
                         
                        else:
                            val_acc_max = dico_archs[hash_]
                            # ENAS !!!!

                    A = 1000
                    B = 840

                    if(len(all_accs_rews) > 11):
                        ema = moving_average(all_accs_rews[:-1], 10, type='exponential') 
                        acc_rew = ( 1 / (1 + np.exp(-val_acc_max*A+B)) ) * 100 - ema
                        print("acc_rew: "+str(acc_rew))
                    else:
                        acc_rew = ( 1 / (1 + np.exp(-val_acc_max*A+B)) ) * 100

                        
                    all_accs_rews.append(acc_rew)
                    #acc_rew = rew_func(val_acc_max)
                    #acc_rew = val_acc_max
                    accuracies.append(val_acc_max)

                    
                    """
                    if(cells_array == [[1, 2], [0]]):
                      
                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        ax1.set_xlabel('epochs')
                        ax2.set_xlabel('epochs')
                        ax1.set_ylabel('val_accuracy')
                        ax2.set_ylabel('val_loss')
                        
                        
                        print("cells_array INSIDE = "+str([[1, 2], [0]]))
                
                        val_accuracies = history.history['val_accuracy']
                        val_losses = history.history['val_loss']
  
                        plt.setp((ax1, ax2), xticks=np.arange(0, len(val_accuracies), 1))
                       
                        nbre_epochs = range(len(val_accuracies))
                        ax1.plot(nbre_epochs, val_accuracies, linestyle='-', color='b')
                        ax2.plot(nbre_epochs, val_losses, linestyle='-', color='r')

                        fig.tight_layout(pad=5.0)
                        plt.savefig("accuracy_loss_child_"+str(epoch)+".png")
                        plt.close()
                    """
                                        
                    ema = 0
                    if (len(accuracies) > mva1):
                        print("here1")
                        means_mva1.append(mean(accuracies[-mva1:]))
                        ema = moving_average(accuracies, mva1, type='exponential')
                        f = open("log_Controller_ema.txt", "a")
                        f.write("ema: "+str(ema)+" mean_acc = "+str(means_mva1[-1])+" Epoch: "+str(epoch)+ " time: "+ str(time.time() - start)+"\n")
                        f.close()

                        if(epoch % 10):
                            plt.figure()
                            plt.plot(np.arange(len(means_mva1))+mva1, means_mva1, 'b')    
                            plt.xlabel("NAS iteration")
                            plt.ylabel("accuracy (mva "+str(mva1)+")")
                            plt.savefig('incre_mva1_controller.png')
                        

                        
                        
                    if( len(accuracies) > mva2 ):
                        print("here2")
                        means_mva2.append(mean(accuracies[-mva2:]))
                        if(epoch % 10):
                            plt.figure()
                            plt.plot(np.arange(len(means_mva2))+mva2, means_mva2, 'b')  
                            plt.xlabel("NAS iteration")
                            plt.ylabel("accuracy (mva "+str(mva2)+")")
                            plt.savefig('incre_mva2_controller.png')            
                    
                    
                    total_sum *= ( acc_rew )
                    del model

                total_sum/=sampling_number
                
            grads = tape.gradient(total_sum, controller.trainable_weights)
            optimizer.apply_gradients(zip(grads, controller.trainable_weights))

            if( len(accuracies) > mva1 ):
                if(max(means_mva1) == means_mva1[-1]):
                    controller.save_weights("Controller_weights.h5")

        total_weights.close()



    # MEASUREMENT TOOLS


    def search_space_size(self):
        
        return self.num_alt*self.num_block_conv*self.num_block_reduc*self.num_op_conv*self.num_op_reduc
        
        

    def best_epoch(self, strategy, global_batch_size, data_options, class_w, nb_child_epochs = 20, ite=10):
        
        # sample and train 10 child archs  on  20 epochs each
        
        #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, )]
        callbacks = []
        # PLOT all eval_loss

        X, y = self.load_shaped_data_train(random=0)

        #plt.figure()
        
        controller, array_normal, array_reduc = self.generateController()
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
                
        plt.setp((ax1, ax2), xticks=np.arange(0, 10, 1))
        
        ax1.set_xlabel('epochs')
        ax2.set_xlabel('epochs')
        
        ax1.set_ylabel('val_accuracy')
        
        ax2.set_ylabel('val_loss')

        max_epochs = []
        
        all_arrays = []
        
        
        linestyles = ['-', '--', '-.']
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'slategray', 'magenta']
        
        combis = []
        
        for i in range(ite):
            
            if(i > 27):
                break

            k = 0
            no_valid = True
            while (no_valid or cells_array in all_arrays):
                cells_array, sum_proba = self.sample_arch(controller, array_normal, array_reduc)
                k+=1
                print("k = "+str(k))
                if(k>10):
                    return   
                try:
                    with strategy.scope():
                        model = self.get_compiled_cnn_model(cells_array)
                    no_valid = False
                except ValueError:
                    print("Oops! not valid arch sampled, trying again...")
                
            all_arrays.append(cells_array)

            history = self.train_child(X, y, model, global_batch_size, nb_child_epochs, data_options, callbacks, class_w)
            val_accuracy = history.history['val_accuracy']
            val_loss = history.history['val_loss']
            del model
            nbre_epochs = range(len(val_accuracy))
            max_epochs.append(len(val_accuracy))
            
            style = None
            colo = None
            found = False
            
            for sty in linestyles:
                for col in colors:
                    if [sty, col] not in combis:
                        style = sty
                        colo = col
                        combis.append([style, colo])
                        found = True
                        print("style = "+str(style)+" color = "+str(colo)+" best acc = "+str(max(val_accuracy)))
                        break
                if(found == True):
                    break
                        
            if(colo != None and style != None):

                ax1.plot(nbre_epochs, val_accuracy, linestyle=style, color=colo)
                ax2.plot(nbre_epochs, val_loss, linestyle=style, color=colo)
                fig.tight_layout(pad=5.0)
                plt.savefig('all_accuracies_losses.png')
                print("color: "+str(colo)+", style: "+str(style)+", model:"+str(cells_array))
        
        print(max_epochs)
        
        print(mean(max_epochs))
        
        return mean(max_epochs), max_epochs



    # check if the train_metric (used during training of each child)
    # "follows" the "final" metrics (kappa score), ie, on the test set
    #
    
    def match_train_test_metrics(self, train_metric, test_metric, nb_child_epochs, strategy, global_batch_size, data_options, class_w):


        X, y = self.load_shaped_data_train(random=0)
        
        X_test, y_test = self.load_shaped_data_test()
        
        controller, array_normal, array_reduc = self.generateController()
        
            
        train_metrics = []
        
        test_metrics = []
        
        test_metrics_2 = []
        
        kappas_av = []
        
        all_arrays = []
        
        
        for i in range(15):

            #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)] #restore_best_weights=True
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath='tmp_weights.h5',
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)

            callbacks = [model_checkpoint_callback]

            
            
            no_valid = True
            while (no_valid or cells_array in all_arrays):
                cells_array, sum_proba = self.sample_arch(controller, array_normal, array_reduc) 
                try:
                    with strategy.scope():
                        model = self.get_compiled_cnn_model(cells_array)
                    no_valid = False
                    all_arrays.append(cells_array)
                except ValueError:
                    print("Oops! not valid arch sampled, trying again...")
                        
            history = self.train_child(X, y, model, global_batch_size, nb_child_epochs, data_options, callbacks, class_w)
            #val_acc = max(history.history[train_metric])
            
            val_acc = max(history.history[train_metric])

            train_metrics.append(val_acc)

            # LOAD BEST WEIGHTS
            model.load_weights('tmp_weights.h5')
            
            # then test network on test set
            predictions = model.predict(X_test)
        
            del model
            predictions = np.argmax(predictions, axis=1)
          
            real_values = y_test

            test_metric_value = 0
            if(test_metric == 'kappa_score'):
                test_metric_value = metrics.cohen_kappa_score(predictions, real_values)
                print("kappa = "+str(test_metric_value))
            else:
                test_metric_value = metrics.accuracy_score(predictions, real_values)
       
            test_metrics.append(test_metric_value)
        
            #predictions_av = mvaverage(predictions, 6)
            #real_values_av = mvaverage(y_test, 6)
            #kappa_av = metrics.cohen_kappa_score(predictions_av, real_values_av)
            #kappas_av.append(kappa_av)
        
            test_metric_value_2 = metrics.accuracy_score(predictions, real_values)
            test_metrics_2.append(test_metric_value_2)
        

            plt.figure()

            plt.plot(np.arange(len(test_metrics)), test_metrics, 'o', color='green', label='test_kappa')

            plt.plot(np.arange(len(train_metrics)), train_metrics, 'o', color='red', label='train_acc')

            plt.plot(np.arange(len(test_metrics_2)), test_metrics_2, 'o', color='blue', label='test_acc')

            #plt.plot(np.arange(len(kappas_av)), kappas_av, 'o', color='orange', label='kappas_av')

            plt.xlabel("iteration")
            plt.ylabel("metrics value")

            plt.xticks(np.arange(0, len(test_metrics), 1))

            plt.savefig('match_test_train_metrics_best.png')

            print(train_metrics)

            print(test_metrics)
        
        
    # inc: considered increase of the metrics value
    # nb_child_epochs : nb of epochs of child training
    # return nber of necessary iterations meeting the input conditions
    
    def test_nb_iterations(self, file_accs = 'accuracies.txt', inc=0.16, mva1=10, mva2=500, limit=30000):
                
        if (inc > 0.4 or inc <= 0):
            raise Exception("Sorry, inc must be < 0.4 and > 0") 
            
        controller, array_normal, array_reduc = self.generateController()

        ####################################################################
        ##  Draw distribution of accuracy values among archs in search space
        ####################################################################
        
        f = open('accuracies.txt', 'r')
        lines = f.readlines()
        accuracies = []
        for line in lines:
            accuracies.append(float(line))
        f.close()
        
        if ( len(accuracies) < 100 ):
            raise Exception("Sorry, you need at least 100 accuracies to sample from a 'realistic' distribution") 
    
        print(accuracies)
        
                

        # 2) retrieve the freq of distri of accuracies in N quantiles
        N=100
        dico = frequency_distr(N, accuracies) # { mean_acc : freq }
        
        
        # 3) instantiate the dico of archs
        
        dico_archs = {}  # each key is a hash of the array representing the arch
                         # each value is the corresponding accuracy (mean acc)
        
        
        # print dico
        print(dico)
        
        
        
        # display search space size
        print(self.search_space_size())
        
        
        ##################################################################
        ## Train RNN while sampling children with above defined accuracies
        ##################################################################
        
        # 4) train RNN
        

        # loop forever until mean of accuracies increased by inc
        optimizer = keras.optimizers.SGD(learning_rate=1e-2)
        all_accs = []
        all_accs_rews = []
        means_mva1 = []
        means_mva2 = []
        count_iter = 0
        
        tmp_accuracies = accuracies.copy()
        
        while(count_iter < limit):
            
            total_sum=0
            with tf.GradientTape(persistent=False) as tape:

                cells_array, sum_proba = self.sample_arch(controller, array_normal, array_reduc)
                total_sum += sum_proba
                
                hash_ = hash(str(cells_array))
                
                
                # if model is in dico_archs
                if hash_ in dico_archs:
                    acc = dico_archs[hash_]
                else:
                    # for the given freq distri returns an accuracy
                    #acc = random.choices(list(dico.keys()), weights=list(dico.values()))[0]
                    #dico_archs[hash_] = acc
                          
                    # if size tmp_accuracies = 0, reload it to be accuracies
                    if(len(tmp_accuracies) == 0):
                        tmp_accuracies = accuracies.copy()
                    # randomly takes an ele from tmp_accuracies as acc and remove it
                    acc = random.choice(tmp_accuracies)
                    tmp_accuracies.remove(acc)

                    dico_archs[hash_] = acc
                        
                #acc_rew = acc
                             
                #acc_rew = (np.exp(acc)**20) / 8000
                
  
                A = 1000
                B = 840
                
                if(len(all_accs_rews) > 11):
                    ema = moving_average(all_accs_rews[:-1], 10, type='exponential') 
                    acc_rew = ( 1 / (1 + np.exp(-acc*A+B)) ) * 100 - ema
                    print("acc_rew: "+str(acc_rew))
                else:
                    acc_rew = ( 1 / (1 + np.exp(-acc*A+B)) ) * 100
                
                
                all_accs_rews.append(acc_rew)
                
                all_accs.append(acc)
        

                if( len(all_accs) > mva1 ):
                    means_mva1.append(mean(all_accs[-mva1:]))

                    if(count_iter % 1000):
                        plt.figure()
                        plt.plot(np.arange(len(means_mva1))+mva1, means_mva1, 'b')  
                        plt.xlabel("NAS iteration")
                        plt.ylabel("mva"+str(mva1)+" accuracy")
                        
                        #plt.title("Moving average with lag of "+str(mva))
                        plt.savefig('incre_mva1BIS55.png')

        
                if( len(all_accs) > mva2 ):
                    means_mva2.append(mean(all_accs[-mva2:]))
                    print("iter: "+str(count_iter)+" mean_acc with lag of "+str(mva2)+": "+str(means_mva2[-1]))
                    
                    if(count_iter % 1000):
                        plt.figure()
                        plt.plot(np.arange(len(means_mva2))+mva2, means_mva2, 'b')   
                        plt.xlabel("NAS iteration")
                        plt.ylabel("mva"+str(mva2)+" accuracy")
                        #plt.title("Moving average with lag of "+str(mva))
                        plt.savefig('incre_mva2BIS55.png')
                    
                    if(means_mva2[-1] - means_mva2[0] > inc):
                        print(str(count_iter)+" iterations were needed for an increase of "+str(inc)+" of the moving average")
                    
                total_sum *= ( acc_rew )
         
            count_iter+=1
            grads = tape.gradient(total_sum, controller.trainable_weights)
            optimizer.apply_gradients(zip(grads, controller.trainable_weights))
        
        return
        
    # compute and store nber accuracies
    # in file accuracies.txt
    # and displays stats
    def compute_accuracies(self, nber, nb_child_epochs, strategy, options, weights="Controller_weights_.h5", callbacks = [], global_batch_size=64, print_file=0):

        
        class_w = None
        
        controller, array_normal, array_reduc = self.generateController()
        
        if(weights != ''):
            controller.load_weights(weights)
        
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        accuracies = []
        
        start_time = time.time()
        
        #cells_array, __ = self.sample_arch(controller)
        
        X, y = self.load_shaped_data_train(random=0)

        start_time = time.time()
    
        for i in range(nber):
            
            #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


            start_time_tmp = time.time()
                        
            no_valid = True
            while (no_valid):
                cells_array, sum_proba = self.sample_arch(controller, array_normal, array_reduc) 
                try:
                    with strategy.scope():
                        model = self.get_compiled_cnn_model(cells_array)
                    no_valid = False
                except ValueError:
                    print("Oops! not valid arch sampled, trying again...")
                
                
            #callbacks=[tensorboard_callback]
            
            history = self.train_child(X, y, model, global_batch_size, nb_child_epochs, options, callbacks, class_w)
            
            val_acc = max(history.history['val_accuracy'])
            del model
            accuracies.append(val_acc)
            
            print("training child: "+str(i)+ ", time passed: "+str(time.time() - start_time_tmp)+"s, mean acc: "+str(mean(accuracies)))            
            
            if( print_file==1 ):
                with open("accuracies.txt", "a") as f:
                    f.write(str(val_acc)+"\n")
        
        est_time = time.time() - start_time
        
        print("training time per child: "+str(est_time/nber)+" s")
        
        
        print("estimated time for 5000 iterations: "+str((est_time/nber)*5000)+"s, or "+str(((est_time/nber)*5000)/60)+"mins, or "+str(((est_time/nber)*5000)/(60*60))+" hours, or "+ str(((est_time/nber)*5000)/(60*60*24)) +" days."   )
                
        

        
    # Train ONE arch from random data and plot val_loss
    # (used to do experiments on EarlyStopping)
    def simple_train(self, nb_child_epochs, strategy, name_exp, global_batch_size, data_options):

        
        model, array_normal, array_reduc = self.generateController()
        
        X, y = self.load_shaped_data_train()
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
        
        #callbacks = []

        cells_array, __ = self.sample_arch(controller, array_normal, array_reduc)
        
        #cells_array = [[4, 5], [0], [3, 6], [0]]
        
        plt.figure()

        with strategy.scope():
            model = self.get_compiled_cnn_model(cells_array)

        history = self.train_child(X, y, model, global_batch_size, nb_child_epochs, data_options, callbacks, class_w)

        val_loss = history.history['val_loss']
        nbre_epochs = range(len(val_loss))
        plt.plot(nbre_epochs, val_loss, 'b')
        
        plt.savefig("losses_"+str(name_exp)+".png")
        
        print("Losses for experiment: "+str(name_exp))
        for l in val_loss:
            print(l)
        
        
        
        


    def children_stats(self, by_block=1):
                
        controller, array_normal, array_reduc = self.generateController()
        
        count_iter = 0
        
        limit = 100

        dico = {}
       
        while(count_iter < limit):
            
            total_sum=0
            
            with tf.GradientTape(persistent=False) as tape:

                cells_array, __ = self.sample_arch(controller, array_normal, array_reduc)
                #total_sum += sum_proba

                __ , all_blocks = self.get_compiled_cnn_model(cells_array)
                
                ## 

                if( by_block ):

                    for b in all_blocks:

                        # if model is in dico_archs
                        if str(b) in dico:
                            dico[str(b)] += 1
                        else:
                            dico[str(b)] = 1
                else:

                    for b in all_blocks:

                        for l in b:

                            if str(l) in dico:
                                dico[str(l)] += 1
                            else:
                                dico[str(l)] = 1

                
            count_iter += 1


        retour = {k: v for k, v in sorted(dico.items(), key=lambda item: item[1], reverse=True)}

        print(retour)
        
        return

        
    

    def get_compiled_cnn_model_extended(self, cells_array, stack_num, start_nb_filters):

        # Init of the child net
        outputs = []
        input_shape = (8, 900, 1)
        inputs = keras.Input(shape=(8, 900), name="outside_input")
        x = layers.Reshape((8,900,1), name="outside_reshape")(inputs)
        #outputs.append(x)
        x = layers.BatchNormalization(name="outside_batchnorm")(x)
        #outputs.append(x)
        #x = layers.Conv2D(10, (2, 10), padding="same", activation='relu', name="outside_conv2")(x)
        outputs.append(x)
        
        # loop over # of stacks
        
        num_filters = start_nb_filters / 2
        
        for s in range(stack_num):
            
            outputs = []
            outputs.append(x)
            
            num_filters = num_filters * 2

            # loop over the cells and construct the model
            for i, arr in enumerate(cells_array):
                

                if(i>0):
                    outputs = [x]

                if( i%2==0 ):

                    # Conv Cell, cell_inputs must ALWAYS be an array with the possible DIRECT inputs for the cell
                    cell = Cell('conv', cell_inputs=outputs, scheme=self.scheme, num_filters = num_filters)
                    blocks_array = np.array( arr )
                    cell.generateCell( blocks_array, 'conv', num_cell=(s*len(cells_array) + i) )
                    x = cell.cell_output

                else:
                    
                    #if(s == (stack_num - 1)):
                    #    fil = num_filters
                    #else:
                    fil = num_filters * 2
                        
                    cell = Cell('reduc', cell_inputs=outputs, scheme=self.scheme, num_filters = fil)
                    blocks_array = np.array( arr )
                    cell.generateCell(blocks_array, 'reduc', num_cell=(s*len(cells_array) + i) )
                    x = cell.cell_output

        x = layers.Flatten(name="outside_flatten")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25, name="outside_droupout1")(x)
        x = layers.Dense(10)(x)
        x = layers.Dropout(.25, name="outside_droupout2")(x)

        # Create the model
        outputss = layers.Dense(2, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputss, name="ansari_model")
        utils.plot_model(model, to_file="child.png")

        # metrics=['acc',f1_m,precision_m, recall_m]
        # Compute the accuracy
        #acc = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.3)
        loss = keras.losses.CategoricalCrossentropy()
        # f1_m, precision_m, recall_m,tfa.metrics.CohenKappa(num_classes=2)
        model.compile(loss=loss, optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
        
        return model



    def train_model(self, X_train, X_val, y_train, y_val, model, batch_size, epochs_child, options, callbacks, class_w):

        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))

        train_data = train_data.batch(batch_size)
        val_data = val_data.batch(batch_size)
        train_data = train_data.with_options(options)
        val_data = val_data.with_options(options)

        y = out_categorical(y_train)
        class_w = class_weights(y)

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
                class_weight=class_w,
                #validation_split=0.1
            )

        return history     

    
    def cross_val_training(self, model, X, y, X_test, y_test, global_batch_size, data_options, num_stack):

        best_kappa_av = None

        #  10-FOLD
        cv = KFold(10)
        print("len CV SPLIT")
        print(cv.get_n_splits())   #For KFold(10)

        if os.path.exists("tmp_weights.h5"):
            os.remove("tmp_weights.h5")

        y_true, y_pred = list(), list()
        
        for train_ix, val_ix in cv.split(X):

            if os.path.exists("best_weights.h5"):
                model.load_weights('best_weights.h5', by_name=True)


            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath='tmp_weights.h5',
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
            callbacks = [model_checkpoint_callback]


            # split data  ('test' is 'val'
            X_train, X_val = X[train_ix, :], X[val_ix, :]
            y_train, y_val = y[train_ix], y[val_ix]

            epochs_child = 10 * num_stack
            history = self.train_model(X_train, X_val, y_train, y_val, model, global_batch_size, epochs_child, data_options, callbacks, {0:1 , 1:4})

            model.load_weights('tmp_weights.h5')

            predictions = model.predict(X_test)

            predictions = np.argmax(predictions, axis=1)

            #print("prediction after argmax")
            #print(predictions[:10])

            kappa = metrics.cohen_kappa_score(predictions, y_test)

            # Applying moving average
            predictions_av = mvaverage(predictions, 6)
            #real_values_av = mvaverage(y_test, 6)
            kappa_av = metrics.cohen_kappa_score(predictions_av, y_test)


            if best_kappa_av == None:
                best_kappa_av = kappa_av
                model.save_weights('best_weights.h5')


            else:
                if kappa_av > best_kappa_av:
                    best_kappa_av = kappa_av
                    model.save_weights('best_weights.h5')

            print("num_stack "+str(num_stack)+", best_kappa_av "+str(best_kappa_av))

    
    
    def sampling_and_training_one_arch(self, global_batch_size, data_options, weights=""):
        
        
        controller, array_normal, array_reduc = self.generateController()
        
        if(weights != ""):
            controller.load_weights(weights)
        
        cells_array, __ = self.sample_arch(controller, array_normal, array_reduc)
        
        # loop over # of stacks

        if os.path.exists("best_weights.h5"):
            os.remove("best_weights.h5")
        
        # loop over nber of stacked pairs (here 3 pairs)
        for s in range(2):

            # construct a model with s+1 stacked pairs of cells
            model = self.get_compiled_cnn_model_extended(cells_array, s+1, 10)
            
            
            
            # train the model 
            model.summary()

            # train the model
            

            # LOAD TRAIN DATAS
            X, y = self.load_shaped_data_train(random=0, seed=0)

            # LOAD TEST DATAS
            #self.cross_val_training(model, X, y, X_test, y_test, global_batch_size, data_options, s+1)
            X_test, y_test = self.load_shaped_data_test()
            
            
           
            
            
        
        
        
        
        
        
        
        
        
        
