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


        self.num_block_conv = num_block_conv      # Number of blocks (one block = 2 inputs/2 operators) per conv Cell

        self.num_block_reduc = num_block_reduc     # Number of blocks per reduc Cell

        self.num_op_conv = num_op_conv    # Number of operations to choose from when building a conv Cell

        self.num_op_reduc = num_op_reduc    # Number of operations to choose from when building a reduc Cell

        self.num_units = 100    # Number of units in the LSTM layer(s)

        self.num_alt = num_alt # Number of times alterning a conv/reduc cell

        self.scheme = scheme

        self.path_train = path_train

        self.path_test = path_test

    # new 'sampling' function, only return an integer (to feed the Embedding layer)
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

        x = ActivityRegularizationLayer()(x)

        x = layers.Lambda(lambda t: self.sampling(t, depth=num_classes))(x)

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



                if ( self.scheme == 1 ):
                    # for each block/node, we choose 2 inputs and 2 operations

                    if i == 0 and j == 0:
                        _x, _initial = input_cell, True
                    else:
                        _x, _initial = xx, False # output of previous LSTM_softmax

                    _num_classes = self.num_op_conv


                    count+=1
                    xx = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="conv")
                    #x, rx, emb = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, reshaped_inputs=_rx, initial=_initial)

                    outputs.append(xx)


                elif ( self.scheme == 2 ):


                    for o in ["inputL", "operL"]:
                        
                        if i == 0 and j == 0 and o == "inputL":
                            _x, _initial = input_cell, True
                        else:
                            #_x, _rx, _initial = emb, rx, False # output of previous LSTM_softmax
                            _x, _initial = xx, False # output of previous LSTM_softmax

                        if o == "inputL" or o == "inputR" :
                            # 1 softmax de taille 2 pour inputL et inputR
                            _num_classes = j+1
                        else:
                            # 1 softmax de taille #num_op
                            _num_classes = self.num_op_conv


                        count+=1
                        xx = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="conv")
                        #x, rx, emb = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, reshaped_inputs=_rx, initial=_initial)

                        outputs.append(xx)


                elif ( self.scheme == 3 ):


                    for o in ["operL", "operR"]:

                        if i == 0 and j == 0 and o == "operL":
                            _x, _initial = input_cell, True
                        else:
                            _x = xx
               
                        _num_classes = self.num_op_conv
                 

                        count+=1
                        xx = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="conv")
                        #x, rx, emb = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, reshaped_inputs=_rx, initial=_initial)

                        outputs.append(xx)


                else: # scheme == 4

                    for o in ["inputL", "inputR", "operL", "operR"]:

                        if i == 0 and j == 0 and o == "inputL":
                            _x, _initial = input_cell, True
                        else:
                            #_x, _rx, _initial = emb, rx, False # output of previous LSTM_softmax
                            _x, _initial = xx, False # output of previous LSTM_softmax

                        if o == "inputL" or o == "inputR" :
                            # 1 softmax de taille 2 pour inputL et inputR
                            _num_classes = j+1
                        else:
                            # 1 softmax de taille #num_op
                            _num_classes = self.num_op_conv



                        count+=1
                        xx = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="conv")
                        #x, rx, emb = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, reshaped_inputs=_rx, initial=_initial)

                        outputs.append(xx)



            # Generate the reduc blocks
            for j in range(0, self.num_block_reduc):

                _x, _initial = xx, False # output of previous LSTM_softmax

                _num_classes = self.num_op_reduc

                count+=1
                xx = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, initial=_initial, count=count, type_cell="reduc")
                #x, rx, emb = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, reshaped_inputs=_rx, initial=_initial)

                outputs.append(xx)

        model = Model(inputs=controller_input, outputs=outputs)


        return model






    def get_compiled_cnn_model(self, cells_array):


        # Init of the child net
        outputs = []
        input_shape = (8, 900, 1)
        inputs = keras.Input(shape=(8, 900), name="outside_input")
        x = layers.Reshape((8,900,1), name="outside_reshape")(inputs)
        #outputs.append(x)
        x = layers.BatchNormalization(name="outside_batchnorm")(x)
        #outputs.append(x)
        x = layers.Conv2D(10, (2, 10), padding="same", activation='relu', name="outside_conv2")(x)
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
        utils.plot_model(model, to_file="child.png")

        # metrics=['acc',f1_m,precision_m, recall_m]
        # Compute the accuracy
        #acc = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.3)
        loss = keras.losses.CategoricalCrossentropy()
        # f1_m, precision_m, recall_m,tfa.metrics.CohenKappa(num_classes=2)
        model.compile(loss=loss, optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
        
        return model



    def load_shaped_data_train(self, random=0):


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






    def sample_arch(self, controller):

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
        sum_proba = 0


        ###################
        # MAKING CHILD ARCH
        ###################


        if(self.scheme==1):
            v = 1
        elif(self.scheme==2):
            v = 2
        elif(self.scheme==3):
            v = 2
        else:
            v=4

        # loop over each layer (choice)
        for i in range(0, len(layer_outs), 2):

            classe = layer_outs[i+1][0][0]
            proba = controller.losses[int(i/2)][0][0][classe]
            sum_proba -= tf.math.log(proba)

            # when layer is for a convCell choice
            if ( k < self.num_block_conv*v ):
                cell1.append(classe)                    
                
                #print("k: "+str(k)+ " "+str(cell1))
                if(k==(self.num_block_conv*v-1)):
                    cells_array.append(cell1)
                    cell1=[]
                k+=1

            # when layer is for a reducCell choice
            else:
                cell1=[]
                if(u<self.num_block_reduc):
                    #print("u: "+str(u))
                    cell2.append(classe)
                    if(u==(self.num_block_reduc-1)):
                        cells_array.append(cell2)
                        cell2=[]
                    u+=1
                else:
                    k=0
                    u=0
                if(u==self.num_block_reduc):
                    k=0
                    u=0
            count+=1
            

        return cells_array, sum_proba



    def train_child(self, X, y, model, batch_size, epochs_child, options):
                    


        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True)
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
        train_data = train_data.batch(batch_size)
        val_data = val_data.batch(batch_size)
        train_data = train_data.with_options(options)
        val_data = val_data.with_options(options)
        

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss')

        history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs_child,
                batch_size=batch_size,
                #callbacks=callbacks,
                verbose=1,
                #class_weight=class_weight,
                #validation_split=0.1
            )
       

        return history     


    def train(self, epochs=5, epochs_child=2):

        
        X, y = self.load_shaped_data_train(random=1)

        controller = self.generateController()

        sampling_number = 1 # number of arch child to sample at each epoch

        optimizer = keras.optimizers.SGD(learning_rate=1e-3)

        accuracies = []
        mean_acc = []
        
        strategy = tf.distribute.MirroredStrategy()

        #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        total_weights = h5py.File("total_weights.h5", "w")

        # Loop over the epochs
        for epoch in range(epochs):

            #print("Epoch: ",epoch, " time: ", time.time() - start)

            with tf.GradientTape(persistent=False) as tape:

                total_sum = 0

                # Loop over the number of samplings
                for s in range(sampling_number):

                    val_acc = tf.Variable(0)
                    cells_array, sum_proba = self.sample_arch(controller)
                    total_sum += sum_proba

                    with strategy.scope():
                        model = self.get_compiled_cnn_model(cells_array)

                    model.load_weights("total_weights.h5", by_name=True)

                    history = self.train_child(X, y, model, 32, 10)
                    
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
                        f = open("log_Controller_ENAS2.txt", "a")
                        f.write("ema: "+str(ema)+" mean_acc = "+str(mean_acc[-1])+" Epoch: "+str(epoch)+ " time: "+ str(time.time() - start)+"\n")
                        f.close()
                        

                    
                    total_sum *= ( val_acc )
                    del model

                total_sum/=sampling_number
                
            grads = tape.gradient(total_sum, controller.trainable_weights)
            optimizer.apply_gradients(zip(grads, controller.trainable_weights))

        if(len(mean_acc)>0):
            controller.save_weights("Controller_weights_.h5")

        total_weights.close()


        
    # MEASUREMENT TOOLS


    def search_space_size(self):
        
        return self.num_alt*self.num_block_conv*self.num_block_reduc*self.num_op_conv*self.num_op_reduc
        
        

    def best_epoch(self, strategy, global_batch_size=64):
        
        nb_child_epochs = 20
        ite=10
        
        # sample and train 10 child archs  on  20 epochs each

        # PLOT all eval_loss

        X, y = self.load_shaped_data_train(random=0)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        
        controller = self.generateController()

        plt.figure()

        start = time.time()
        
        for i in range(ite):

            cells_array, __ = self.sample_arch(controller)
            
            with strategy.scope():
                model = self.get_compiled_cnn_model(cells_array)

            history = self.train_child(X, y, model, global_batch_size, nb_child_epochs, options)
            val_loss = history.history['val_loss']
            del model
            nbre_epochs = range(len(val_loss))
            plt.plot(nbre_epochs, val_loss, 'b')
        
        plt.savefig('all_losses.png')



    # check if the train_metric (used during training of each child)
    # "follows" the "final" metrics (kappa score), ie, on the test set
    #
    
    def match_train_test_metrics(self, train_metric, test_metric, nb_child_epochs, strategy, global_batch_size):



        X, y = self.load_shaped_data_train(random=0)

        X_test, y_test = self.load_shaped_data_test()

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        
        controller = self.generateController()

        train_metrics = []
        
        test_metrics = []

        for i in range(50):

            cells_array, __ = self.sample_arch(controller)

            with strategy.scope():
                model = self.get_compiled_cnn_model(cells_array)

            history = self.train_child(X, y, model, global_batch_size, nb_child_epochs, options)
            val_acc = history.history[train_metric][-1]

            train_metrics.append(val_acc)

            # then test network on test set

            predictions = model.predict(X_test)
            del model
            predictions = np.argmax(predictions, axis=1)
            real_values = y_test

            test_metric_value = 0
            if(test_metric == 'kappa_score'):
                test_metric_value = metrics.cohen_kappa_score(predictions, real_values)
            else:
                test_metric_value = metrics.accuracy_score(predictions, real_values)

            test_metrics.append(test_metric_value)


        plt.figure()

        plt.plot(np.arange(len(train_metrics)), train_metrics, 'o', color='blue')

        plt.plot(np.arange(len(test_metrics)), test_metrics, 'o', color='red')

        plt.savefig('match_test_train_metrics.png')
        
        
    # inc: considered increase of the metrics value
    # nb_child_epochs : nb of epochs of child training
    # return nber of necessary iterations meeting the input conditions
    
    def test_nb_iterations(self, file_accs = 'accuracies.txt', inc=0.16, nb_child_epochs=5, mva=500, limit=30000):
                
        if (inc > 0.4 or inc <= 0):
            raise Exception("Sorry, inc must be < 0.4 and > 0") 
            
        controller = self.generateController()
        strategy = tf.distribute.MirroredStrategy()

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
        
                
        #accuracies = [ 0.2, 0.8, 0.8, 0.8, 0.9, 0.7, 0.2 ]


        # 2) retrieve the freq of distri of accuracies in N quantiles
        N=2
        dico = frequency_distr(N, accuracies) # { mean_acc : freq }
        
        
        # 3) loop over dico and build dico_archs
        
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
        means_accs = []
        count_iter = 0
        
        while(count_iter < limit):
            
            total_sum=0
            with tf.GradientTape(persistent=False) as tape:

                cells_array, sum_proba = self.sample_arch(controller)
                total_sum += sum_proba
                
                hash_ = hash(str(cells_array))
                
                
                # if model is in dico_archs
                if hash_ in dico_archs:
                    acc = dico_archs[hash_]
                else:
                    # for the given freq distri returns an accuracy
                    acc = random.choices(list(dico.keys()), weights=list(dico.values()))[0]
                    dico_archs[hash_] = acc
                                
                
                all_accs.append(acc)
        
                if( len(all_accs) > mva ):
                
                    means_accs.append(mean(all_accs[-mva:]))

                    print("iter: "+str(count_iter)+" mean_acc: "+str(means_accs[-1]))
                    
                    if(count_iter % 1000):

                        plt.figure()
                            
                        plt.plot(np.arange(len(means_accs))+mva, means_accs, 'b')    

                        plt.title("Moving average with lag of "+str(mva))
                        plt.savefig('incre.png')

                    
                    # if( (means_accs[-1] - means_accs[0]) > inc):
                    #     print("nber of iter to increase acc by "+ str(inc) + " : "+ str(count_iter))
                    #     return
                    
                    #print("mean[0] = "+str(means_accs[0])+" mean[-1] = "+str(means_accs[-1]))
                    
                total_sum *= ( acc )
            
            
            if(means_accs[-1] - means_accs[0] > inc):
                print(str(counter_iter)+" iterations were needed for an increase of "+str(inc)+" of the moving average")
            
            count_iter+=1
            grads = tape.gradient(total_sum, controller.trainable_weights)
            optimizer.apply_gradients(zip(grads, controller.trainable_weights))
        
        return
        
    # compute and store nber accuracies
    # in file accuracies.txt
    # and displays stats
    def compute_accuracies(self, nber, nb_child_epochs, strategy, global_batch_size=64, print_file=0):

        
        controller = self.generateController()
        
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        
        accuracies = []
        
        start_time = time.time()
        
        cells_array, __ = self.sample_arch(controller)

        
        X, y = self.load_shaped_data_train(random=0)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    
        start_time = time.time()
    
        for i in range(nber):
            
            #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


            start_time_tmp = time.time()
                        
            cells_array, __ = self.sample_arch(controller)

            with strategy.scope():
                model = self.get_compiled_cnn_model(cells_array)

            #callbacks=[tensorboard_callback]
            
            history = self.train_child(X, y, model, global_batch_size, nb_child_epochs, options)
            val_acc = history.history['val_accuracy'][-1]
            del model
            accuracies.append(val_acc)
            
            print("training child: "+str(i)+ ", time passed: "+str(time.time() - start_time_tmp)+"s")

            if( print_file==1 ):
                with open("accuracies.txt", "a") as f:
                    f.write(str(val_acc)+"\n")
            
        
        
        est_time = time.time() - start_time
        
        print("training time per child: "+str(est_time/nber)+" s")
        
        
        print("estimated time for 5000 iterations: "+str((est_time/nber)*5000)+"s, or "+str(((est_time/nber)*5000)/60)+"mins, or "+str(((est_time/nber)*5000)/(60*60))+" hours, or "+ str(((est_time/nber)*5000)/(60*60*24)) +" days."   )
                
        
        
