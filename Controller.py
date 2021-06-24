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
import Utils
from Utils import ActivityRegularizationLayer
from Utils import moving_average, mvaverage, recall_m, precision_m, f1_m, kappa, truncate_float, duplicate_in_array
import matplotlib.pyplot as plt


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

    def __init__(self, num_block_conv=2 , num_block_reduc=2, num_op_conv=5, num_op_reduc=2, num_alt=5, scheme=1):


        self.num_block_conv = num_block_conv      # Number of blocks (one block = 2 inputs/2 operators) per conv Cell

        self.num_block_reduc = num_block_reduc     # Number of blocks per reduc Cell

        self.num_op_conv = num_op_conv    # Number of operations to choose from when building a conv Cell

        self.num_op_reduc = num_op_reduc    # Number of operations to choose from when building a reduc Cell

        self.num_units = 100    # Number of units in the LSTM layer(s)

        self.num_alt = num_alt # Number of times alterning a conv/reduc cell

        self.scheme = scheme




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






    def train(self, best_epoch=0, epochs_child=2):

        tracemalloc.start()
        
        #loaded = load_data("../../../../data_conv_training.mat", "../../../../data_conv_testing.mat",clip_value=300)
        #X = loaded[0].transpose((0,2,1)) # eeg training
        #y = loaded[1] 
        #y = keras.utils.to_categorical(y, num_classes=2)

        

        X = np.random.rand(10000,8,900)
        y = np.random.choice(2, 10000)
        y = keras.utils.to_categorical(y, num_classes=2)
        
        
        
        # Init data option & disable AutoShard.
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        
        ##########################
        # Controller Instanciation
        ##########################



        controller = self.generateController()
        #utils.plot_model(controller, to_file="controller_example2.png")

        
        ###############
        # TRAINING LOOP
        ###############


        epochs = 5
        
        
        sampling_number = 1 # number of arch child to sample at each epoch

        sum_over_choices = 0 # outer sum of the policy gradient


        optimizer = keras.optimizers.SGD(learning_rate=1e-3)

        accuracies = []
        mean_acc = []
        
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        
        #total_weights = h5py.File("total_weights.h5", "w")
        
        
        start = time.time()

        
        # Loop over the epochs
        for epoch in range(epochs):

            #print("Epoch: ",epoch, " time: ", time.time() - start)

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
                        total_sum -= tf.math.log(proba)

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
                        
                    

                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True)

                    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                                                    
                    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
                    
                    
                    batch_size = 32

                    train_data = train_data.batch(batch_size)
                    val_data = val_data.batch(batch_size)

                    train_data = train_data.with_options(options)
                    val_data = val_data.with_options(options)
                    

                    with strategy.scope():

                        model = self.get_compiled_cnn_model(cells_array)


                    callback = tf.keras.callbacks.EarlyStopping(monitor='loss')

                    classes_y = np.argmax(y_train, axis=1)
                    
                    #weight_for_0 = 1.0 / len(classes_y[classes_y==0])
                    #weight_for_1 = 1.0 / len(classes_y[classes_y==1])
                        
                    #classes_y = np.argmax(y_train, axis=1)
                    #print("nbre 1 / nbre de 0 = "+str(len(classes_y[classes_y==1])/len(classes_y[classes_y==0])))

                        
                    #class_weight = {0: weight_for_0, 1: weight_for_1}
                        
                    if(best_epoch==1):
                        epochs_child=10

                    history = model.fit(
                        train_data,
                        validation_data=val_data,
                        epochs=epochs_child,
                        batch_size=32,
                        #callbacks=[callback],
                        verbose=1,
                        #class_weight=class_weight,
                        #validation_split=0.1
                    )

                    
                    val_loss = history.history['val_loss']
                    epochs = range(len(val_loss))
                    
                    if (best_epoch==1):
                        plt.plot(epochs, val_loss, 'b')
                    

                    
                    predictions = model(X_val)
                    predictions = np.argmax(predictions, axis=1)
                    real_pred = np.argmax(y_val, axis=1)
                    
                    
                    """
                    metric = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=False)
                    metric.update_state(real_pred , predictions)
                    result = metric.result()
                    print("kappa_tfa : "+str(result.numpy()))
                    """
                   
                                                    
                    
                    val_acc = history.history['val_accuracy'][-1]
                
                    
                    
                    #val_acc = history.history['val_accuracy'][-1]
                    #print(val_acc)
                    
                    """
                    exit()
                    
                    # cells_array
                    
                    #print(cells_array)
                    if(cells_array[0][0] == 1):
                        val_acc = 0.80
                    else:
                        val_acc = 0.2
                    
                    
                    accuracies.append(val_acc)
                    b=0
                    if (len(accuracies) > 10):
                        mean_acc = mean(accuracies[-10:])
                        b = moving_average(accuracies, 10, type='exponential')
                        print("mean = "+str(mean_acc))
                        f = open("log_Controller_ENAS2.txt", "a")
                        f.write(str(mean_acc)+" Epoch: "+str(epoch)+ " time: "+ str(time.time() - start)+"\n")
                        f.close()
                      
                    
                    total_sum *= ( val_acc - b )
                    """

                    
                total_sum/=sampling_number
                
            
            grads = tape.gradient(total_sum, controller.trainable_weights)
            optimizer.apply_gradients(zip(grads, controller.trainable_weights))

        
            
        print("total allocated memory")
        print(tracemalloc.get_traced_memory())
        tracemalloc.stop()





    def best_epoch(self):

        plt.figure()

        self.train(best_epoch=1)

        plt.savefig('all_losses.png')


