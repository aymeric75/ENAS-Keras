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
import random








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
                        #if ( i > 0 and j == 0 ):
                        #    _num_classes = 1
                        #else:
                        #    _num_classes = j+2
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



def main():


    ##########################
    # Controller Instanciation
    ##########################

    num_block_conv=1
    num_block_reduc=1

    num_alt=1
    # num_op = 5  <=====> classes = ["Conv2D_2_10", "Conv2D_1_5", "Conv2D_4_1", "MaxPooling2D_2", "MaxPooling2D_1_6"]

    controllerInst = Controller(num_block_conv=num_block_conv, num_block_reduc=num_block_reduc, num_op_conv=5, num_op_reduc=2, num_alt=num_alt)
    controller = controllerInst.generateController()
    utils.plot_model(controller, to_file="controller_example2.png")




    ###############
    # TRAINING LOOP
    ###############


    n = 5 # number of arch to sample (= size of the batch ?)

    epochs = 1

    sampling_number = 5000 # number of arch child to sample at each epoch

    sum_over_choices = 0 # outer sum of the policy gradient


    optimizer = keras.optimizers.SGD(learning_rate=1e-3)


    counter = 0

    # Loop over the epochs
    for epoch in range(epochs):


        sum_over_samples = 0 # inner sum of the policy gradient

        # Loop over the number of samplings
        for s in range(sampling_number):

            #sum_over_samples += sum_over_choices


            counter+=1


            with tf.GradientTape(persistent=False) as tape:

                sampling_number = 50

                # FEEDFORWARD
                inp = controller.input  # input placeholder
                outputs = [layer.output for layer in controller.layers if ( layer.__class__.__name__ == "Dense" or layer.__class__.__name__ == "Lambda" )]  # all layer outputs
                functor = K.function([inp], outputs )   # evaluation function
                test = np.random.random((1,1))[np.newaxis,...]
                layer_outs = functor([test]) # Here are all the outputs of all the layers

                #print(controller.input.shape)


                #logits = controller([1], training=True)


                # sum over the hyperparameters (i.e, over the choices made my the RNN)
                #sum_over_choices = 0

                # final array of all blocks/cells
                cells_array = []

                cell1 = [] # tmp array for conv cells
                cell2 = [] # tmp array for reduc cells

                k=0
                u=0
                count=0


                #print(controller.losses)
                

                ###################
                # MAKING CHILD ARCH
                ###################

                # loop over each layer (choice)
                for i in range(0, len(layer_outs), 2):

                    #print("DEBUT")

                    classe = layer_outs[i+1][0][0]

                    proba = controller.losses[int(i/2)][0][0][classe]

                    rew = 0.2



                    nb_classes = layer_outs[i][0][0].shape[0]

                    if ( nb_classes == 5 ):

                        if ( classe == 4 ):
                            rew = 99

                    else:

                        if ( classe == 0 ):
                            rew = 99


                    sum_over_choices -= tf.math.log(proba)*rew

                    #sum_over_choices = tf.divide(sum_over_choices, sampling_number)

                    print(str(classe)+"   "+str(counter))


                    # i+1 <=> to the "losses" layer

                    #proba = layer_outs[i+1][0][0][layer_outs[i+2][0][0]]


                    
                    #print(controller.layers[i+2].losses)
                    #print(layer_outs[i+1].losses)


                    # # Class that was chosen by the RNN
                    

            

                    # DIRE QUE SI CLASSE 0 PARTOUT ALORS REWARD DE 1.2 SINON DE 0.2

                    #   total



                    # # Proba of having chosen class 'classe' knowing the previous choices
                    # proba = layer_outs[i][0][0][layer_outs[i+1][0][0]]
                    # log_proba = tf.math.log(proba)
                    # #grad = tape.gradient(tf.convert_to_tensor(proba.item()), controller.trainable_weights)


                # log_proba = np.log(log_proba)
                # sum_over_choices += log_proba * val_acc

        


            grads = tape.gradient(sum_over_choices, controller.trainable_weights)

            optimizer.apply_gradients(zip(grads, controller.trainable_weights))


    #controller.save_weights("controller_weights.h5")


if __name__ == "__main__":

    main()