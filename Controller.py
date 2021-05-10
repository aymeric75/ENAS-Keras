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

        x = layers.Lambda(lambda t: sampling(t, depth=num_classes))(x)

        x = layers.Embedding(num_classes, 100)(x)


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

    num_block_conv=3
    num_block_reduc=2

    num_alt=2
    # num_op = 5  <=====> classes = ["Conv2D_2_10", "Conv2D_1_5", "Conv2D_4_1", "MaxPooling2D_2", "MaxPooling2D_1_6"]

    controllerInst = Controller(num_block_conv=num_block_conv, num_block_reduc=num_block_reduc, num_op_conv=5, num_op_reduc=2, num_alt=num_alt)
    controller = controllerInst.generateController()
    utils.plot_model(controller, to_file="controller_example2.png")


    ###############
    # TRAINING LOOP
    ###############


    n = 5 # number of arch to sample (= size of the batch ?)

    train_dataset = [[1] for i in range(n)] # len =  number of arch to sample



    epochs = 1




    # Loop over the epochs (Unnecessary, to remove!)
    for epoch in range(epochs):



        # Loop over the 
        for step, (x_batch_train) in enumerate(train_dataset):

            with tf.GradientTape(persistent=True) as tape:


                # FEEDFORWARD
                inp = controller.input  # input placeholder
                outputs = [layer.output for layer in controller.layers if ( layer.__class__.__name__ == "Dense" or layer.__class__.__name__ == "Lambda" )]  # all layer outputs
                functor = K.function([inp], outputs )   # evaluation function
                test = np.random.random((1,1))[np.newaxis,...]
                layer_outs = functor([test, 1.]) # Here are all the outputs of all the layers



                # sum over the hyperparameters (i.e, over the choices made my the RNN)
                sum_choices = 0


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



                    # Class that was chosen by the RNN
                    classe = layer_outs[i+1][0][0]

                    # Proba of having chosen class 'classe' knowing the previous choices
                    proba = layer_outs[i][0][0][layer_outs[i+1][0][0]]
                    log_proba = np.log(proba)
                    #grad = tape.gradient(tf.convert_to_tensor(proba.item()), controller.trainable_weights)


                    # when layer is for a convCell choice
                    if ( k < 12 ):
                        cell1.append(classe)
                        #print("k: "+str(k)+ " "+str(cell1))
                        if(k==11):
                            cells_array.append(cell1)
                            cell1=[]
                        k+=1

                    # when layer is for a reducCell choice
                    else:
                        cell1=[]
                        if(u<2):
                            #print("u: "+str(u))
                            cell2.append(classe)
                            if(u==1):
                                cells_array.append(cell2)
                                cell2=[]
                            u+=1
                        else:
                            k=0
                            u=0
                        if(u==2):
                            k=0
                            u=0
                    count+=1




                # Init of the child net
                outputs = []
                input_shape = (8, 900, 1)
                inputs = keras.Input(shape=(8, 900))
                x = layers.Reshape((8,900,1))(inputs)
                #outputs.append(x)
                x = layers.BatchNormalization()(x)
                #outputs.append(x)
                x = layers.Conv2D(10, (2, 10), padding="same", activation='relu')(x)
                outputs.append(x)

                # loop over the cells and construct the model
                for i, arr in enumerate(cells_array):

                    if(i>0):
                        outputs = [x]

                    if(i%2==0):

                        # Conv Cell, cell_inputs must ALWAYS be an array with the possible DIRECT inputs for the cell
                        cell = Cell('conv', cell_inputs=outputs)
                        blocks_array = np.array( arr )
                        cell.generateCell(blocks_array, 'conv')
                        x = cell.cell_output

                    else:

                        cell = Cell('reduc', cell_inputs=outputs)
                        blocks_array = np.array( arr )
                        cell.generateCell(blocks_array, 'reduc')
                        x = cell.cell_output


                x = layers.Flatten()(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.25)(x)
                x = layers.Dense(10)(x)
                x = layers.Dropout(.25)(x)

                # Create the model
                outputss = layers.Dense(2, activation='softmax')(x)
                model = keras.Model(inputs=inputs, outputs=outputss, name="ansari_model")
                utils.plot_model(model)

                # Compute the accuracy
                loss = keras.losses.CategoricalCrossentropy()
                model.compile(loss=loss, optimizer=keras.optimizers.Adam(0.1), metrics=['accuracy'])

                callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=1) #min_delta=0.1,

                loaded = load_data("data_conv_training.mat", "data_conv_testing.mat",clip_value=300)
                X_train = loaded[0].transpose((0,2,1)) # eeg training
                y_train = loaded[1]
                y_train = keras.utils.to_categorical(y_train, num_classes=2)



                history = model.fit(
                    X_train,
                    y_train,
                    epochs=25,
                    #batch_size=1,
                    callbacks=[callback],
                    validation_split=0.1,
                )

                #loss = history.history['loss']

                val_acc = history.history['val_accuracy']


                print("val_acc")
                print(val_acc)
                

                
                break

if __name__ == "__main__":

    main()