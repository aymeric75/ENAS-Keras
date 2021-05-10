# import numpy as np
# import os
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

from tensorflow import keras
from tensorflow.keras import layers



###################
#   Defines a block
###################



class Block():

    def __init__(self, name='', input1='', input2='', op1 = '', op2 = '', comb = 'add', output = '', inputs='', cell_type='conv'):

        self.name = name
        self.input1 = input1
        self.input2 = input2
        self.op1 = op1
        self.op2 = op2
        self.comb = comb
        self.output = output
        self.inputs = inputs
        self.cell_type = cell_type

    #@tf.function
    def construct(self):


        # TO be defined elsewhere
        conv_layers = [
                layers.Conv2D(10, (2, 10), padding="same", activation='relu'),
                layers.Conv2D(10, (1, 5), padding="same", activation='relu'), 
                layers.Conv2D(10, (4, 1), padding="same", activation='relu'),
                layers.Conv2D(10, (8, 1), padding="same", activation='relu'),            
                layers.Conv2D(10, (1, 900), padding="same", activation='relu'), 
            ]


        reduc_layers = [
            layers.MaxPooling2D(pool_size=(2,2), padding="same"),
            layers.MaxPooling2D(pool_size=(1, 6), padding="same")
        ]

        if ( self.cell_type == 'conv' ):
            the_layers = conv_layers

        else:
            the_layers = reduc_layers


        if(self.comb == 'add'):
            self.comb = layers.Add()


        if(self.input1 != None and self.input2 == None): # if input1 set to None, just consider input2
            self.output = (the_layers[self.op1])(self.inputs[self.input1])
        elif(self.input2 != None and self.input1 == None): # if input2 set to None, just consider input1
            self.output = (the_layers[self.op2])(self.inputs[self.input2])
        elif(self.input1 == self.input2 and self.op1 == self.op2): # if inputs and ops same, as if block with one input and one op
            self.output = (the_layers[self.op2])(self.inputs[self.input2])
        else: # normal construction (2 inputs, 2 ops)

            #print("self.op1 = "+str(self.op1)+"  self.input1 = "+str(self.input1))
            #print("self.op2 = "+str(self.op2)+"  self.input2 = "+str(self.input2))

            # Reshape input shape of 

            #  Reshape()
            self.output = (self.comb)([(the_layers[self.op1])(self.inputs[self.input1]), (the_layers[self.op2])(self.inputs[self.input2])])

        #print(self.output)





def main():


    initializer = keras.initializers.Zeros()



    input_shape = (8, 900, 1)
    inputs = keras.Input(shape=(8, 900))
    x = layers.Reshape((8,900,1))(inputs)


    block1 = Block()
    block1.name = "BatchNorm1"
    block1.input1 = x
    block1.input2 = None
    block1.op1 = layers.BatchNormalization()


    block2 = Block()
    block2.name = "Conv2_1"
    block2.input1 = block1.output
    block2.input2 = None
    block2.op1 = layers.Conv2D( 10, (2, 10), padding="same", activation='relu', input_shape=input_shape[:-1] )


    exit()


    block3 = Block()
    block3.name = "MaxPool_1"

    block4 = Block()
    block4.name = "Conv2_2"

    block5 = Block()
    block5.name = "Conv2_3"

    block6 = Block()
    block6.name = "MaxPool_2"





if __name__ == "__main__":
    #block = Block()
    main()