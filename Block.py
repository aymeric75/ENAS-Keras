# import numpy as np
# import os
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

from tensorflow import keras
from tensorflow.keras import layers

import random

###################
#   Defines a block
###################




def randomgen():

    random_string = ''

    for i in range(4):
        # Considering only upper and lowercase letters
        random_integer = random.randint(97, 97 + 26 - 1)
        flip_bit = random.randint(0, 1)
        # Convert to lowercase if the flip bit is on
        random_integer = random_integer - 32 if flip_bit == 1 else random_integer
        # Keep appending random characters using chr(x)
        random_string += (chr(random_integer))
     
    return random_string




class Block():

    def __init__(self, name='', input1='', input2='', op1 = '', op2 = '', comb = 'add', output = '', inputs='', cell_type='conv', name_cell='', num_cell=''):

        self.name = name
        self.input1 = input1
        self.input2 = input2
        self.op1 = op1
        self.op2 = op2
        self.comb = comb
        self.output = output
        self.inputs = inputs
        self.cell_type = cell_type
        self.name_cell = name_cell
        self.num_cell = num_cell



        
    def construct(self):


        # TO be defined elsewhere
        conv_layers = [
                layers.Conv2D(10, (2, 10), padding="same", activation='relu', name=self.name+"-2_10"),
                layers.Conv2D(10, (2, 5), padding="same", activation='relu', name=self.name+"-2_5"),
                layers.Conv2D(10, (2, 5), dilation_rate=4, padding="same", activation='relu', name=self.name+"-2_5_dl-4"),
                layers.Conv2D(10, (2, 10), dilation_rate=4, padding="same", activation='relu', name=self.name+"-2_10_dl-4"),
                layers.Conv2D(10, (8, 1), padding="same", activation='relu', name=self.name+"-8_1"),            
                layers.Conv2D(10, (1, 900), padding="same", activation='relu', name=self.name+"-1_900"),
                layers.Dropout(0.2, name=self.name+"-dropout"),
                layers.LeakyReLU(name=self.name+"-leaky")
        ]


        reduc_layers = [
            layers.MaxPooling2D(pool_size=(2,2), padding="same", name=self.name+"-pool_2_2"),
            layers.MaxPooling2D(pool_size=(1, 6), padding="same", name=self.name+"-pool_1_6")
        ]
        
        the_dropout_layer = None

        if ( self.cell_type == 'conv' ):
            the_layers = conv_layers
            #the_dropout_layer = layers.Dropout(0.2, name=self.name+"-dropout")
            #the_leaky_layer = layers.LeakyReLU(name=self.name+"-leaky")

        else:
            the_layers = reduc_layers


        if(self.comb == 'add'):
            self.comb = layers.Add( name = self.name+"_add" )

        
        if(self.input1 != None and self.input2 == None): # if input1 set to None, just consider input2
            self.output = (the_layers[self.op1])(self.inputs[self.input1])
        
        elif(self.input2 != None and self.input1 == None): # if input2 set to None, just consider input1
            self.output = (the_layers[self.op2])(self.inputs[self.input2])
         
        #elif(self.input1 == self.input2 and self.op1 == self.op2): # if inputs and ops same, as if block with one input and one op
        elif( tf.math.equal(self.input1,self.input2) and tf.math.equal(self.op1, self.op2) ):
                self.output = (the_layers[self.op2])(self.inputs[self.input2])
                
        else: # 

                self.output = (self.comb)([(the_layers[self.op1])(self.inputs[self.input1]), (the_layers[self.op2])(self.inputs[self.input2])])

        #print(self.output)





def main():

    return 1




if __name__ == "__main__":
    #block = Block()
    main()