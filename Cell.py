import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.models import Model

from keras import initializers, regularizers
from Controller import Controller
from Block import Block

import numpy as np



class Cell():



    def __init__(self, type_cell = 'conv', cell_input = '', cell_output=''):

        self.type_cell = type_cell
        self.cell_input = cell_input
        self.cell_output = cell_output



    def generateCell(self, blocks_array=[], outputs=[]):


        x = self.cell_input


        if(self.type_cell == 'conv'):

            used_inputs = []


            for i in range(0, len(blocks_array), 4):


                print("i = "+str(i))

                used_inputs.append(blocks_array[i])
                used_inputs.append(blocks_array[i+1])

                new_block = Block('block'+str(i), input1=blocks_array[i], input2=blocks_array[i+1], op1=blocks_array[i+2], op2=blocks_array[i+3], inputs=outputs)

                new_block.construct(i)

                print(new_block.output)

                outputs.append(new_block.output)


            outputs_ind = np.arange(len(outputs))
            
            used_inputs = np.unique(np.array(used_inputs))


            unused_outputs = np.setdiff1d(np.union1d(used_inputs, outputs_ind), np.intersect1d(used_inputs, outputs_ind))

            outputs = np.array(outputs)


            if(len(unused_outputs) == 1):
                self.cell_output = outputs[unused_outputs[0]]

            else:
                self.cell_output = layers.Concatenate()(list(outputs[unused_outputs]))


        else: # type_cell = reduc
            outputs = list(outputs)
            self.cell_output = layers.MaxPooling2D( (2,2) )(outputs[-1])
            outputs.append(self.cell_output)


        return outputs


def main():

    outputs = []

    # REPRENDRE ICI !!!!!!
    input_shape = (8, 900, 1)
    inputs = keras.Input(shape=(8, 900))
    x = layers.Reshape((8,900,1))(inputs)
    
    x = layers.BatchNormalization()(x)
    outputs.append(x)

    convCell = Cell('conv', cell_input=x)

    # input, input, op, op etc.
    blocks_array = np.array([0,0,3,3,1,1,2,2,2,2,3,3])


    outputs = convCell.generateCell(blocks_array, outputs)


    reducCell = Cell('reduc')

    outputs = reducCell.generateCell([], outputs)

    



    outputss = layers.Dense(2, activation='softmax')(reducCell.cell_output)
    model = keras.Model(inputs=inputs, outputs=outputss, name="ansar_model")
    
    model.summary()

    utils.plot_model(model)

    exit()









if __name__ == "__main__":

    main()
