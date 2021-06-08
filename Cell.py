import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from keras import initializers, regularizers
#from Controller import Controller
from Block import Block
import numpy as np



class Cell():



    def __init__(self, type_cell = 'conv', cell_inputs = [], cell_output=''):

        self.type_cell = type_cell
        self.cell_inputs = cell_inputs
        self.cell_output = cell_output



    def generateCell(self, blocks_array, type_cell, num_cell):


        #x = self.cell_input

        outputs = self.cell_inputs

        chosen_outputs = []


        if(self.type_cell=='conv'):


            


            for i in range(0, len(blocks_array), 4):


                chosen_outputs.append(blocks_array[i])
                chosen_outputs.append(blocks_array[i+1])
                
                block_name = 'cell-'+str(num_cell)+'-block-'+str(int(i/4))+"-"+str(blocks_array[i])+"-"+str(blocks_array[i+1])+"-"+str(blocks_array[i+2])+"-"+str(blocks_array[i+2])
                

                new_block = Block(block_name, input1=blocks_array[i], input2=blocks_array[i+1], op1=blocks_array[i+2], op2=blocks_array[i+3], inputs=outputs, cell_type=self.type_cell, num_cell=num_cell)

                new_block.construct()


                outputs.append(new_block.output)

        else:

            for i in range(0, len(blocks_array)):


                new_block = Block('cell-'+str(num_cell)+'-block-'+str(i), input1=i, input2=i, op1=blocks_array[i], 
                    op2=blocks_array[i], inputs=outputs, cell_type=self.type_cell, num_cell=num_cell)

                new_block.construct()

                #print(new_block.output)

                outputs.append(new_block.output)



        if(self.type_cell=='conv'):

            # retrieve all unchosen outputs and add them 
            unchosen_outputs = []
            for i, out in enumerate(outputs):
                if (i not in chosen_outputs):
                    unchosen_outputs.append(out)

            # merge all unplugged outputs
            if ( len(unchosen_outputs) > 1 ):
                self.cell_output = layers.Add( name=str(num_cell)+"-all_and_one" )( unchosen_outputs )

            else:
                self.cell_output = outputs[-1]

        else:
            # add the add to outputs[-1]
            self.cell_output = outputs[-1]




def main():

    return 1






if __name__ == "__main__":

    main()
