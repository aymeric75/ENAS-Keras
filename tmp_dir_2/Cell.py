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



    def __init__(self, type_cell = 'conv', cell_inputs = [], cell_output='', scheme=1, num_filters=10):

        self.type_cell = type_cell
        self.cell_inputs = cell_inputs
        self.cell_output = cell_output
        self.scheme = scheme
        self.num_filters = num_filters



    def generateCell(self, blocks_array, type_cell, num_cell):


        #x = self.cell_input

        outputs = self.cell_inputs

        chosen_outputs = []


        if(self.type_cell=='conv'):




            if(self.scheme==1):
                step=1
                for i in range(0, len(blocks_array), step):
                    chosen_outputs.append(blocks_array[i])
                    block_name = 'C'+str(num_cell)+'-B'+str(int(i/step))
                    new_block = Block(block_name, input1=i, input2=i, op1=blocks_array[i], op2=blocks_array[i], inputs=outputs, cell_type=self.type_cell, num_cell=num_cell, num_filters=self.num_filters)
                    new_block.construct()
                    outputs.append(new_block.output)


            elif(self.scheme==2):
                step=2

                for i in range(0, 2):
                    chosen_outputs.append(blocks_array[i])
                    block_name = 'C'+str(num_cell)+'-B'+str(int(i))
                    new_block = Block(block_name, input1=int(i), input2=int(i), op1=blocks_array[i], op2=blocks_array[i], inputs=outputs, cell_type=self.type_cell, num_cell=num_cell, num_filters=self.num_filters)
                    new_block.construct()
                    outputs.append(new_block.output)
                for i in range(2, len(blocks_array), step):
                    chosen_outputs.append(blocks_array[i])
                    block_name = 'C'+str(num_cell)+'-B'+str(int((i+2)/step))
                    new_block = Block(block_name, input1=int(i/2), input2=blocks_array[i], op1=blocks_array[i+1], op2=blocks_array[i+1], inputs=outputs[1:], cell_type=self.type_cell, num_cell=num_cell, num_filters=self.num_filters)
                    new_block.construct()
                    outputs.append(new_block.output)

            elif(self.scheme==3):
                step=2
                for i in range(0, len(blocks_array), step):
                    chosen_outputs.append(blocks_array[i])
                    block_name = 'C'+str(num_cell)+'-B'+str(int(i/step))
                    new_block = Block(block_name, input1=int(i/2), input2=int(i/2), op1=blocks_array[i], op2=blocks_array[i+1], inputs=outputs, cell_type=self.type_cell, num_cell=num_cell, num_filters=self.num_filters)
                    new_block.construct()
                    outputs.append(new_block.output)

            else:
                step=3

                # Faut que ça renvoie que
                for i in range(0, 4, 2):
                    #chosen_outputs.append(blocks_array[i])
                    chosen_outputs.append(int(i/2))
                    block_name = 'C'+str(num_cell)+'-B'+str(int(i/2))
                    new_block = Block(block_name, input1=int(i/2), input2=int(i/2), op1=blocks_array[i], op2=blocks_array[i+1], inputs=outputs, cell_type=self.type_cell, num_cell=num_cell, num_filters=self.num_filters)
                    new_block.construct()
                    outputs.append(new_block.output)        
             

                for i in range(4, len(blocks_array), step):
                    chosen_outputs.append(int(i/4))
                    chosen_outputs.append(blocks_array[i])
                    block_name = 'C'+str(num_cell)+'-B'+str(int((i+1)/4) + 1)
                    new_block = Block(block_name, input1=int((i+1)/4) + 1, input2=blocks_array[i]+1, op1=blocks_array[i+1], op2=blocks_array[i+2], inputs=outputs, cell_type=self.type_cell, num_cell=num_cell, num_filters=self.num_filters)
                    new_block.construct()
                    outputs.append(new_block.output)





        else:

            for i in range(0, len(blocks_array)):


                block_name = 'C'+str(num_cell)+'-B'+str(i)

                new_block = Block(block_name, input1=i, input2=i, op1=blocks_array[i], 
                    op2=blocks_array[i], inputs=outputs, cell_type=self.type_cell, num_cell=num_cell, num_filters=self.num_filters)

                new_block.construct()

                #print(new_block.output)

                outputs.append(new_block.output)


        if(self.type_cell=='conv' and (self.scheme==4) ):

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
