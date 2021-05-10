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



    def generateCell(self, blocks_array, type_cell):


        #x = self.cell_input

        outputs = self.cell_inputs


        if(self.type_cell=='conv'):


            for i in range(0, len(blocks_array), 4):


                #print("i = "+str(i))

                new_block = Block('block'+str(i), input1=blocks_array[i], input2=blocks_array[i+1], op1=blocks_array[i+2], 
                    op2=blocks_array[i+3], inputs=outputs, cell_type=self.type_cell)

                new_block.construct()

                print(new_block.output)

                outputs.append(new_block.output)

        else:

            for i in range(0, len(blocks_array)):


                new_block = Block('block'+str(i), input1=i, input2=i, op1=blocks_array[i], 
                    op2=blocks_array[i], inputs=outputs, cell_type=self.type_cell)

                new_block.construct()

                #print(new_block.output)

                outputs.append(new_block.output)



        self.cell_output = outputs[-1]

        #return outputs[-1]


# 1) définir ce que je peux mettre !!!!

# 2) ce que je peux mettre en faire un tableau puis tester la construction de ce type de Cell



def main():

    return 1



    # # Init
    # outputs = []
    # input_shape = (8, 900, 1)
    # inputs = keras.Input(shape=(8, 900))
    # x = layers.Reshape((8,900,1))(inputs)
    # outputs.append(x)
    # x = layers.BatchNormalization()(x)
    # outputs.append(x)




    # datas = [[1, 0, 4, 3, 2, 1, 1, 2, 1, 2, 0, 3], [0, 0, 1, 0, 0, 0, 0, 1], [0, 4, 3, 0, 2, 3, 1, 3, 3, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 0]]



    # # Conv Cell, cell_inputs must ALWAYS be an array with the possible DIRECT inputs for the cell
    # convCell1 = Cell('conv', cell_inputs=outputs)
    # blocks_array = np.array( [1, 0, 4, 3, 2, 1, 1, 2, 1, 2, 0, 3] )
    # convCell1.generateCell(blocks_array)
    # x = convCell1.cell_output


    # outputs = [x]
    # reducCell1 = Cell('reduc', cell_inputs=outputs)
    # blocks_array = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    # reducCell1.generateCell(blocks_array)
    # x = reducCell1.cell_output


    # outputs = [x]
    # convCell2 = Cell('conv', cell_inputs=outputs)
    # blocks_array = np.array([0, 0, 3, 0, 1, 0, 1, 3, 2, 2, 1, 0])
    # convCell2.generateCell(blocks_array)
    # x = convCell2.cell_output

    # #
    # #   ouais....
    # #
    # #
    # #           Sofmax, qui choisit parmi UNE classe ET une seule pour l'input
    # #
    # #               ET pour les opérations, parmi 2 classes 
    # #
    # #                       MAIS pas de sofmax
    # #
    # #




    # outputss = layers.Dense(2, activation='softmax')(convCell2.cell_output)
    # model = keras.Model(inputs=inputs, outputs=outputss, name="ansari_model")
    
    # utils.plot_model(model)


    # exit()


    # outputs = [x]
    # ## Reduc Cell
    # reducCell1 = Cell('reduc', cell_inputs=outputs)
    # blocks_array = np.array([0,0,1,1])
    # reducCell1.generateCell(blocks_array)
    # x = reducCell1.cell_output

    # outputs = [x]
    # # Conv Cell
    # convCell2 = Cell('conv', cell_inputs=outputs)
    # blocks_array = np.array([0,0,3,4,0,1,2,3])
    # convCell2.generateCell(blocks_array)
    # x = convCell2.cell_output



    # outputss = layers.Dense(2, activation='softmax')(convCell2.cell_output)
    # model = keras.Model(inputs=inputs, outputs=outputss, name="ansari_model")
    
    # utils.plot_model(model)

    # exit()






if __name__ == "__main__":

    main()
