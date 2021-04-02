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



    def generateCell(self, blocks_array, outputs):


        x = self.cell_input


        for i in range(0, len(blocks_array), 4):


            print("i = "+str(i))

            new_block = Block('block'+str(i), input1=blocks_array[i], input2=blocks_array[i+1], op1=blocks_array[i+2], op2=blocks_array[i+3], inputs=outputs)

            new_block.construct()

            print(new_block.output)

            outputs.append(new_block.output)

        self.cell_output = outputs[-1]

        #return outputs[-1]


# 1) définir ce que je peux mettre !!!!

# 2) ce que je peux mettre en faire un tableau puis tester la construction de ce type de Cell



def main():

    outputs = []

    # REPRENDRE ICI !!!!!!
    input_shape = (8, 900, 1)
    inputs = keras.Input(shape=(8, 900))
    x = layers.Reshape((8,900,1))(inputs)
    
    x = layers.BatchNormalization()(x)
    outputs.append(x)

    convCell = Cell('conv', cell_input=x)


    blocks_array = np.array([0,0,3,4,0,1,2,3])


    convCell.generateCell(blocks_array, outputs)


    print(convCell.cell_output)

    outputss = layers.Dense(2, activation='softmax')(convCell.cell_output)
    model = keras.Model(inputs=inputs, outputs=outputss, name="ansari_model")
    
    utils.plot_model(model)

    exit()




    input_shape = (8, 900, 1)

    inputs = keras.Input(shape=(8, 900))

    x = layers.Reshape((8,900,1))(inputs)

    x1 = layers.BatchNormalization()(x)


    ####   ConvCell



    #x = layers.Conv2D( 10, (2, 10), padding="same", activation='relu', input_shape=input_shape[:-1] ) (x)
    #x = layers.Conv2D( 10, (2, 10), padding="same", activation='relu')(x)

    x2 = layers.Conv2D( 10, (1, 900), padding="same", activation='relu')(x1)

    
    x3 = layers.Conv2D( 10, (8, 1), padding="same", activation='relu')(x1)

    x4 = layers.Add()([x2, x3])

    exit()

    x3 = layers.MaxPooling2D(pool_size=(2, 2))(x1)

    x = layers.Add()([x2,x3])

    x = layers.Conv2D(20, (1, 5), padding="same", activation='relu')(x)

    x = layers.Conv2D(20, (4, 1), padding="valid", activation='relu')(x)

    x = layers.MaxPooling2D(pool_size=(1, 6))(x)








    outputs = []

    input_shape = (8, 900, 1)

    inputs = keras.Input(shape=(8, 900))

    x = layers.Reshape((8,900,1))(inputs)

    outputs.append(x)

    x = layers.BatchNormalization()(x)

    outputs.append(x)



    #controllerObj = Controller(num_middle_nodes=5, num_op=5)
    #controller = controllerObj.generateController()
    #steps = controller.predict([3])

    # PAS DE VAR POUR COMBINAISON !
    blocks_array = np.array([1,1,0,0,2,2,3,3,3,3,1,1,4,4,2,2,5,5,4,4])


    for i in range(0, len(blocks_array), 4):



        print("i = "+str(i))


        all_layers = [
            layers.Conv2D( 10, (2, 10), padding="same", activation='relu'),
            layers.Conv2D(20, (1, 5), padding="same", activation='relu'), 
            layers.Conv2D(20, (4, 1), padding="valid", activation='relu'), 
            layers.MaxPooling2D(pool_size=(2, 2)), 
            layers.MaxPooling2D(pool_size=(1, 6)) 
        ]


        new_block = Block('block'+str(i), input1=blocks_array[i], input2=blocks_array[i+1], op1=blocks_array[i+2], op2=blocks_array[i+3], inputs=outputs)

        new_block.construct()

        print(new_block.output)

        outputs.append(new_block.output)





if __name__ == "__main__":

    main()
