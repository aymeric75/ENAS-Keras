import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.models import Model

from keras import initializers, regularizers
from keras import backend as K
import numpy as np



class Controller():

    def __init__(self, some_var='', num_middle_nodes=5, num_nodes=6, num_op=5):

        self.some_var = some_var

        self.num_nodes = num_nodes      # TOTAL number of nodes (so input and ouput nodes also)

        self.num_middle_nodes = num_middle_nodes      # 

        self.num_op = num_op

        self.num_units = 100



    def LSTM_softmax(self, inputs, num_classes, reshaped_inputs, initial):

        # (rx, initial_state=[y1_lstm_h2, y1_lstm_c])

        if initial:
            x = layers.LSTM(self.num_units, return_state=True)(inputs)

        else: # HERE inputs is reshaped
            #print(inputs)
            x = layers.LSTM(self.num_units, return_state=True)(reshaped_inputs, initial_state=inputs[1:])
        
        #
        rx = layers.Reshape((-1, self.num_units),)(x[0])

        y = layers.Dense(num_classes, activation="softmax")(x[0])

        return x, rx, y # r_y = reshaped 


    def generateController(self):

        y_soft, r_y = None, None

        controller_input = keras.Input(shape=(1,1,))


        outputs = []


        # we choose only for "middle" layers here (without input and output nodes)
        for i in range(0, self.num_middle_nodes):

            # Pour chaque noeud, construction d'un stack de sotfmax (<=> block)

            for o in ["inputL", "inputR", "operL", "operR"]:

                if i == 0 and o == "inputL":
                    _x, _rx, _initial = controller_input, None, True
                else:
                    _x, _rx, _initial = x, rx, False # output of previous LSTM_softmax


                if o == "inputL" or o == "inputR" :
                    # 1 softmax de taille 2 pour inputL et inputR
                    _num_classes = i+2
                else:
                    # 1 softmax de taille #num_op
                    _num_classes = self.num_op


                #print("i = "+str(i)+" v = "+str(v))

                x, rx, y = self.LSTM_softmax(inputs=_x, num_classes=_num_classes, reshaped_inputs=_rx, initial=_initial)


                outputs.append(y)



        model = Model(inputs=controller_input, outputs=outputs)


        return model



def main():






    # num_op = 5  <=====> classes = ["Conv2D_2_10", "Conv2D_1_5", "Conv2D_4_1", "MaxPooling2D_2", "MaxPooling2D_1_6"]

    controllerObj = Controller(num_middle_nodes=5, num_op=5)

    #controller = controllerObj.generateController()

    model = controllerObj.generateController()


    exit()

    test=None
    inp = None
    i=0
    for layer in model.layers:
        if(layer.__class__.__name__ == 'Dense'):
            
            if(i==0):
                test = np.random.random((1,1,))[np.newaxis,...]
            else:
                test = [[[res]]]

            test = np.random.random((1,1,))[np.newaxis,...]
            #print("test")
            #print(test)
            inp = model.input
            out = layer.output
            res = K.function([inp], [out])([test])

            print(res)
            # print(res)
            # print(np.argmax(res))
            # print(inp.shape)
            i+=1

            if(i>4):
                break
            #exit()

    exit()

    functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions

    # Testing
    test = np.random.random((1,1,))[np.newaxis,...]
    layer_outs = [func([test]) for func in functors]
    print(layer_outs)



    exit()


    utils.plot_model(controller)



    steps = controller.predict([3])

    #print(steps)

    for step in steps:
        #print(step)
        print(np.argmax(step))













if __name__ == "__main__":

    main()