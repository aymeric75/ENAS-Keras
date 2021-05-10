#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from keras import initializers, regularizers
from keras import backend as K
import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



"""
    sampling returns a one_hot tensor of the class randomly sampled from the 'x' softmax tensor
"""
# def sampling(x, depth):
#     zeros = x*0. ### useless but important to produce gradient
#     samples = tf.random.categorical(tf.math.log(x), 1)
#     samples = tf.squeeze(tf.one_hot(samples, depth=depth), axis=1)
#     return zeros+samples

# new 'sampling' function, only return an integer (to feed the Embedding layer)
def sampling(x, depth):
    x = np.array([[0.1, 0.9]])
    zeros = x*0. ### useless but important to produce gradient
    samples = tf.random.categorical(tf.math.log(x), 1)
    return samples



"""
    Custom Loss, to be built !
"""
def my_loss_fn(y_true, y_pred):
    # here, we don't use y_true or y_preds
    # P(At | At-1) = softmax * result_of_sampling
    #                 [ 0.32 0.1 0.58 ] * [ 0 1 0 ]
    #                     class "1" was chosen, with a proba of 0.1
    return 1

 

def main():


    num_block_conv=3
    num_block_reduc=2
    

    conv_indices = []

    reduc_indices = []


    for i in range(0, 100):

        to_print = ""

        # test all potential values
        for j in range(0, 4*num_block_conv):

            res = (i-j)/12

            if ( res == int(res) ):
                #conv_indices.append(i)
                print(str(i)+"-conv")
                to_print="yes"

        if to_print!="yes":
            #reduc_indices.append(i)
            print(str(i)+"-reduc")



    #print(conv_indices)

    #print(reduc_indices)

        # if ( res2 == int(res2) ):
        #     print(i)


        #   
        #
        #   0,1,2,3  4,5,6,7            CONV
        #
        #
        #   8,9,10,11   12,13,14,15     REDUC
        #
        #
        #   16,17,18,19  20,21,22,23    CONV
        #
        #
        #   24,25,26,27  28,29,30,31    REDUC



    exit()


    X = np.random.uniform(0,1, (100,1,1))
    y = tf.keras.utils.to_categorical(np.random.randint(0,3, (100,)))
    # y = y.reshape(-1,1,3) FOR return_sequences=True

    inp = keras.Input(shape=(1,1,))

    x, _, _ = layers.LSTM(100, return_sequences=False, return_state=True)(inp)
    x = layers.Dense(2, activation="softmax", name="dense1")(x)
    x = layers.Lambda(lambda t: sampling(t, depth=2), name="lambda")(x)
    x = layers.Embedding(2, 100)(x)
    # x = layers.Reshape((-1,2),)(x) (Only used when no Embedding Layer)
    x, _, _ = layers.LSTM(100, return_sequences=False, return_state=True)(x)


    exit()

    #x = layers.Dense(2, activation="softmax", name="dense2")(x)

    model = Model(inp, x)
    model.compile('adam', 'categorical_crossentropy')
    model.fit(X,y, epochs=3)


    utils.plot_model(model, to_file="example.png")






if __name__ == "__main__":

    main()
