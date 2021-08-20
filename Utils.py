from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
import numpy as np
import math
from statistics import mean

class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):

        self.add_loss(inputs)
        #self.add_loss(1e-2 * tf.reduce_sum(inputs))
        # proba ET 

        return inputs
    

def moving_average(x, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return mean(a)



# for Kappa
def mvaverage(vals, window):
    tmp = np.copy(vals)
    for i, val in enumerate(vals):
        if(i >= window):
            # calcul de la moyenne
            average = np.average(vals[i-window: i])
            if(average < 0.5):
                tmp[i] = 0
            else:
                tmp[i] = 1
    return tmp


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def kappa(y_true, y_pred):
    
    #y_pred_av = mvaverage(y_pred, 6)
    #y_true_av = mvaverage(y_true, 6)
    
    
    return metrics.cohen_kappa_score(y_pred, y_true)


def truncate_float(number, length):
    """Truncate float numbers, up to the number specified
    in length that must be an integer"""

    number = number * pow(10, length)
    number = int(number)
    number = float(number)
    number /= pow(10, length)
    return number


def duplicate_in_array(array, decimals):
    decimal.getcontext().rounding = decimal.ROUND_DOWN
    for i in range(len(array)):
        if ( i < (len(array)-1) ):
            if(truncate_float(array[i], decimals) == truncate_float(array[i+1], decimals)):
                return True
    return False


# return a dict with keys = mean of considered quantile
# and values = frequency of the considered quantile

def frequency_distr( nb_quantiles, datas ):
    
    gap = 1. / nb_quantiles
    
    dico = {}
    
    for i in range(nb_quantiles):
        
        
        #dico[ sum_quantile / count ] = 0.
        
        sum_quantile = 0
        count = 0 # count of elements belonging to the quantile


        for ele in datas:
                        
            if( ele >= gap*i and ele < gap*(i+1) ):
                
                sum_quantile += ele
                
                count+=1
                
        
        if(count>0):
            dico[ sum_quantile / count ] = count / len(datas)
        
        
        
    return dico



def compute_credits(nodes, hours, nber_gpus):

    return 0.000278*nodes*(hours*3600)*(nber_gpus*5)



def rew_func_nothing(x):
    
    return x

def rew_func_exp(x):
    add=0
    minus=40
    
    #return np.exp(add+x)**5 - minus
    return (np.exp(x)**20) / 800000
    


# inverse of to_categorical
def out_categorical(y):
    return [np.argmax(x) for x in y]

# take [1, 1, 0, 1, 0 ..] as input
def count_pos_class(y):
    ones = sum(y)
    total = len(y)

    print(str(ones)+" pos in total, i.e, "+str((ones/total)*100)+" %")

    
def class_weights(y):
    ones = sum(y)
    total = len(y)

    perce = ones/total

    return {0: 1, 1: 1/perce}

def max_accuracies(file):

    # 'accuracies.txt'
    f = open(file, 'r')
    lines = f.readlines()
    accuracies = []
    for line in lines:
        accuracies.append(float(line))
    f.close()


    print(max(accuracies))









