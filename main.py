import os
import sys
import json
import time
from sklearn.utils import class_weight

import tensorflow as tf

per_worker_batch_size = 256
    
if (sys.argv[1] == "onenode"):
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = per_worker_batch_size * strategy.num_replicas_in_sync
    data_options = tf.data.Options()
    data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

else:

    hosts = sys.argv[1].split(',')
    task_id = int(sys.argv[2])

    num_workers = len(hosts)

    tab = []
    for host in hosts:
        tab.append(str(host)+":2222")

    cluster = tf.train.ClusterSpec({'worker': tab})

    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(cluster,
                                             task_type='worker', task_id=task_id,
                                             num_accelerators={'GPU': 1})


    data_options = tf.data.Options()
    data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    options = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
    strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver, communication_options=options)       
    
    global_batch_size = per_worker_batch_size * num_workers

    
from Controller import *
from Utils import *
import numpy as np

def main():

    num_block_conv=4
    num_block_reduc=1
    num_alt=1

    controller_instance = Controller(num_block_conv=num_block_conv, num_block_reduc=num_block_reduc, num_op_conv=7, num_op_reduc=4, num_alt=num_alt, scheme=3, path_train="./data_conv_training.mat", path_test="./data_conv_testing.mat")
        
    
    #####################
    # Preparation steps #
    #####################
    
    class_w = {0:1 , 1:4} # is actually redefined in train_child function (this param should be removed now)
    
    # 1) best_epochs, plot 'ite' times the val_loss and val_accuracy over 'nb_child_epochs' epochs
    controller_instance.best_epoch(strategy, global_batch_size, data_options, class_w, nb_child_epochs=10, ite=15)
    

    # 2) test if eval and final metrics match (on test set)
    nbre_epochs_child = 4
    controller_instance.match_train_test_metrics("val_accuracy", "kappa_score", nbre_epochs_child, strategy, global_batch_size, data_options, class_w)
        
        
    # 3) compute and write accuracies in a file
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    nbre_epochs_child = 6
    controller_instance.compute_accuracies(350, nbre_epochs_child, strategy, data_options, "", [], global_batch_size, print_file=1)
    
    
    # 4) test for the number of iterations needed to obtain specific increase
    controller_instance.test_nb_iterations(file_accs = 'accuracies.txt', inc=0.1, mva1=10, mva2=500, limit=5000)
    
    
    #####################
    #       Training    #
    #####################
    
    # 5) train the RNN and save its weights in Controller_weights_.h5
    #callbacks = []
    #controller_instance.train(strategy, global_batch_size, callbacks, rew_func_exp_2, data_options, nb_child_epochs=4, mva1=50, mva2=500, epochs=1200)
    
    # nb_child_epochs=3, mva1=10, mva2=100, epochs=500
    
    
  
    ###############################
    #       Testing the results   #
    ###############################
    
    # Tests
    #controller_instance.sampling_and_training_one_arch(global_batch_size, data_options, "Controller_weights.h5")
    
    # Display distribution of layers/blocks (Monte Carlo sampling)
    controller_instance.children_stats(by_block=0)
    
    
# is actually defined directly in the functions,
# should be defined outside, like here
def rew_func_exp_2(x):
    A = 100000
    B = 83000
    acc_rew = ( 1 / (1 + np.exp(-x*A+B)) ) * 10
    return acc_rew
    
    
if __name__ == "__main__":

    main()