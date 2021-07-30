import os
import sys
import json
import time




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

    
    num_block_conv=3
    num_block_reduc=1
    num_alt=1



    controller_instance = Controller(num_block_conv=num_block_conv, num_block_reduc=num_block_reduc, num_op_conv=7, num_op_reduc=4, num_alt=num_alt, scheme=1, path_train="./data_conv_training.mat", path_test="./data_conv_testing.mat")
    
    #controller = controller_instance.generateController()
    #cells_array, sum_proba = controller_instance.sample_arch(controller)
    #controller_instance.get_compiled_cnn_model(cells_array)
    
    #controller_instance.train(epochs=5, epochs_child=2)

    ####################################################################
    # sample and train 10 children over 20 epochs and plot the val_loss on same graph 'all_losses' (allows to choose a unique number of epochs)
    ####################################################################

    
    
    #controller_instance.match_train_test_metrics("val_accuracy", "kappa_score", 5)

    #controller_instance.train()

    
    
    
    print("num_replicas_in_sync : "+str(strategy.num_replicas_in_sync))
    print("per_worker_batch_size : "+str(per_worker_batch_size))
    
    
    #for i in range(10):
    #    controller_instance.simple_train(20, strategy, i, global_batch_size)

        
    # 2)
    #controller_instance.best_epoch(strategy, global_batch_size, nb_child_epochs=10, ite=25)
    
        
    # 3)
    #controller_instance.match_train_test_metrics("val_accuracy", "kappa_score", 2, strategy, global_batch_size)
    
    
    
    # 4)
    #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    #controller_instance.compute_accuracies(2, 2, strategy, options, "", [], global_batch_size, print_file=1)

    
    # 5) test for the number of iterations needed to obtain specific increase
    #controller_instance.test_nb_iterations(inc=0.1, mva1=10, mva2=500, limit=5000)
    
        
    # 6) train the RNN and save its weights in Controller_weights_.h5
    
    rew_func = rew_func_exp
    
    callbacks = []
    controller_instance.train(strategy, global_batch_size, callbacks, rew_func, data_options, nb_child_epochs=2, mva1=10, mva2=500, epochs=5000)
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":

    main()