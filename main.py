import os
import sys
import json
import time

if(sys.argv[1] == "multinodes" and sys.argv[2] == 0):
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    os.environ.pop('TF_CONFIG', None)

if(sys.argv[1] == "multinodes" and sys.argv[2] == 1):
    time.sleep(10)


if (sys.argv[1] == "multinodes"):

    if '.' not in sys.path:
        sys.path.insert(0, '.')

    os.environ["TF_CONFIG"] = json.dumps({
        'cluster': {
            'worker': ["localhost:12345", "localhost:23456"]
        },
        'task': {'type': 'worker', 'index': sys.argv[2]}
    })

    
    



import tensorflow as tf

        


if (sys.argv[1] == "multinodes"):
    #os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    
if (sys.argv[1] == "onenode"):
    strategy = tf.distribute.MirroredStrategy()

    
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

    
    
    per_worker_batch_size = 256
    
    
    if (sys.argv[1] == "multinodes"):

        tf_config = json.loads(os.environ['TF_CONFIG'])
        num_workers = len(tf_config['cluster']['worker'])
        global_batch_size = per_worker_batch_size * num_workers

    
    if (sys.argv[1] == "onenode"):
        
        global_batch_size = per_worker_batch_size * strategy.num_replicas_in_sync
        
    
    print("num_replicas_in_sync : "+str(strategy.num_replicas_in_sync))
    print("per_worker_batch_size : "+str(per_worker_batch_size))
    
    
    #for i in range(10):
    #    controller_instance.simple_train(20, strategy, i, global_batch_size)

        
    # 2)
    #controller_instance.best_epoch(strategy, global_batch_size, nb_child_epochs=10, ite=25)
    
        
    # 3)
    #controller_instance.match_train_test_metrics("val_accuracy", "kappa_score", 2, strategy, global_batch_size)
    
    
    
    # 4)
    #controller_instance.compute_accuracies(100, 2, strategy, global_batch_size, print_file=1)
    

    # 5) test for the number of iterations needed to obtain specific increase
    #controller_instance.test_nb_iterations(inc=0.1, mva1=10, mva2=500, limit=3300)
    
    # 6) train the RNN and save its weights in Controller_weights_.h5

    train(strategy, epochs=5000, epochs_child=2)

    
if __name__ == "__main__":

    main()