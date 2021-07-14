import os
import sys
import json
import time


if (sys.argv[1] == "multinodes"):

    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    os.environ.pop('TF_CONFIG', None)
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
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    
if (sys.argv[1] == "onenode"):
    strategy = tf.distribute.MirroredStrategy()

    
from Controller import *
from Utils import *
import numpy as np
import tensorflow as tf

def main():

    num_block_conv=2
    num_block_reduc=1

    num_alt=1



    controller = Controller(num_block_conv=num_block_conv, num_block_reduc=num_block_reduc, num_op_conv=7, num_op_reduc=2, num_alt=num_alt, scheme=2)

    #controller.train(epochs=5, epochs_child=2)

    # sample and train 10 children over 20 epochs and plot the val_loss on same graph 'all_losses' (allows to choose a unique number of epochs)
    #controller.best_epoch()

    #controller.match_train_test_metrics("val_accuracy", "kappa_score", 5)

    #controller.train()

    # sample and train 10 children on nb_child_epochs, plot the val_acc from train set (blue) 
    # and the acc_score on test set (red) on same graph 'match_test_train_metrics'
    # controller.match_train_test_metrics("val_accuracy", "accuracy_score", nb_child_epochs=5)
    #print(frequency_distr(3, [0.1, 0.2, 0.5, 0.7, 0.1, 0.3]))

    #controller.test_nb_iterations(mva=500)
    
    per_worker_batch_size = 64
    
    if (sys.argv[1] == "multinodes"):

        tf_config = json.loads(os.environ['TF_CONFIG'])
        num_workers = len(tf_config['cluster']['worker'])
        global_batch_size = per_worker_batch_size * num_workers

    
    if (sys.argv[1] == "onenode"):
        
        global_batch_size = per_worker_batch_size * strategy.num_replicas_in_sync
        
    
    print("num_replicas_in_sync : "+str(strategy.num_replicas_in_sync))
    
    controller.compute_accuracies(5, 5, strategy, global_batch_size, print_file=0)
    
    


if __name__ == "__main__":

    main()