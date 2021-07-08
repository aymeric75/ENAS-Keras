from Controller import *
from Utils import *
import numpy as np
import tensorflow as tf

def main():


	print(np.__version__)



	num_block_conv=2
	num_block_reduc=1

	num_alt=1



	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
	  try:
	    # Currently, memory growth needs to be the same across GPUs
	    for gpu in gpus:
	      tf.config.experimental.set_memory_growth(gpu, True)
	    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
	    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	  except RuntimeError as e:
	    # Memory growth must be set before GPUs have been initialized
	    print(e)




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

	accuracies = [ 0.2, 0.8, 0.8, 0.8, 0.9, 0.7, 0.1 ]

	print("ICI")

	controller.test_nb_iterations()
	#controller.test_nb_iterations()

	# sample PUIS 



	# scheme 1: 1 op no-skip

	# scheme 2: 1 op skip

	# scheme 3: 2 op no-skip


if __name__ == "__main__":

    main()