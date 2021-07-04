from Controller import *


def main():

	num_block_conv=4
	num_block_reduc=1

	num_alt=1




	controller = Controller(num_block_conv=num_block_conv, num_block_reduc=num_block_reduc, num_op_conv=8, num_op_reduc=2, num_alt=num_alt, scheme=2)

	controller.train(epochs=5, epochs_child=2)

	#controller.train()



	# scheme 1: 1 op no-skip

	# scheme 2: 1 op skip

	# scheme 3: 2 op no-skip


if __name__ == "__main__":

    main()