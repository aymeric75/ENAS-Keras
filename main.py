from Controller import *


def main():

	num_block_conv=1
	num_block_reduc=1

	num_alt=1




	controller = Controller(num_block_conv=num_block_conv, num_block_reduc=num_block_reduc, num_op_conv=8, num_op_reduc=2, num_alt=num_alt, scheme=2)

	controller.train()









if __name__ == "__main__":

    main()