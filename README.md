# ENAS-Keras
Implementation of Efficient Neural Architecture Search with Keras, based on my Master thesis "Learning to learn: automatic design of neural network architectures
for biomedical signal processing tasks"


## Block.py
outputs a combination of the layers building a block

## Cell.py
outputs a stack of Blocks (from Block.py)

## Controller.py
generates the RNN controller for ENAS and provides all functions for training and testing the NAS

## Main.py
where all the code is ran from, all steps for prep and training of the NAS algo are shown
