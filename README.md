# ENAS-Keras
Implementation of Efficient Neural Architecture Search with Keras


## Block.py
outputs a combination of the layers building a block

## Cell.py
outputs a stack of Blocks (from Block.py)

## Controller.py
generates the RNN controller for ENAS and provides all functions for training and testing the NAS

## Main.py
where all the code is ran from, all steps for prep and training of the NAS algo are shown