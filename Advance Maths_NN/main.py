#! /usr/bin/env python
# ------------------------------------
import data
import model
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # get data
    train_seq = data.get_data(9000)
    val_seq = data.get_data(2000)
    # get batches 
    train_data = data.get_batch(train_seq)
    val_data = data.get_batch(val_seq)
    model.req_data(train_data) # feed batches to the model 
    model.bigram() # build and train bigram model
    model.f1(train_data, val_data) # Fit the transformer model1, learning rate of 0.001, 4 encoder layers and 4 decoder layers, 10 epoches
    model.f2(train_data, val_data) # Fit the transformer model2, learning rate of 0.01, 2 encoder layers and 2 decoder layers, 10 epoches
    model.f3(train_data, val_data) # Fit the transformer model3, learning rate of 0.1, 2 encoder layers and 4 decoder layers, 10 epoches