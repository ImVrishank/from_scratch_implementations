#this is a simple implementation of linear regression from scratch

import numpy as np
from SGD import SGD


def LR(training_features, training_lables, learning_rate, query_features):
    # takes in training_features, training_lables, learning_rate, query_features
    # gives out prediction based off the query_features

    # learning_rate must be a float, 
    # training_features must be an (m x n) numpy array, 
    # training_lables must be an (m x 1) numpy array,
    # query_features must be (1 x n) numpy array
    # where n is the number of features and m is the number of training examples

    parameters = np.zeros((training_features.shape[1],1))

    query_features = query_features.reshape(1, -1)

    #getting updated predictions from SGD built earlier in form of (n x 1) numpy array
    parameters = SGD(learning_rate=learning_rate, parameters=parameters, features=training_features, lables=training_lables, loss_fn="MSE")

    prediction = np.dot(query_features, parameters)[0][0]
        
    return prediction




