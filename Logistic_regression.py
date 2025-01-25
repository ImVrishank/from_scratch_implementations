### in this file, i implement the concept of logistic regression for binary classification from scratch

### This is a way i visualise the algorithm, hence the excessive commenting. 


import numpy as np

def gradient_ascent(features, lables, learning_rate, parameters):
    ## takes in features, lables, learnig rate and paramaters and updates and returns parameters

    # learning rate must be a float
    # parameters must be an (n x 1) numpy array 
    # features must be an (m x n) numpy array, 
    # lables must be an (m x 1) numpy array
    # where n is the number of features and m is the number of training examples

    m = features.shape[0]
    n = features.shape[1]

    for j in range(n):
        for i in range(m):
            x_i = features[i, :]
            x_i_j = x_i[j]#redundancy: use features matrix and put it in the outer for loop
            y_i = lables[i][0]
            hypothesis_i = (1) / (1 + np.exp(-np.dot(x_i.reshape(1,-1), parameters).flatten())) 
                        
            summation = (y_i - hypothesis_i)*x_i_j
            
            parameters[j][0] += learning_rate*summation

        

    return parameters
        
