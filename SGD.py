#this is a simple implementation of the stochastic gradient descent algorithm from scratch

import numpy as np

def SGD(learning_rate, parameters, features, lables, loss_fn): 
    # takes in learning_rate, parameters, features, lables and gives out updated paramters
    # learning rate must be a float, 
    # parameters must be an (n x 1) numpy array,
    # features must be an (m x n) numpy array, 
    # lables must be an (m x 1) numpy array
    # where n is the number of features and m is the number of training examples


    training_examples = features.shape[0]

    #iterating through the parameters
    for j in range(0,parameters.shape[0]):
        #grabbing the perticular parameter we are going to be altering in the for loop
        parameter = parameters[j][0]

        #iterating through the training examples
        for i in range(0,training_examples):
            #grabbing the array of features for the ith training example
            x_i = features[i,:].reshape(-1,features.shape[1])

            hypothesis_i = np.dot(x_i,parameters)[0][0]

            lable_i = lables[i][0]

            x_j = x_i[0][j]

            #updating the parameter
            parameter = parameter - learning_rate*differential_of_loss(loss_fn=loss_fn, hypothesis_i=hypothesis_i, lable_i=lable_i, x_j=x_j)
            #uploading the changed parameter to the parameter matrix
            parameters[j][0] = parameter

        
    return parameters


def differential_of_loss(loss_fn, hypothesis_i = None, lable_i = None, x_j = None, weight_i = None):
    # loss_fn is the type of loss function we are using, it is input as a string:
    # could be MSE (mean square error), 
    # LOWESS (used in locally weighted regression)


    if loss_fn == "MSE":
        return (hypothesis_i - lable_i)*x_j
    
    if loss_fn == "LOWESS":
        pass








        


