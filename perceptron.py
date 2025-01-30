import numpy as np

### in this file, i implement the perceptron model from scratch. 
### the hypothesis is the main thing that changes in the perceptron model. Instead of passing x_i.theta_transpose to the sigmoid
### as we did in logistic regression, we pass it to the perceptron. 
### perceptron(x) returns 1 if x >= 0 and 0 if x < 0
### the equation to update the parameters is: theta_j = theta_j + learning_rate[summation from i = 0 to i = M({y_i - hypothesis(x_i)} * x_i_j )]

def perceptron(X, Y, learning_rate):
    
    theta = np.zeros((X.shape[1], 1)) # the parameters of model, initially all zeroes.

    for i in range(X.shape[0]): # range m
        y_i = Y[i,1] # the lable of the ith training example
        x_i = X[i,:] # all features of the ith training example

        hypo_i = 1 if (np.dot(x_i, theta.flatten() >= 0)) else 0 # the perceptron eqn as hypothesis. 

        for j in range(X.shape[0]): # range n
            x_i_j = x_i[1,j] # the jth value of the features matrix of the ith training example
            theta[j,1] += learning_rate*(y_i - hypo_i)*x_i_j # updating jth parameter of parameter matrix

    return theta

            