"""
In this file, i implement the softmax regression algorithm from scratch. 

I can get you the result in 2 ways: 
--> one hot vector method: 
    for every training example, calculate logit vector {logit[k]}. 
    for q in range(k):
        logit[q] = [exp(dot(x_i, theta_q))] / [summation for p = class 1 to p = class k {exp(dot(x_i, theta_p))}]
    
    compare logit with one hot vector. this is done using cross entropy

    cross_entropy = -(summation for p = class 1 to p = class k {(one hot vector for that training example) * ( log(logit for that training example) )})
    
    this simplifies to: 
    cross_entropy(p, p_hat) = - log(p_hat)
        where: 
        p - one hot vector
        p_hat - logit

    use the cross_entropy as the loss function and apply gradient descent algo. 
    this will give you the paramters matrix (n cross k)
    use this to get your predictions.


--> poisson's PDF method: 
    where i calculate the hypothesis as: h_theta(x) = exp(dot(x_i, theta_k))
    will end with a solution that takes 3 nested loops, horrid time complexity [O[n^3]]



"""

import numpy as np

def softmax_regression_method_1(X, Y, learning_rate, query_features):
    """
    this is the implementaion of method 1:

    m is the number of training examples
    n is the number of features
    k is the number of classes

    X(m x n) is a 2D array containing features of dataset
    Y(m x k) is a 2D array containg one hot vectors as lables of dataset
    learning_rate is a scalar 
    query_feature(1 x k) is the features we want to predict lable for

    theta(n x k) is a 2D array containing the paramters
    Z(m x k) is a 2D array containing the dot product value of X and theta
    y_hat(1 x k) is a 1D array containing the prediction for ith example 
    y_given(1 x k) is the 1D array of the ith example

    predictions(1 x k) is the vector of probabilites each poiniting to a certain class
    
    """

    n = X.shape[1] # number of features
    m = X.shape[0] # number of training examples
    k = Y.shape[1] # number of classes

    theta = np.zeros((n, k)) 
    y_hat = np.zeros(k)


    for i in range(m):
        y_given = Y[i, :].flatten()
        
        Z = np.exp(np.dot(X, theta))

        y_hat = Z[i,:].flatten()
        y_hat = (y_hat) / (np.sum(y_hat))
        
        theta = theta - np.multiply(learning_rate, np.dot(X.T, (y_hat - y_given).reshape((m, k))))  

    # parameters are now optimized
    
    # now predicting the value as a vector of probabilities

    predictions = np.zeros(k)

    for k in range(k):
        predictions[k] = np.dot(theta[:, k], query_features).flatten()

    return predictions




def softmax_regression_method_2(X, Y, learning_rate, query_features):
    """

    this is the implementation of method 2:

    """


    n = X.shape[1] # number of features
    m = X.shape[0] # number of training examples

    k = len(np.unique(Y.reshape(-1))) # number of classes

    theta = np.zeros((n, k)) # paramters matrix, initially all zeros

    for k in range(k):
        
        theta_k = theta[k, :] # the paramters of the kth line

        for i in range(m):

            y_i = Y[i,1] # the lable of the ith training example
            x_i = X[i,:] # all features of the ith training example

            hypo_i = np.exp(np.dot(x_i, theta_k)) # hypothesis calc acc to poisson's PDF

            for j in range(n):
                x_i_j = x_i[1,j] # the jth value of the features matrix of the ith training example
                theta_k[j,1] += learning_rate*(y_i - hypo_i)*x_i_j # updating jth parameter of parameter matrix

        theta[k, :] = theta_k

    # paramters are now optimized.

    predictions = np.zeros(k)

    for k in range(k):
        predictions[k] = np.exp(np.dot(query_features, theta[k, :]))

    return predictions

        




        

    














