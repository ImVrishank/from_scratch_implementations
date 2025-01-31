import numpy as np

"""
here i implement a few GLMs,
bernoulli's ---> for {1,0} lables
gaussian ---> for (-infinity, infinity) lables
poissons ---> for count value lables
"""



def GLM(X, Y, learning_rate, PDF, query_features):

    theta = np.zeros((X.shape[1], 1)) # the parameters of model, initially all zeroes.

    for i in range(X.shape[0]): # range m
        y_i = Y[i,1] # the lable of the ith training example
        x_i = X[i,:] # all features of the ith training example

        if(PDF == "bernoulli"): # expected prediction is either 0 or 1
            # hypothesis is: sigmoid(theta_transpose . X_i)
            hypo_i = (1) / (1 + np.exp(-(np.dot(x_i, theta).flatten())))

        elif(PDF == "gaussian"): # expected prediction is (-infinity , infinity)
            # hypothesis is: theta_transpose . X_i. 
            hypo_i = np.dot(x_i, theta).flatten()

        elif(PDF == "poissons"): # expected prediction is a count value. i.e [{1}, {2}, {3}.....] . will give a float, but the round it off
            # hypothesis is: exp(theta_transpose . X_i)
            hypo_i = np.exp(np.dot(x_i, theta).flatten())

    

        for j in range(X.shape[0]): # range n
            x_i_j = x_i[1,j] # the jth value of the features matrix of the ith training example
            theta[j,1] += learning_rate*(y_i - hypo_i)*x_i_j # updating jth parameter of parameter matrix

    # paramters are now calculated
    
    # output 
    return np.dot(query_features, theta).flatten()

    