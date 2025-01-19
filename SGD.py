#this is a simple implementation of the stochastic gradient descent algorithm



#dataset is made global, we have only training data in dataset and we also seperated the dataset into features and labels
#features if going to be a 2D array of m x n where m is the number of training examples and n is the number of features
import numpy as np


# for testing purposes i am going to use a paramaters numpy array as such:
class test1:
    def __init__(self):
        self.parameters = np.array([[0.1,0.2,0.3]])
        self.n = 3
        self.features = np.array([[1200,1,1],
                    [1500,2,0],
                    [1700,5,0],
                    [1100,3,1]])
        
        self.training_examples = 4

        self.lables  = np.array([[100],
                            [200],
                            [300],
                            [400]])
        




def SDG(learning_rate, parameters, features, lables, training_examples):
    training_examples = features.shape[0]
    for parameter in parameters:
        for i in range(0,training_examples):
            #grabbing the array of featurs for the ith training example
            x_i = features[i,:]
            parameters_i = parameters #for simplicity i am assuming that parameters are a n x 1 array
            hypothesis_i = np.cross(x_i,parameters_i)
            parameter = parameter - learning_rate*((hypothesis_i - lables[i][0])[0][0])*x_i[i]

    return parameters

test = test1()

print(SDG(learning_rate = 0.01,parameters = test.parameters, features = test.features, lables = test.lables ,training_examples = test.training_examples))