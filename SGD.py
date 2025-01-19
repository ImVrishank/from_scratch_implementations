#this is a simple implementation of the stochastic gradient descent algorithm from scratch

import numpy as np


# for testing purposes i am going to use a paramaters numpy array as such:
class test1:
    def __init__(self):
        self.parameters = np.array([[0.0],
                                    [0.0],
                                    [0.0],
                                    [0.0]])
        self.n = 4
        self.features = np.array([
                    [1200,1,1,2],
                    [1500,2,0,2],
                    [1700,5,0,4],
                    [1100,3,1,6],
                    [1300,3,0,2]])
        
        self.training_examples = 5

        self.lables  = np.array([[100],
                            [200],
                            [300],
                            [400],
                            [300]])
        




def SGD(learning_rate, parameters, features, lables): 

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
            parameter = parameter - learning_rate*(hypothesis_i - lable_i)*x_j

            #uploading the changed parameter to the parameter matrix
            
            parameters[j][0] = parameter

        
    return parameters





            



        


test = test1()

print(SGD(learning_rate = 0.01,parameters = test.parameters, features = test.features, lables = test.lables))