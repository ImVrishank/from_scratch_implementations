""""
implementing gaussian discriminant analysis from scratch

here is a quick revision for the bayes theorem:
https://github.com/ImVrishank/from_scratch_implementations/tree/main/images/GDA/bayes%20theorem


here is the derivation for the line of best fit for gaussian discriminant analysis:
https://github.com/ImVrishank/from_scratch_implementations/tree/main/images/GDA/derivation%20of%20line%20of%20best%20fit



"""


import numpy as np

def gaussian_discriminant_analysis(X, Y, query_features):
    x_0 = X[Y == 0]
    x_1 = X[Y == 1]

    mean_0 = ((1/x_0.shape[0]) * np.sum(x_0, axis=0)).T
    mean_1 = ((1/x_1.shape[0]) * np.sum(x_1, axis=0)).T

    cov = np.cov(X, rowvar=False)

    pi_0 = x_0.shape[0] / X.shape[0]
    pi_1 = x_1.shape[0] / X.shape[0]

    query_features = query_features.T

    #check the dervation to know how i got this equation, i am passing the equation through a sigmoid as it is a binary classification problem

    return (1 / (1 + np.exp(-np.dot(np.dot((mean_1 - mean_0).T, np.linalg.inv(cov)), query_features) - 0.5*(np.dot(mean_1.T, (np.dot(np.linalg.inv(cov), mean_1))) - np.dot(mean_0.T, (np.dot(np.linalg.inv(cov), mean_0)))) - np.log(pi_0 / pi_1))))

    


    

    




    


 