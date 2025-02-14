""""
implementing gaussian discriminant analysis from scratch

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

    return (1 / (1 + np.exp(-np.dot(np.dot((mean_1 - mean_0).T, np.linalg.inv(cov)), query_features) - 0.5(np.dot(np.dot(mean_1, mean_1.T) - np.dot(mean_0, mean_0.T)), np.linalg.inv(cov)) - np.log(pi_0 / pi_1))))

    


    

    




    


 