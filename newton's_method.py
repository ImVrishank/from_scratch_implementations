import numpy as np

### here, i implement the newtons method to update parameters as an alternative to gradient descent


def NM(X,Y, learning_rate):
    theta = np.zeros((-1, 1))
    gradient = gradient(X, Y, theta)
    hessian_inv = np.linalg.inv(hessian(X, Y, theta))

    theta = np.sum(theta, np.multiply(learning_rate, np.dot(hessian_inv, gradient)))

    return theta

    

def gradient(X, Y, theta):
    m = X.shape[0]
    theta = theta.reshape((1,-1)) # of shape (1 x n)



    for i in range(m):
        x_i = X[i,:] # of shape (n x 1)

        gradient = (np.multiply(x_i, Y[i,0] + np.dot(theta, x_i).flatten())).reshape(-1,1)

    return gradient


def hessian(X, Y, theta):

    combinations = np.array([(p,q) for p in range(X.shape[1]) for q in range(X.shape[1])]).flatten()


    hessian = np.empty((X.shape[1], X.shape[1]), dtype=object)
    hessian.fill(None)

    theta = theta.reshape[-1,1]

    for p,q in combinations:
        
        if hessian[p,q] is None:
            sum = 0
            for i in range(X.shape[0]):
                x_i = X[i,:]
                sum += (np.exp(-(np.dot(theta, x_i).flatten()))) / (1 + np.exp(-(np.dot(theta, x_i).flatten())))

            hessian[p,q] = theta[p,0] * theta[q,0] * sum

            

        if hessian[q,p] is None:
            hessian[q,p] = hessian[p,q]

    return hessian
        



    


    

