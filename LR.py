# a simple implementation of linear regression from scratch

import torch 

class LinearRegression():
    def __init__(self, X, Y, learning_rate=0.01):
        self.X = torch.cat((X, torch.ones(X.shape[0], 1)), dim=1)
        self.Y = Y
        self.m, self.n = self.X.shape
        self.w = torch.randn(self.n, 1)
        self.learning_rate = learning_rate
        
    def iterate(self):
        error = self.X @ self.w - self.Y # (w^T x + b)- y --> (m x n )@(n x 1) - (m x 1) = (m x 1)
        self.w = self.w - self.learning_rate * (1/self.m) * (self.X.T @ (error)) # theta_old = theta_new - alpha*(1/m)*(X^T*(error)) --> (n x 1) = (n x 1) - {(n x m)@(m x 1)}
        return self.w
    
    def calculate_loss(self):
        error = self.X @ self.w - self.Y
        return (1/(2*self.m)) * torch.sum(error**2) # MSE loss
    
X = torch.randn(100, 2)
Y = torch.randn(100, 1)
model = LinearRegression(X, Y)
parameters = model.w
prev_loss = model.calculate_loss()

while(model.calculate_loss() >= prev_loss):
    prev_loss = model.calculate_loss()
    parameters = model.iterate()

# predicting a value
query_features = torch.randn(1, 2)
query_labels = torch.randn(1,1)

query_features = torch.cat((query_features, torch.ones(1,1)), dim=1)

query_predictions = query_features @ parameters
print(query_predictions)


    
    
    
    
    
    
    
    