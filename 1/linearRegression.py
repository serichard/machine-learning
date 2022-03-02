'''
Lnear regression sample code for CSCI5105 KBTU
Written by Jongtae Park, Feb. 2021
'''
import numpy as np
import random

eta = 0.5    # learning rate
epoch = 1000 # iteration

# Linear Regression Model
class LinearRegression:
    
    def __init__(self, x, w, y):
        self.inputs  = x
        self.weights = w               
        self.target  = y
        self.output  = np.zeros(self.target.shape)

    def forward_proc(self):
        # forward processing of inputs and weights
        self.output = np.dot(self.weights, self.inputs.T)

    def backprop(self):
        # backward processing of appling the chain rule to find derivative of the mean square error function with respect to weights
        dw = (self.output - self.target) * self.inputs

        # update the weights with the derivative of the loss function
        self.weights -= eta * dw

    def predict(self, x):
        # predict the output for a given input x
        return (np.dot(self.weights, x.T))
        
    def calculate_error(self):
        # calculate error
        error = self.target - self.output
        return abs(error)

# Training 

if __name__ == "__main__":

    # data normalization on number of rooms and age of the house
    input_data = np.array(
                  [[.3, .7, 50],
                   [.5, .5, 100]])
    weights = np.random.rand(1, 2)
    print("Initial Weights:", weights)

    # SGD Optimization
    for i in range(epoch):
   
        if i == 0: w = weights       

        np.random.shuffle(input_data) # shuffle the input data
        X = input_data[:, 0:2]
        y = input_data[:, 2:3]

        for j in range(len(input_data)):
         
            model = LinearRegression(X[j], w, y[j])
            model.forward_proc()   # forward processing
            model.backprop()       # backward processing
            w = model.weights 

        if (i % 50) == 0:
             print("Loss: ", model.calculate_error())
        
    print("Output:", model.output)
    print("Adjusted Weights:", model.weights)

    # Prediction
    new_data = np.array([[.4, .6]])
    print("Price for 4 beds and 6 years old is predictd as:", model.predict(new_data))
    

  
