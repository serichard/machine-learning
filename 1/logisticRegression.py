'''
Logistic regression sample code for CSCI5105 KBTU
Written by Jongtae Park, Feb. 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import math

eta = 0.7   # learning rate
epoch = 500 # iteration

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

# Logistic Regression Model
class LogisticRegression:
    
    def __init__(self, x, w, y):
        self.inputs  = x
        self.weights = w               
        self.target  = y
        self.output  = np.zeros(self.target.shape)

    def forward_proc(self):
       # forward processing of inputs and weights using sigmoid activation function 
        self.output = sigmoid(np.dot(self.weights, self.inputs.T))

    def backprop(self):
        # backward processing of appling the chain rule to find derivative of the mean square error function with respect to weights
        dw = (self.output - self.target) * self.inputs # same formular for both linear and logistic regression

        # update the weights with the derivative of the loss function
        self.weights -= eta * dw

    def predict(self, x):
        # predict the output for a given input x
        return (sigmoid(np.dot(self.weights, x.T)))
        
    def calculate_error(self):
        # calculate error
        error = -self.target * math.log(self.output) - (1-self.target) * math.log(1-self.output)
        return abs(error)

# Training 

if __name__ == "__main__":

    # data normalization on number of rooms and age of the house

    input_data = np.array(
                  [[.4, 1.0, 1.0],
                  [1.0, .3, 0.0]])
          
    
    '''
    target = [[1.0],  # fail
              [0.0]]  # pass
              
    '''

    weights = np.random.rand(1, 2)
    print("Initial Weights:", weights)

    training_loss = []
   
    # SGD Optimization
    for i in range(epoch):
   
        if i == 0: w = weights

 
        np.random.shuffle(input_data) # shuffle the input data
        X = input_data[:, 0:2]
        y = input_data[:, 2:3]
  
        # eta *= 0.95  # decreasing learning rate is found to be not good for this case
        loss_sum = 0

        for j in range(len(X)): 
       
            model = LogisticRegression(X[j], w, y[j])
            model.forward_proc()   # forward processing
            model.backprop()       # backward processing
            w = model.weights
            loss_sum += model.calculate_error()
            

        if (i % 10) == 0:
             training_loss.append(loss_sum / len(X)) 
             print("Loss: ", model.calculate_error())

   # show the training loss
    epoch_count = range(1, len(training_loss) + 1)
    plt.plot(epoch_count, training_loss, 'b-')
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch * 10')
    plt.ylabel('Loss')
    plt.show()
        
    #print("Output:", model.output)
    print("Adjusted Weights:", model.weights)

    # verify the output with the adjusted weights
    x1 = np.array([[0.4, 1.0]])
    print ("Output for the input data [.4, 1.0]:", model.predict(x1))
    x2 = np.array([[1.0, 0.3]])
    print ("Output for the input data [1.0, 0.3]:", model.predict(x2))
    
    # predicting and testing the output for a given input data
    x_prediction = np.array([[0.6, 0.4]])
    predicted_output = model.predict(x_prediction)
    print("Predicted data based on trained weights: ")
    print("Input (scaled): ", x_prediction)
    print("Output probability is : ", predicted_output)
    if predicted_output >= 0.5:
        print("Predicted output is PASS.")
    elif predicted_output < 0.5:
        print("Predicted output is Fail.")
                                      
    

  
