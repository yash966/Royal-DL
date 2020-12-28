# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 21:35:40 2020

@author: yash
"""

import numpy as np
class Perceptron(object):
    
    def __init__(self, no_of_inputs, thresold=100, learning_rate=0.01):
        self.thresold=thresold
        self.learning_rate=learning_rate
        self.weights=np.zeros(no_of_inputs + 1)
        
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation
    
    def train(self, train_inputs, labels):
        for _ in range(self.thresold):
            print("number"+ str(_))
            print("weights--"+str(self.weights))
            for inputs, label in zip(train_inputs, labels):
             prediction = self.predict(inputs)
             self.weights[1:] += self.learning_rate * (label - prediction) * inputs
             self.weights[0] += self.learning_rate * (label-prediction)
        