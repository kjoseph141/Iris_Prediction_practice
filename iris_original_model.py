#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:48:32 2020

@author: kevinjoseph
"""
#Credit for inspiration and code: 
#https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
#by Jason Brownlee 


# Import modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split

# Select device to compute cpu/gpu
device = torch.device('cpu')


# Load dataset
data = pd.read_csv('IRIS.csv')


mean_data = np.mean(data[:,:4], axis=0)
std_data = np.std(data[:,:4], axis=0)


# Look at and analyze data

print(data.shape)
print(data.head())
print(data.describe())

# class distribution
print(data.groupby('class').size())

# box plots
data.plot(kind='box' , sharex = False , sharey = False, figsize=(15,10))
plt.show()

# histograms
data.hist(edgecolor = 'black', linewidth=1, figsize=(15,5))
plt.show()

# scatter plot matrix
scatter_matrix(data)
plt.show()

# seaborn pairplot: relationship between pairs of features
sns.pairplot(data, hue="class")
plt.show()


# Split-out validation dataset
dataset_values = data.values
X = data[:,0:4]
y = pd.get_dummies(data[:,4])   # one-hot encoded target variable

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# num_classes = len(set(dataset_values[:,4]))


# Build NN Model

batch_sz, D_in, H, D_out = 4, 4, 8, 3

# Use the nn package to define our model and loss function.
NN_model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
          torch.nn.Softmax(dim=0),
        )

loss_fn = torch.nn.MSELoss(reduction='mean')


learning_rate = 1e-4
optimizer = torch.optim.Adam(NN_model.parameters(), lr=learning_rate)


idx = np.arange(X_train.size()[0])

avg_loss_list = list()
epoch_list = list()

for epoch in range(20):
    
    total_loss = 0
    np.random.shuffle(idx)
    
    for id in idx:
        
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = NN_model(X_train[id,0:4])
        y = Y_train[id,4:]

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        total_loss += loss

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

    avg_loss = total_loss/X_train.size()[0]
    avg_loss_list.append(avg_loss)
    epoch_list.append(epoch)
    
# Plot loss
plt.plot(epoch_list, avg_loss_list, 'r-', lw=2)
plt.xlabel("epoch")
plt.ylabel("average error")
plt.grid(True)
plt.show()

#NN_model.save('Iris_NN_model.h5')


# Predict on the Test Data and Compute Evaluation Metrics

#pred_train= NN_model.predict(X_train)
#scores = NN_model.evaluate(X_train, Y_train, verbose=0)
#print(f'Accuracy on training data: {scores[1]*100}% \n Error on training data: {(1 - scores[1])*100}%')   
 
#pred_test= NN_model.predict(X_test)
#scores2 = NN_model.evaluate(X_test, Y_test, verbose=0)
#print(f'Accuracy on testing data: {scores2[1]*100}% \n Error on testing data: {(1 - scores2[1])*100}%') 



# Sample test

#species_list = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)

#sample = np.array([[5.1, 3.5, 1.4, 0.2]])
#sample_pred = (np.round(NN_model.predict(sample), decimals=0) != 0)[0]


# print(species_list[sample_pred])


#test_model = load_model('Iris_NN_model.h5')

#sepal_len = 5.1
#sepal_wid = 3.5
#petal_len = 1.3
#petal_wid = 0.2

#input_measures = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
#pred_on_input = (np.round(test_model.predict(input_measures), decimals=0) != 0)[0]

#print(test_model.predict(input_measures))
#print(np.round(test_model.predict(input_measures), decimals=0))
#print((np.round(test_model.predict(input_measures), decimals=0) != 0))
#print(pred_on_input)
#species_prediction = species_list[pred_on_input]
