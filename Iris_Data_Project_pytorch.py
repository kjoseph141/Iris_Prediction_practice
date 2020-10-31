# Import modules
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import seaborn as sns
from pandas.plotting import scatter_matrix

from iris_custom_model import build_network



def analyze_data(data):
    """
    Displays key metrics of data, distribution, and visualizations

    Parameters
    ----------
    data : array or DataFrame
        analyzes data and displays key metrics.

    Returns
    -------
    None.

    """
    mean_data = np.mean(data[:,:-1], axis=0)
    std_data = np.std(data[:,:-1], axis=0)
    
    print(f'Mean of data features: {mean_data}')
    print(f'Std of data features: {std_data}')

    # Look at and analyze data
    
    print(f'Shape: {data.shape}')
    print(f'Head: {data.head()}')
    print(f'Tail: {data.tail()}')
    print(f'Describe data: {data.describe()}')
    
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



def prepare_iris_data(data):
    """
    One-hot encodes target variable and converts X and Y to tensors

    Parameters
    ----------
    data : DataFrame
        DataFrame of Iris dataset to be used for model input.

    Returns
    -------
    trainset and testset as tensors.

    """

    # One-Hot Encode target variable y
    
    X = data.iloc[:, 0:4]
    y = data.iloc[:,-1]
    Y = pd.get_dummies(y)
    
    # Recombine X and Y
    
    data_w_dummies = pd.concat([X, Y], axis=1)
    data_w_dummies.head()
    
    
    # Split data into train and test sets and 
    # Make data iterable by batches with DataLoader
    
    train, test = train_test_split(data_w_dummies, test_size=0.2, shuffle=True)
    
    
    trainset = torch.Tensor(train.values)
    
    testset = torch.Tensor(test.values)
    
    return trainset, testset
        


def plot_loss_curve(num_epochs, losses):
    """
    Displays loss curve plot    
    
    Parameters
    ----------
    num_epochs : int
        number of epochs used to train and store losses.
    losses : list or array
        list or array of losses at each epoch.

    Returns
    -------
    None.

    """
    plt.xlabel('Epochs')
    plt.ylabel('Loss')  
    plt.title('Loss Curve')      
    plt.plot(range(num_epochs), losses)
    plt.show()
  


def test_model(model, trainset, testset):
    """
    Displays accuracy score and confusion matrix
    
    Parameters
    ----------
    model : pytorch model
        instantiated model object which has been trained.

    Returns
    -------
    None.

    """
    model.eval()
    
    predictions = []
    actuals = []
    
    for data in testset:
        # data will have batch of features and labels
        X = data[0:4]
        y = data[4:]
    
        pred = np.round(model(X).detach().numpy())
        actual = y.detach().numpy()
        # print(f'pred: {pred}')
        # print(f'actual: {actual}')
        predictions.append(pred)
        actuals.append(actual)
        
    print(accuracy_score(y_true=actuals, y_pred=predictions))
          
        
    # Confusion Matrix
    
    confusion_matrix = np.zeros((3, 3))
    for i,j in zip(predictions, actuals):
        confusion_matrix[i, j] += 1
    print("Confusion matrix:\n", confusion_matrix)




def save_model(path_name, model):
    """
    Saves state_dict of model

    Parameters
    ----------
    path_name : string
        path or filename for saved model state_dict.

    Returns
    -------
    None.

    """

    # Specify a path
    PATH = path_name
    
    # Save
    torch.save(model.state_dict(), PATH)
    


def train_model(model, optimizer, trainset, num_epochs):
    """
    Create and run training loop on model

    Parameters
    ----------
    model : pytorch model
        instantiated model object (trained or untrained).
    trainset : tensor
        training set of data.
    num_epochs : int
        number of epochs for training.

    Returns
    -------
    list of losses for each epoch.

    """

    EPOCHS = num_epochs
    losses = []
    
    for epoch in range(EPOCHS):
        
        for data in trainset:
            
            # data will have batch of features and labels
            X = data[0:4]
            y = data[4:]
    
            # pass input through network
            output = model(X)
    
            # zero out gradients for each batch
            optimizer.zero_grad()
            
            # compute loss and backpropagate
            loss = F.mse_loss(output, y)
            loss.backward()
            optimizer.step()
        
        print(f'Loss at epoch {epoch+1}: {loss}')
        losses.append(loss)
        
    return losses






# Build NN Model and Instantiate model object
net = build_network()
#print(net)

# Load Data
data = pd.read_csv('~/Python/Iris Data Project/IRIS.csv')

# Analyze data
analyze_data(data)

# Prepare data for model
trainset, testset = prepare_iris_data(data)


# Define optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-2)

# Train model
losses = train_model(net, optimizer, trainset, num_epochs=20)
    

# Plot loss curve
plot_loss_curve(losses=losses, num_epochs=20)

# Evaluate on testset
test_model(net, trainset, testset)

# Save model if necessary
save_model('state_dict_iris_model_n.pt', net)