# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_network(input_dim=4, hidden_1=16, output_dim=3):
    """
    Build network and define structure to be instantiated

    Parameters
    ----------
    input_dim : int, optional
        number of input nodes. The default is 4.
    hidden_1 : int, optional
        number of hidden layer 1 nodes. The default is 16.
    output_dim : int, optional
        number of output nodes. The default is 3.

    Returns
    -------
    Net object class.

    """

    input_dim, hidden_1, output_dim = input_dim, hidden_1, output_dim
    
    
    class Net(nn.Module):
        
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_1)
            self.fc2 = nn.Linear(hidden_1, output_dim)
            self.fc3 = nn.Softmax(dim=0)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
            return x
    
    return Net



def get_model(network):
    """
    

    Parameters
    ----------
    network : network object
        network object to be instantiated.

    Returns
    -------
    model : pytorch model
        DESCRIPTION.

    """
    # Specify a path
    PATH = "/Users/kevinjoseph/Python/Iris Data Project/state_dict_Iris_model_2.pt"

    # Load model
    model = network()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    return model



def get_tensor(user_inputs):
    """
    Converts user inputs to tensor for model inference
    
    Parameters
    ----------
    user_inputs : list or array
        measurements from user to be converted to tensor.

    Returns
    -------
    tensor.

    """
    user_inputs = [float(num) for num in user_inputs]
    tensor_user_inputs = torch.Tensor(user_inputs)
    
    return tensor_user_inputs

