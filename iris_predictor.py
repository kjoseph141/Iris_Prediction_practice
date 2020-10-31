import torch

import numpy as np

from iris_custom_model import build_network, get_model, get_tensor




def get_flower_prediction(user_inputs):
    """
    Take user input and generate the model's prediction
    
    Parameters
    ----------
    user_inputs : list or array
        user input of measurements to be converted to tensor and used as model input.

    Returns
    -------
    string
        prediction of flower species.

    """
    with torch.no_grad():

            
        species_list = np.array(['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'], dtype=object)
        
        # Build network object and instantiate model
        network = build_network()
        model = get_model(network)
        
        # Convert user inputs to tensor
        tensor_user_inputs = get_tensor(user_inputs)
        
        # Use model to generate prediction on user inputs
        pred = np.round(model.forward(tensor_user_inputs).detach().numpy())
        pred = np.array(pred, dtype=bool)
        
        # Select actual species name by indexing into names with prediction (another np.array)
        species_prediction = species_list[pred][0]

    return species_prediction