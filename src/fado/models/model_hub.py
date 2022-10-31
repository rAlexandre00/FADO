from importlib import import_module
import os
from fedml.model.model_hub import *

def get_model(dataset: str, model: str = None):

    if model:
        if os.path.exists(model): # is file
            # load file -> user has to implemented the get_model() method!!!!!
            model = model.split('.py')[0]            
            model = getattr(import_module('.', model), 'get_model')()
        else: # is not a file, user is asking for a specific model
            if model == 'cnn':
                return CNN_DropOut(False)
            elif model == 'rnn_fed':
                return RNN_FedShakespeare()
            elif model == 'rnn_orig':
                return RNN_OriginalFedAvg()
            elif model == 'lr':
                return LogisticRegression(28 * 28, 62)
    else:
        if dataset == 'femnist':
            return CNN_DropOut(False)
        elif dataset in ['shakespeare', 'sent140']:
            return RNN_FedShakespeare()
        else:
            raise Exception(f"Dataset {dataset} not supported!")
    return model