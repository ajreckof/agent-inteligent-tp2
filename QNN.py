import torch
import torch.nn as nn
import torch.nn.functional as F

class QNN(nn.Module):
    """Reseau de neurones pour approximer la Q fonction."""

    def __init__(self, seed=0):
        """Initialisation des parametres ...
        Params
        ======
            seed (int): Random seed
        """
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        "*** TODO ***"
        
    def forward(self, state):
        """Forward pass"""
        
        "*** TODO ***"
            
        return state


