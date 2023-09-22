import numpy as np
import random
import torch

from QNN import QNN

class AgentGlouton():
    """Agent qui utilise la prédiction de son réseau de neurones pour choisir ses actions selon une stratégie d’exploration (pas d'apprentissage)."""

    def __init__(self, seed=0):
        """Constructeur.
        
        Params
        ======
            seed (int): random seed
        """
        self.seed = random.seed(seed)

    
        



