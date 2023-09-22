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
        self.seed = seed
        random.seed(self.seed)
        self.QNN = QNN(self.seed)
        self.eps = 0.8

    def getAction(self, state):
        if random.random() < self.eps:
            q_values = self.QNN(torch.from_numpy(state))
            return int(torch.argmax(q_values))
        else:
            return random.choice(range(4))

    
        



