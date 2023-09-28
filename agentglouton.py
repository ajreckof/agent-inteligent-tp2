import numpy as np
import random
import torch

from QNN import QNN

class AgentGlouton():
    """Agent qui utilise la prédiction de son réseau de neurones pour choisir ses actions selon une stratégie d’exploration (pas d'apprentissage)."""

    def __init__(self, seed=0, eps=1):
        """Constructeur.
        
        Params
        ======
            seed (int): random seed
        """
        self.seed = seed
        random.seed(self.seed)
        self.QNN = QNN(self.seed)
        self.eps = eps

    def __getstate__(self):
        return {
            "eps": self.eps,
            "seed": self.seed,
            "state_dict": self.QNN.state_dict(),
        }

    def __setstate__(self, state):
        print(state)
        self.eps = state["eps"]
        self.seed = state["seed"]
        self.QNN = QNN(self.seed)
        self.QNN.load_state_dict(state["state_dict"])

    def getAction(self, state):
        if random.random() < self.eps:
            return random.choice(range(4))
        else:
            q_values = self.QNN(torch.from_numpy(state))
            return int(torch.argmax(q_values))

    
        



