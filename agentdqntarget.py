import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class AgentDQNTarget():
    """Agent qui utilise l'algorithme DQN."""

    def __init__(self, seed=0):
        """Constructeur.
        
        Params
        ======
            seed (int): random seed
        """
        self.seed = random.seed(seed)

      