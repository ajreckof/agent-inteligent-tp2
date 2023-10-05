import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    """Buffer de taille fixe pour mémoriser les tuples d'expériences rencontrés."""

    def __init__(self,  buffer_size: int, batch_size: int):
        """Constructeur.

        Params
        ======
            buffer_size (int): taille max du buffer
            batch_size (int): taille d'un batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "next_state", "reward", "done"])

    def add(self, state, action, next_state, reward, done):
        """Ajout d'une experience au buffer."""
        experience = self.experience(state, action, next_state, reward, done)
        self.memory.append(experience)

    def sample(self):
        """Recuperation d'un minibatch de données aléatoires dans le buffer."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.tensor(
            np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None]).astype(np.uint8))
        next_states = torch.tensor(np.vstack(
            [e.next_state for e in experiences if e is not None])).float()
        rewards = torch.tensor(
            [e.reward for e in experiences if e is not None]).float()
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, next_states, rewards, dones)

    def __len__(self):
        """Taille courante du buffer."""
        return len(self.memory)
