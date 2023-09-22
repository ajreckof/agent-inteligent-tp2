import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class AgentDQN():
    """Agent qui utilise l'algorithme DQN."""

    def __init__(self, env, seed=0):
        """Constructeur.
        
        Params
        ======
            seed (int): random seed
        """
        self.env = env
        self.seed = random.seed(seed)
        self.observation
        self.terminated
        self.truncated
        self.replay_buffer = ReplayBuffer(100_000, 64, seed)
        self.interaction_before_learn_phase = 4
        self.QNN = QNN(seed)

    def runOneEpisode(self):
        self.observation, _ = self.env.reset()
        self.terminated = False
        self.truncated = False 

        while not( self.truncated or self.terminated) :
            self.runOneInteraction(self.observation)
            self.interaction_before_learn_phase -= 1
            if self.interaction_before_learn_phase == 0 :
                self.interaction_before_learn_phase = 4
                self.runLearnPhase()

    def runLearnPhase(self):
        states, actions, next_states, rewards, dones = ReplayBuffer.sample()

        y = self.QNN(torch.from_numpy(state))
        t = rewards +

        pass

    def runOneInteraction(self, state):
        action = self.getAction()  # agent policy that uses the observation and info
        self.observation, reward, self.terminated, self.truncated, _ = self.env.step(action)
        self.replay_buffer.add(state, action, self.observation, reward, self.terminated or self.truncated)


    def getAction(self, state):
        if random.random() < self.eps:
            q_values = self.QNN(torch.from_numpy(state))
            return int(torch.argmax(q_values))
        else:
            return random.choice(range(4))

        

        