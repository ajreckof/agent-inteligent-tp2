import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from agentdqn import AgentDQN


class AgentDQNTarget(AgentDQN):
	"""Agent qui utilise l'algorithme DQN."""

	def __init__(self, env, eps=1, gamma=0.9, learning_rate=0.05, seed=0, tau=0.001):
		"""Constructeur.
		
		Params
		======
			seed (int): random seed
		"""
		super().__init__(env=env, eps=eps, gamma=gamma,
                   learning_rate=learning_rate, seed=seed)
		self.tau = tau

		self.interaction_before_copy_target = 10_000

		self.QNNt = QNN(seed)
		self.update_target()

	def runLearnPhase(self):
		super().runLearnPhase()
		self.update_target(copy=False)

	def runOneInteraction(self, state):
		# agent policy that uses the observation and info
		reward = super().runOneInteraction(state)
		if self.interaction_before_copy_target == 0:
			self.interaction_before_copy_target = 10_000
			self.update_target()
		return reward

	def update_target(self, copy=True):
		tau = 1 if copy else self.tau
		for param_duplicat, param_source in zip(self.QNNt.parameters(), self.QNN.parameters()):
			param_duplicat.data.copy_(
				(1-tau) * param_duplicat.data + tau * param_source.data)
