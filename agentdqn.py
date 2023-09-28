import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from agentglouton import AgentGlouton


class AgentDQN(AgentGlouton):
	"""Agent qui utilise l'algorithme DQN."""

	def __init__(self, env, eps=1, gamma=0.9, learning_rate=0.05, seed=0):
		"""Constructeur.
		
		Params
		======
			seed (int): random seed
		"""
		super().__init__(seed = seed, eps = eps)

		self.gamma = gamma
		self.env = env
		self.observation = self.env.reset()
		self.terminated = False
		self.truncated = False
		self.replay_buffer = ReplayBuffer(100_000, 64, seed)
		# avant d'apprendre on attend d'avoir 64 interactions
		self.interaction_before_learn_phase = 64
		self.QNN = QNN(seed)
		self.loss_func = torch.nn.MSELoss()
		self.optim = torch.optim.SGD(self.QNN.parameters(), lr=learning_rate)



	def runOneEpisode(self):
		total_reward = 0
		self.observation, _ = self.env.reset()
		self.terminated = False
		self.truncated = False 

		while not (self.truncated or self.terminated):
			total_reward += self.runOneInteraction(self.observation)
		return total_reward

	def runLearnPhase(self):
		states, actions, next_states, rewards, dones = self.replay_buffer.sample()
		actions = actions.to(torch.int64)
		outputs = torch.gather(self.QNN(states), 1, actions)
		value_next_states, _ = torch.max(self.QNN(next_states), dim=1)
		t = rewards.unsqueeze(1) + self.gamma * (1 - dones) * \
			value_next_states.unsqueeze(1)
		loss = self.loss_func(outputs, t)
		loss.backward()
		self.optim.step()
		self.optim.zero_grad()

	def runOneInteraction(self, state):
		# agent policy that uses the observation and info
		action = self.getAction(state)
		self.observation, reward, self.terminated, self.truncated, _ = self.env.step(
			action)
		self.replay_buffer.add(
			state, action, self.observation, reward, self.terminated or self.truncated)
		self.interaction_before_learn_phase -= 1
		self.interaction_before_copy_target -= 1
		if self.interaction_before_learn_phase == 0:
			# on réapprend toute les 4 intéractions
			self.interaction_before_learn_phase = 4
			self.runLearnPhase()
		return reward 


		

		