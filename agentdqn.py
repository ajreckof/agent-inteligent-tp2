import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import gymnasium as gym
import copy
import warnings
from cmath import inf

from agentglouton import AgentGlouton
from torch.optim.lr_scheduler import ChainedScheduler

def class_name(o):
		if not o:
			return "None"
		klass = o.__class__
		return klass.__qualname__

class AgentDQN(AgentGlouton):
	"""Agent qui utilise l'algorithme DQN."""

	def __init__(self, env, eps=1, gamma=0.99, planner = None, optim = optim.Adam, base_lr = 0.001, repr= ""):
		"""Constructeur.
		
		Params
		======
		"""
		super().__init__(env = env, eps = eps)
		self.gamma = gamma
		self.observation = self.env.reset()
		self.terminated = False
		self.truncated = False
		self.replay_buffer = ReplayBuffer(100_000,128)
		# avant d'apprendre on attend d'avoir 64 interactions
		self.interaction_before_learn_phase = 128
		self.QNN = QNN()
		self.loss_func = torch.nn.MSELoss(reduction="sum")
		self.optim = optim(self.QNN.parameters(), lr=base_lr)
		if planner :
			if isinstance(planner,list):
				self.scheduler = [x(self.optim) for x in planner]
			else:
				self.scheduler = planner(self.optim)
		else : 
			self.scheduler = None
		self.repr = repr



	def runOneEpisode(self):
		total_reward = 0
		self.observation, _ = self.env.reset()
		self.terminated = False
		self.truncated = False 

		while not (self.truncated or self.terminated):
			total_reward += self.runOneInteraction(self.observation)
		if self.scheduler :
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				if isinstance(self.scheduler,list):
					for x in self.scheduler :
						x.step(total_reward)
				else:
					self.scheduler.step(total_reward)
		
		return total_reward
	
	def get_value(self, state):
		value, _ = torch.max(self.QNN(state), dim=1)
		return value

	def runLearnPhase(self):
		states, actions, next_states, rewards, dones = self.replay_buffer.sample()
		actions = actions.to(torch.int64)
		outputs = torch.gather(self.QNN(states), 1, actions)
		value_next_states = self.get_value(next_states)
		t = rewards.unsqueeze(1) + self.gamma * (1 - dones) * \
			value_next_states.unsqueeze(1)
		loss = self.loss_func(outputs, t)
		loss.backward()
		self.optim.step()
		self.optim.zero_grad()

	def runOneInteraction(self, state):
		# agent policy that uses the observation and info
		action = self.getAction(state)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.observation, reward, self.terminated, self.truncated, _ = self.env.step(action)
		self.replay_buffer.add(
			state, action, self.observation, reward, self.terminated or self.truncated)
		self.interaction_before_learn_phase -= 1
		if self.interaction_before_learn_phase == 0:
			# on réapprend toute les 4 intéractions
			self.interaction_before_learn_phase = 4
			self.runLearnPhase()
		return reward 
		
	def get_current_lr(self):
		return self.optim.param_groups[0]['lr']

	def runFullLearning(self, eps_min = 0.01, eps_decay = 0.999, n_ep = 4_000, verbose = True):
		rewards = []
		state_memory = copy.deepcopy(self.QNN.state_dict())
		state_memory_avg_reward = -inf

		for i in range(n_ep):
			if i % 100 == 0 :
				avg_reward = sum(rewards[-50:])/50 if rewards else -inf
				if avg_reward > state_memory_avg_reward :
					state_memory_avg_reward = avg_reward
					state_memory = copy.deepcopy(self.QNN.state_dict())
				if verbose :
					print("epoch :", i)
					print("learning rate :", self.optim.param_groups[0]['lr'] )
					print("epsilon :",self.eps)
					print("average reward on last 50 epidodes :", avg_reward)
					print()
				else :
					print(f"epoch : {i} ({self.repr})")
				
			rewards.append(self.runOneEpisode())

			self.eps = max(self.eps * eps_decay, eps_min)
		self.QNN.load_state_dict(state_memory)
		return rewards


		

		