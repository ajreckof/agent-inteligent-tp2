import numpy as np
import random
import torch

from QNN import QNN

class AgentGlouton():
	"""Agent qui utilise la prédiction de son réseau de neurones pour choisir ses actions selon une stratégie d’exploration (pas d'apprentissage)."""

	def __init__(self, env, seed=0, eps=1):
		"""Constructeur.
		
		Params
		======
			seed (int): random seed
		"""
		self.QNN = QNN()
		self.eps = eps
		self.env = env

	def __getstate__(self):
		return {
			"env": self.env,
			"eps": self.eps,
			"state_dict": self.QNN.state_dict(),
		}

	def __setstate__(self, state):
		self.eps = state["eps"]
		self.QNN = QNN()
		self.QNN.load_state_dict(state["state_dict"])
		self.env = state["env"]

	def getAction(self, state):
		if random.random() < self.eps:
			return self.env.action_space.sample()
		else:
			q_values = self.QNN(torch.from_numpy(state))
			return int(torch.argmax(q_values))

	
		



