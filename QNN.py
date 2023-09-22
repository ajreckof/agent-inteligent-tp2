import torch
import torch.nn as nn
import torch.nn.functional as F

class QNN(nn.Module):
	"""Reseau de neurones pour approximer la Q fonction."""

	def __init__(self, seed=0):
		"""Initialisation des parametres ...
		Params
		======
			seed (int): Random seed
		"""
		super().__init__()
		self.seed = torch.manual_seed(seed)

		self.seq = nn.Sequential(
			nn.Linear(8, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 4)
		)

	def forward(self, state):
		"""Forward pass"""

		return self.seq(state)


