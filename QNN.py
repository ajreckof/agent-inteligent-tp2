import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
	if isinstance(m, nn.Linear):
		nn.init.uniform_(m.weight)
		m.bias.data.fill_(0.01)

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
		self.seq.apply(init_weights)

	def forward(self, state):
		"""Forward pass"""

		return self.seq(state)


