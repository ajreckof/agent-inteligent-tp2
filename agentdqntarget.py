

from QNN import QNN
import torch.optim as optim
from torch import max
from agentdqn import AgentDQN


class AgentDQNTarget(AgentDQN):
	"""Agent qui utilise l'algorithme DQN."""

	def __init__(self, env, eps=1, gamma=0.99, planner = None, optim = optim.Adam, base_lr = 0.001, tau=0.001, repr= ""):
		"""Constructeur.
		
		Params
		======
			seed (int): random seed
		"""
		super().__init__(env=env, eps=eps, gamma=gamma,  base_lr = base_lr, planner= planner, optim= optim, repr = repr)
		self.tau = tau
		self.interaction_before_copy_target = 10_000

		self.QNNt = QNN()
		self.update_target()

	def get_value(self, state):
		value, _ = max(self.QNNt(state), dim=1)
		return value


	def runLearnPhase(self):
		super().runLearnPhase()
		self.interaction_before_copy_target -= 1
		if self.interaction_before_copy_target == 0:
			self.interaction_before_copy_target = 10_000
			self.update_target()
		else:
			self.update_target(copy=False)

	def update_target(self, copy=True):
		tau = 1 if copy else self.tau
		for param_duplicat, param_source in zip(self.QNNt.parameters(), self.QNN.parameters()):
			param_duplicat.data.copy_(
				(1-tau) * param_duplicat.data + tau * param_source.data)
