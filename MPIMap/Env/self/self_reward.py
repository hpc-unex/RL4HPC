import numpy as np
import torch
import sys

# A simple class to maintain a moving average baseline
class Baseline:

	baseline = 0.0
	episode  = 0

	def update(r):
		Baseline.episode += 1
		Baseline.baseline = Baseline.baseline + (1.0 / Baseline.episode) * (r - Baseline.baseline)

	def get():
		return Baseline.baseline



def get_reward(state, actions, params):

	# Policy network
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	graph_params = params["Graph"]
	P = graph_params["P"]

	D = torch.zeros((P, P)).detach().to(device)
	for i in range(0, P):
		for j in range(0, P):
			if i == j:                         ## Same processor
				value = 1 # 10
			elif actions[i] == actions[j]:     ## Same node
				value = 1 # 2
			else:                              ## Different node
				value = 2 # 1
			D[i,j] = value

	r = torch.sum(state * D).item() # - Baseline.get()
	# print("Coste: ", r)
	r = -r / 100000

	valid = True

	info = {"valid": valid, "reward": r, "baseline": Baseline.get()}

	# update baseline
	# Baseline.update(r)

	return r, info


def get_reward_OTRO(state, actions, params):

	# Baseline.episode += 1

	P = params["P"]
	M = params["M"]

	n_errors = 0
	n_intra  = 0
	n_inter  = 0

	# ERRORS
	unique_elements, counts_elements = np.unique(actions.to("cpu"), return_counts=True)
	Q = P // M

	counts_elements = counts_elements - Q
	n_errors = (M - np.size(counts_elements)) * Q + np.sum(np.abs(counts_elements))

	# INTER-COMMUNICATIONS
	for src in range(0, P):
		for dst in range(0, P):
			if state[src,dst] != 0:
				if actions[src] != actions[dst]: # different node
					n_inter += 1

	r = (n_inter + n_errors) - Baseline.get()
	# r = -(self.ro * n_intra + (1 - self.ro) * n_inter + self.c * n_errors)

	if n_errors > 0:
		valid = False
	else:
		valid = True

	info = {"valid": valid, "reward": r, "baseline": Baseline.get()}

	# update baseline
	Baseline.update(r)


	return r, info





# OTRAS POSIBILIDADES:


	"""
	nodes = np.zeros(self.M, dtype=np.int)
	for i,a in enumerate(actions):
		nodes[a] += 1
		if (nodes[a] > Q):
			n_errors += 1
			rewards[i] = -1.0
	"""

	"""
	ccp = 0.25
	n_over = 0

	for src in range(0, self.P):
		for dst in range(0, self.P):
			if self.comms[src,dst] != 0:
				if actions[src] == actions[dst]: # Same or different node
					n_intra += 1
					n_elem = (actions == actions[src]).sum()
					if n_elem.item() > Q:
						# print(actions)
						# print("src, dst, Q / n_elem: ", src, dst, Q, n_elem.item())
						n_over += 1 + np.abs(n_elem.item() - Q) * ccp
				else:
					n_inter += 1
					# n_elem_src = (actions == actions[src]).sum()
					# n_elem_dst = (actions == actions[dst]).sum()
					# if n_elem_src.item() > Q:
					# 	n_errors += np.abs(n_elem_src.item() - Q) * ccp
					# elif n_elem_dst.item() > Q:
					# 	n_errors += np.abs(n_elem_dst.item() - Q) * ccp
	"""
