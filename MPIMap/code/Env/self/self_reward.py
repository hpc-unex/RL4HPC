import numpy as np


def get_reward(state, actions, params):


	P       = params["P"]
	M       = params["M"]

	valid  = True
	r      = 0
	done   = False

	n_errors = 0
	n_intra  = 0
	n_inter  = 0

	# ERRORS
	unique_elements, counts_elements = np.unique(actions, return_counts=True)
	Q = P // M



	counts_elements = counts_elements - Q
	n_errors = (M - np.size(counts_elements)) * Q + np.sum(np.abs(counts_elements))

	r = n_errors


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

	if n_errors > 0:
		valid = False
	else:
		valid = True


	# INTER
	for src in range(0, P):
		for dst in range(0, P):
			if state[src,dst] != 0:
				if actions[src] != actions[dst]: # different node
					n_inter += 1

	r = n_inter + n_errors

	# rewards[-1] = -(self.ro * n_intra + (1 - self.ro) * n_inter + self.c * n_errors)

	info = {"valid": valid, "n_errors": n_errors, "n_intra": n_intra, "n_inter": n_inter}
	
	return r, info
