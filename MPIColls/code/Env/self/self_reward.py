import numpy as np


def get_reward(s, params):

	# Return higher reward for a less deeper tree

	P = params["P"]
	return (P - np.max(np.max(s, axis=0)))

	# Return the number of local communications (!"hops")
	"""
	hops = 0
	acc  = 0
	for i in range(P):
		for j in range(P):
			if s[i,j] > 1:
				acc += 1
				if (M[i] != M[j]):
					hops += 1

	return (acc - hops)
	"""
