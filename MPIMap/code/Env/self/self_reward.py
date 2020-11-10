import numpy as np


def get_reward(s, params):

<<<<<<< HEAD
	# Return higher reward for a less deeper tree

	P = params["P"]
	return (P - np.max(np.max(s, axis=0)) + 1)

	# Return the number of local communications (!"hops")
=======
	# Devuelve menos R cuando mas lineal es el arbol.
	return np.max(np.max(s, axis=0))

	# Devuelve el número de comunicationes locales (!"hops")
>>>>>>> f80fa1da49810c0f077560a3c057db29f54156a5
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
