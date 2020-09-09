import numpy as np


def get_reward(s, M, P):
	
	# Devuelve menos R cuando mas lineal es el arbol.
	return np.max(np.max(s, axis=0))
	
	# Devuelve el número de comunicationes locales (!"hops")
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

