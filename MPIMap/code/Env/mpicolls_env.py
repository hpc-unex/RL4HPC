import numpy as np
import math
import subprocess
import sys

from self.self_reward import get_reward
#from tLop.tLop_reward import get_reward
# from .mpi.mpi_reward   import get_reward



class MPICollsEnv(object):


	def __init__(self, params):
		super(MPICollsEnv, self).__init__()

		self.params = params

		self.state = None
		self.P = self.params["P"]
		self.root = self.params["root"]
		#Devuelve valores uniformemente espaciados dentro de un intervalo dado
		#Generará un vector con dos posiciones 0 y 1, ya que el punto de stop no se incluye
				#Arange -> Start = 0, Stop = M = 2, step = 1
		self.M = np.arange(0, self.params["M"], 1)
		print(self.M)

		self.verbose       = params["verbose"]  # Verbosity
		self.verbosity_int = params["verbosity_interval"]

		# ACTION space
		self.action_space = self.P

		# STATES space
		self.observation_space = self.P

		self.experience = 0

		self.less_states = []
		self.less_reward = -np.inf
		#np.inf es igual a infinito



	# This function returns a reward that favors which process put in some place or not
	def step (self, action):
		#Saca la casilla correspondiente es decir destino y fuente

		p,h = action

		valid  = True
		reward = 0
		done   = False


		#Case 1: Voy a comprobar que la fila obtenida con el sample está vacía
		if (np.any(self.state[p,:])) :
			valid = False

		# Case 2: comprobamos si la columna está vacía

		if (np.any(self.state[:,h])):
			valid = False

		if (not valid):
			reward = -1
			# reward = 0.0
		else: # valid == True
			self.state[p,h] = 1
			reward = 0.0

			if np.min(np.max(self.state, axis=0)) > 0:
				done = True
				reward=  1

				#reward = - math.sqrt(get_reward(self.state, self.params)) #probalidad logaritmica


				self.experience += 1
				self.less_states.append(np.copy(self.state))

		return np.array(self.state), reward, done, {"valid": valid}



	def reset(self):
		#Aqui es donde inciciamos el estados
		self.state = np.zeros((self.P, self.P), dtype=np.int)
		#En mi caso esto lo debería de eliminar por que no hay ningún root, por lo tanto estaría todo inciado a cero
		#self.state[self.root, self.root] = 1
		self.experience = 0

		return np.copy(self.state)



	def render(self, episode):

		if self.verbose and not (episode % self.verbosity_int):
			print(self.state)
			print(self.less_states[-1])
