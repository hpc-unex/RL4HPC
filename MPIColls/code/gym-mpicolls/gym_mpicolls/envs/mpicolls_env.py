import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_mpicolls.envs.matrix import Matrix
from gym_mpicolls.envs.pair   import Pair
from gym_mpicolls.envs.edge   import Edge
import numpy as np
import subprocess




def state_to_list(state, P):
	
	l = list()

	# print("Graph for: ", P, " processes.")
	# print(state, flush=True)
	
	for s in range(P):
		for r in range(P):
			stage = state[s,r]
			if stage != 0:
				# print(s, r, stage, flush=True)
				l.append((s, r, stage))
			
	return l



def get_reward_mpi(state, M, params, iter=0):
	
	t = 0.0
		
	graph_file  = params['graph_file']
	output_file = params['output_file']

	l_graph = state_to_list(state, params["P"])
	l_graph.sort(key=lambda x:(x[2], x[1], x[0])) # Sort by: Stage - Dst - Src

	# Graph file containing the description of the graog of communications
	g_file = open(graph_file, 'w');
	g_file.write('\n'.join('%d %d %d' % (s,r,stage) for (s,r,stage) in l_graph) )
	g_file.close()

	# Output file contains output messages
	o_file = open(output_file, 'a')
	o_file.write("\nOutput File: " + str(iter) + "\n")
	o_file.write('\n'.join('%d %d %d' % (s,r,stage) for (s,r,stage) in l_graph) )


	exec_command = params["exec_command"]
	# print(exec_command)
	
	# -n $numProcs  --mca btl $COMMS $OMPIALG  --bind-to core  -report-bindings --display-map  -nooversubscribe  $BIN_FOLDER/IMB-MPI1 $IMB_OPTIONS
	proc = subprocess.run(exec_command,
						  stdout=subprocess.PIPE,
						  stderr=subprocess.PIPE,
						  shell=False,
						  universal_newlines=True)
							# check=True,
	
	# (output, err) = proc.communicate()
	output = proc.stdout
	err = proc.stderr
				
	if (err != None):
		print(err)


	data = [0.0, 0.0, 0.0, 0.0, 0.0]

	for line in output.splitlines():
		if len(line) > 0:
			w = line.lstrip(" \t")
			# print(w, flush=True)
			if (w[0] != '#'):
				# print(w, flush=True)
				try:
					data = [float(i) for i in w.split()]
				except:
					None  # print(w.split(), flush=True)

	# proc.wait()

	# Format of IMB return
	# data = [m , reps, t_min, t_max, t_avg]

	t = data[4] # AVG

	P = int(params['P'])
	o_file.write("\nREWARD: " + str(t/P) + "\n")
	# o_file.write(output)
	o_file.write("IMB: " + str(data).strip('[]') + "\n")
	o_file.close()
			
	return t/P


def get_reward_tlop(s, M, P, iter=0):
	
	# Obtain reward as communication time
	M_str = str(M).strip('[]()')
	# print(M_str)

	proc = subprocess.run(["/Users/jarico/Documents/Investigacion/Software/RL/bcast",
						   "-P", str(P),
						   "-m", "1024",
						   "-M", M_str,
						   "-c", "MPI_Bcast",
						   "-a", "binomial",
						   "-n", "IB",
						   "-s", "CIEMAT"],
						  stdout=subprocess.PIPE,
						  stderr=subprocess.PIPE,
						  shell=False,
						  universal_newlines=True)
		
	"""
	output = proc.stdout
	err    = proc.stderr
	
	if (err != None):
	print("ERR: ", err)
	print("OUTPUT: ", output)
	"""
				
	time = float(proc.stdout)
	
	return time


def get_reward(s, M, P):
	
	# Devuelve menos R cuando mas lineal es el arbol.
	# return np.max(np.max(s, axis=0))
	
	# Devuelve el número de comunicationes locales (!"hops")
	hops = 0
	acc  = 0
	for i in range(P):
		for j in range(P):
			if s[i,j] > 1:
				acc += 1
				if (M[i] != M[j]):
					hops += 1
		
	return (acc - hops)


class MPICollsEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	
	
	def __init__(self, params):
		super(MPICollsEnv, self).__init__()
		
		self.params = params
		print(self.params)

		# ACTION space
		self.action_space = Edge(P=self.params["P"])
	
		# STATES space
		self.observation_space = Matrix(P=self.params["P"])
		
		self.state = None
		self.P = self.params["P"]
		
		self.M = np.arange(0, self.params["M"], 1)
	
		self.trajectory = 0
	
	
	
	# This function returns a reward that favors Linear Trees.
	def step (self, action):
		
		src, dst = action

		# Stage: last stage of sending/receiving of a process + 1:
		stage = np.maximum( np.amax(self.state[src,:]), np.amax(self.state[:,src]) ) + 1
		valid = True
		reward = 0
		done  = False
		
		# Case 1: A process sends w/o previously receiving:
		recv = np.max(self.state, axis=0)  # Last stage of reception of each process
		if (recv[src] <= 0) or (recv[src] >= stage):
			valid = False
		
		# Case 2: A process receives an already received message:
		#         axis=0 is by COLUMNS
		if (recv[dst] > 0):
			valid = False
		
		# Case 3: send self:
		if (src == dst):
			valid = False
		
		if (not valid):
			reward = -1
			# reward = 0
		else: # if (valid == True):
			self.state[src,dst] = stage
			reward = 0.0
			# reward = 1
			
			# done = True if all processes received (columns have a value != -1)
			#print(self.state)
			#print(np.max(self.state, axis=0))
			#print(np.min(np.max(self.state, axis=0)))
			if np.min(np.max(self.state, axis=0)) > 0:
				done = True
				reward = np.max(np.max(self.state, axis=0))
				#print("Reward: ", reward)
				# reward = self.P * get_reward(self.state, self.M, self.P)
				# reward = -1 * get_reward_tlop (self.state, self.M, self.params)
				# reward = -1 * get_reward_mpi (self.state, self.M, self.params, self.trajectory)
				# print("REWARDS: ", reward, flush=True)
				self.trajectory += 1

		return np.array(self.state), reward, done, {}


		
	def reset(self):

		root = self.observation_space.root
		P = self.observation_space.P
		# self.state = np.ones((P, P), dtype=np.int) * -1
		self.state = np.zeros((P, P), dtype=np.int)
		# print("RESET STATE: ", P, self.state)
		
		self.state[root, root] = 1
		
		return np.copy(self.state)

	
		
	def render(self, mode='human', close=False):
		print(self.state)
	
	
	
	def close(self):
		pass
