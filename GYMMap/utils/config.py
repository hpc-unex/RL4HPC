import json
import sys
import numpy as np

# Read config from the .json file:
def read_config ():
	
	config = {}
	config_file = '/home/usuario/reinforce/GYMMap/utils/config.json'
	if len(sys.argv) == 2:
		config_file = sys.argv[1]
	
	try:
		with open(config_file, 'r') as js:
			config = json.load(js)

	except EnvironmentError:
		print ('Error: file not found: ', config_file)
		
	return config
	
	
def adjacency (P, comms):

    adj = np.zeros((P, P), dtype=np.int)
    for i, edge in enumerate(comms["edges"]):
        src = edge[0]
        dst = edge[1]
        adj[src, dst] = comms["m"][i]
        adj[dst, src] = comms["m"][i]
        # Symmetric

    return adj
