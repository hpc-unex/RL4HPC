import json
import sys
import numpy as np

# Read config from the .json file:
def read_config ():
	
	config = {}
	config_file = '/home/sergio/Documentos/DOCTORADO/reinforce/GYMMap/utils/config.json'
	if len(sys.argv) == 2:
		config_file = sys.argv[1]
	
	try:
		with open(config_file, 'r') as js:
			config = json.load(js)

	except EnvironmentError:
		print ('Error: file not found: ', config_file)
		
	return config
	
	
def adjacency (P, comms, msg):

    adj = np.zeros((P, P), dtype=np.int)
    #for i, edge in enumerate(comms["edges"]):
    for i, edge in enumerate(comms):
        src = edge[0]
        dst = edge[1]
        adj[src, dst] = msg[i]
        adj[dst, src] = msg[i]

        # Symmetric

    return adj

