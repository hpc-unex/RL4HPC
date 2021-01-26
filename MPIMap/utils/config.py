import json
import sys


# Read config from the .json file:
def read_config ():
	
	config = {}
	config_file = './config.json'
	if len(sys.argv) == 2:
		config_file = sys.argv[1]
	
	try:
		with open(config_file, 'r') as js:
			config = json.load(js)

	except EnvironmentError:
		print ('Error: file not found: ', config_file)
		
	return config
