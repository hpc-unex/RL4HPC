import os
from subprocess import PIPE, Popen, call
import numpy as np
from decimal import Decimal
from numpy import sort, floor

import sys





#  MAIN

M   = int(sys.argv[1])
net = sys.argv[2]
m   = int(sys.argv[3])
input_file = sys.argv[4]
output_file = sys.argv[5]

print('Parameters: (M, net, m): ', M, net, m)
print("Reading from the file: ", input_file)
print("Writing to the file: ", output_file)
