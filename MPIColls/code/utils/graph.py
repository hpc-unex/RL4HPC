import numpy  as np

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout



def plot_graph(g, label):
	
	G = nx.DiGraph()
	G = nx.from_numpy_matrix(g)
	nx.draw_networkx(G, arrows=True, with_labels=True, node_color='w', label=label) #, layout='tree')
	
	plt.axis('off')



