import numpy  as np

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout



def plot_graph(state, title):
	
    G = nx.from_numpy_matrix(state, create_using = nx.MultiDiGraph())

	# pos = nx.spring_layout(G)
    pos=nx.nx_pydot.graphviz_layout(G, prog='dot')
    nx.draw_networkx(G, pos, with_labels=True,
					 node_color='lightgray', node_size=500, node_shape='o',
					 arrowstyle="-|>", arrowsize=20, width=2)


    plt.title(title)
    plt.axis('off')
    plt.show()
    plt.savefig('graph.png')
    
