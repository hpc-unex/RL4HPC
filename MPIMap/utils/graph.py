import numpy  as np
import torch

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


def adjacency (P, comms):

    adj = torch.zeros((P, P), dtype=np.int)
    for i, edge in enumerate(comms["edges"]):
        src = edge[0]
        dst = edge[1]
        adj[src, dst] = comms["m"][i]
        adj[dst, src] = comms["m"][i]
        # Symmetric

    return adj
