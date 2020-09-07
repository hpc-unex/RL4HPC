import numpy as np
from gym import spaces
from gym.spaces.space import Space


class Edge(Space):
    """
    - The Edge space consists of three integer values:
        * Source
        * Destination
        * Stage of sending in the tree (graph)
    - Can be initialized as
        Edge(P = 8)
        P = number of the processes in send/receive (edge in a graph)
    """
    def __init__(self, P):

        """
        P: highest value
        """
        assert P >= 1, "Error, P must be higher than 0"
        self.P = P

        super(Edge, self).__init__((), np.int64)

    def sample(self):
                
        print("ERROR: [sample] not implemented")
        
        src   = np.random.randint(self.P)
        dst   = np.random.randint(self.P)
        stage = np.random.randint(self.P)
        
        return src, dst, stage


    def contains(self, x):
        pass
        #if isinstance(x, list):
        #    x = np.array(x)  # Promote list to array for contains check
        #return x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all()

    def __repr__(self):
        return "Edge({})".format(self.P)

    def __eq__(self, other):
        return isinstance(other, Edge) and self.P == other.P
