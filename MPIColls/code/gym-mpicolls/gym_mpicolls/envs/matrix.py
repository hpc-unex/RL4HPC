import numpy as np
from gym import spaces
from gym.spaces.space import Space


class Matrix(Space):
    """
    - The Matrix action space consists of a square matrix of integer values.
    - It is parametrized by passing a matrix dimension
    - Can be initialized as
        Matrix(8)
    """
    def __init__(self, P, root=0):

        """
        P: dimension of the matrix
        """
        self.P = P
        self.root = root
        # self.matrix = np.ones((self.P, self.P), dtype=np.int64) * -1

        super(Matrix, self).__init__((), np.int64)

    def sample(self):
        return 0
        # return (self.np_random.random_sample(self.nvec.shape)*self.nvec).astype(self.dtype)

    def contains(self, x):
        pass
        #if isinstance(x, list):
        #    x = np.array(x)  # Promote list to array for contains check
        #return x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all()

    def __repr__(self):
        return "Matrix({}x{})".format(self.P, self.P)

    def __eq__(self, other):
        return isinstance(other, Matrix) and self.P == other.P and self.root == other.root
    
