import numpy as np
from gym import spaces
from gym.spaces.space import Space


class Pair(Space):
    """
    - The Pair space consists of a pair of (different) integer values.
    - It is parametrized by passing a highest values
    - Can be initialized as
        Pair(8)
    """
    def __init__(self, P):

        """
        P: highest value
        """
        assert P >= 1, "Error, P must be higher than 0"
        self.P = P

        super(Pair, self).__init__((), np.int64)

    def sample(self):

        pair = np.zeros(2, dtype=np.int64)
        pair[0] = np.random.randint(low=0, high=self.P, size=1)
        while True:
            pair[1] = np.random.randint(low=0, high=self.P, size=1)
            if (pair[0] != pair[1]):
                break;

        return pair
        # return (self.np_random.random_sample(self.nvec.shape)*self.nvec).astype(self.dtype)

    def contains(self, x):
        pass
        #if isinstance(x, list):
        #    x = np.array(x)  # Promote list to array for contains check
        #return x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all()

    def __repr__(self):
        return "Pair({})".format(self.P)
