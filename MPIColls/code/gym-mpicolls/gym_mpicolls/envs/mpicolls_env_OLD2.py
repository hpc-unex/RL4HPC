import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_mpicolls.envs.matrix import Matrix
from gym_mpicolls.envs.pair   import Pair
from gym_mpicolls.envs.edge   import Edge
import numpy as np


def get_reward_DEPTH(s):

    if np.min(np.max(s, axis=0)) == 0:
        reward = 0
    else:
        reward = -np.max(np.max(s, axis=0)) # max stage

    return reward

def get_reward_JUMPS(s, src, dst, stage):

    # Assume: Round-Robin, M=2, any P
    M_src = src % 2
    M_dst = dst % 2

    if M_src == M_dst:
        reward = -1
    else:
        reward = -10

    return reward

def get_reward_COMPLEX(s, src, dst, stage):

    # Number of nodes
    M = [0, 1, 2, 0, 1, 2, 0, 2]
    # Assume: Round-Robin, M any, any P
    # M = [0, 1, 2, 0, 1, 2, 0, 1]
    # Assume: Sequential, M any, any P
    # M = [0, 0, 0, 1, 1, 1, 2, 2]

    M_src = M[src]
    M_dst = M[dst]

    if M_src == M_dst:
        reward = -1
    else:
        reward = -(s.shape[0])

    envs = 0 # np.count_nonzero(s[src,:] != -1)
    reward = reward - stage - envs
    return (reward)

def get_reward_CONCURRENT(s, src, dst, stage):

    # Number of nodes
    M = [0, 1, 2, 0, 1, 2, 0, 2]
    # Assume: Round-Robin, M any, any P
    # M = [0, 1, 2, 0, 1, 2, 0, 1]
    # Assume: Sequential, M any, any P
    # M = [0, 0, 0, 1, 1, 1, 2, 2]

    M_src = M[src]
    M_dst = M[dst]

    if M_src == M_dst:       # SHM
        reward = -1
    else:                    # NET
        concurrent = np.count_nonzero(s[:,:] == stage) - 1
        if concurrent > 0:   # Concurrency in NET
            reward = -concurrent
        else:                # No concurrency in NET
            reward = -(s.shape[0])

    return reward

def get_reward(s, src, dst, stage):

    # return get_reward_DEPTH(s)
    # return get_reward_COMPLEX(s, src, dst, stage)
    # return get_reward_JUMPS(s, src, dst, stage)
    # return get_reward_CONCURRENT(s, src, dst, stage)

    # Devuelve menos R cuando mas lineal es el arbol.
    return np.max(np.max(s, axis=0))



class MPICollsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, P=4, M=None):
        super(MPICollsEnv, self).__init__()

        # ACTION space
        self.action_space = Edge(P=P)

        # STATES space
        self.observation_space = Matrix(P=P)

        self.state = None
        self.P = P

        if M == None:
            self.M = np.arange(0, P, 1)
        else:
            self.M = M


    # This function returns a reward that favors Linear Trees.
    def step (self, action):

        src, dst = action

        # Multicore machine. Hop is a network communication.
        hop = 0
        if self.M[src] != self.M[dst]:
            hop = 1

        current_depth = np.max(np.max(self.state, axis=0))

        # Stage: last stage of sending/receiving of a process + 1:
        stage = np.maximum( np.amax(self.state[src,:]), np.amax(self.state[:,src]) ) + 1

        valid = True
        reward = 0
        done  = False

        # Case 1: A process sends w/o previously receiving:
        recv = np.max(self.state, axis=0)  # Last stage of reception of each process
        if (recv[src] <= 0) or (recv[src] >= stage):
            valid = False

        # Case 2: A process receives an already received message:
        #         axis=0 is by COLUMNS
        if (recv[dst] > 0):
            valid = False

        # Case 3: send self:
        if (src == dst):
            valid = False

        if (valid == False):
            reward = -1
        else: # if (valid == True):
            self.state[src,dst] = stage

        # done = True if all processes received (columns have a value != -1)
            if np.min(np.max(self.state, axis=0)) == 0:

                # No añade profundidad:
                if current_depth <= stage:
                    reward = 2
                else:
                    reward = 1

            else: # Tree completed
                done = True
                reward = get_reward(self.state, src, dst, stage)

        return np.array(self.state), reward, done, {}



    # This function returns a reward that favors Linear Trees.
    def step_LINEAR(self, action):

        src, dst = action

        current_depth = np.max(np.max(self.state, axis=0))

        # Stage: last stage of sending/receiving of a process + 1:
        # cols = np.max(s, axis=0)  # Last stage of reception of each process
        stage = np.maximum( np.amax(self.state[src,:]), np.amax(self.state[:,src]) ) + 1
        # print("Stage: ", stage)

        #print("STATE: (env) - BEFORE")
        #print(self.state)

        # Valid or Invalid action?
        valid = True
        reward = 0
        done  = False

        # Case 1: A process sends w/o previously receiving:
        recv = np.max(self.state, axis=0)  # Last stage of reception of each process
        if (recv[src] <= 0) or (recv[src] >= stage):
            valid = False
            # done = True
            # print("INVALID Action (1)")
            reward += -1
            # self.state[src,dst] = stage

        # Case 2: A process receives an already received message:
        #         axis=0 is by COLUMNS
        # recv = np.where(self.state != 0, 1, 0)
        # if (recv.sum(axis=0)[dst] > 0):
        if (recv[dst] > 0):
            valid = False
            # done = True
            # print("INVALID Action (2)")
            reward += -1
            # self.state[src,dst] = stage

        # Case 3: send self:
        if (src == dst):
            valid = False
            # done = True
            # print("INVALID Action (3)")
            reward += -1

        if (valid == False):
            reward = -1
        else: # if (valid == True):
            self.state[src,dst] = stage

        # done = True if all processes received (columns have a value != -1)
        # done = not np.min(np.max(self.state, axis=0)) < 0
            if np.min(np.max(self.state, axis=0)) == 0:
                # done = False
                reward = 0

                if current_depth <= stage:
                    reward = self.P
                else:
                    reward = self.P // 2

            else: # Tree completed
                done = True
                reward = self.P # get_reward(self.state, src, dst, stage)
                # print("+")

            # reward = get_reward(self.state, src, dst, stage)

        #print("STATE: (env) - AFTER")
        #print(self.state)
        #print("Reward: ", reward)

        return np.array(self.state), reward, done, {}
        # return self.state, reward, done, {}

    def step_OLD(self, action):

        src, dst = action

        # Stage: last stage of sending/receiving of a process + 1:
        # cols = np.max(s, axis=0)  # Last stage of reception of each process
        stage = np.maximum( np.amax(self.state[src,:]), np.amax(self.state[:,src]) ) + 1
        # print("Stage: ", stage)

        #print("STATE: (env) - BEFORE")
        #print(self.state)

        # Valid or Invalid action?
        valid = True
        reward = 0

        # Case 1: A process sends w/o previously receiving:
        recv = np.max(self.state, axis=0)  # Last stage of reception of each process
        if (recv[src] <= 0) or (recv[src] >= stage):
            valid = False
            done = True
            # print("INVALID Action (1)")
            reward += -3

        # Case 2: A process receives an already received message:
        #         axis=0 is by COLUMNS
        recv = np.where(self.state != 0, 1, 0)
        if (recv.sum(axis=0)[dst] > 0):
            valid = False
            done = True
            # print("INVALID Action (2)")
            reward += -2

        # Case 3: send self:
        if (src == dst):
            valid = False
            done = True
            # print("INVALID Action (3)")
            reward += -1


        if (valid != False):

            self.state[src,dst] = stage

            # done = True if all processes received (columns have a value != -1)
            # done = not np.min(np.max(self.state, axis=0)) < 0
            if np.min(np.max(self.state, axis=0)) == 0:
                done = False
                reward = 10
            else:
                done = True
                reward = 2
                print("+")

            # reward = get_reward(self.state, src, dst, stage)

        #print("STATE: (env) - AFTER")
        #print(self.state)
        #print("Reward: ", reward)

        return np.array(self.state), reward, done, {}
        # return self.state, reward, done, {}


    def reset(self):

        root = self.observation_space.root
        P = self.observation_space.P
        # self.state = np.ones((P, P), dtype=np.int) * -1
        self.state = np.zeros((P, P), dtype=np.int)
        # print("RESET STATE: ", P, self.state)

        self.state[root, root] = 1

        return np.copy(self.state)


    def render(self, mode='human', close=False):
        print(self.state)


    def close(self):
        pass
