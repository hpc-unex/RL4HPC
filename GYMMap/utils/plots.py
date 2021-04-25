import numpy as np
import matplotlib.pyplot as plt


def plot(info, mode='human'):

	fig = plt.figure("Reward")
	#j = np.arange(len(self.info['episode_reward']))

	X_AXIS = 100
	x = np.arange(0, X_AXIS, 1)
	j = np.array(info['episode_reward'])
	if len(j)%X_AXIS != 0:
		j = j[len(j)%X_AXIS:]


	j = j.reshape((X_AXIS,-1))
	j_max  = j.max(axis=1)
	j_min  = j.min(axis=1)
	j_mean = j.mean(axis=1)
	j_std  = j.std(axis=1)
		
	plt.plot(x, j_mean)
	plt.fill_between(x, j_mean - j_std, j_mean + j_std, alpha=0.1)

	plt.xlabel('Number of Timesteps')
	plt.ylabel('Rewards')
	plt.title("Reward " + " Smoothed")
	plt.show()
