# Gym MPIColls
--------------

Gym MPIColls environment for OpenAI Gym.

## What is?

It is an environment to perform MPI collective optimization using
Reinforcement Learning techniques.

## Installation
1. Install [OpenAi Gym](https://github.com/openai/gym)
```bash
pip install gym
```

2. Download and install `gym-mpicolls`
```bash
git clone https://github.com/https://jaricogallego/gym-mpicolls
cd gym-mpicolls
python setup.py install
```

## Running
Start by importing the package and initializing the environment
```python
import gym
import gym-mpicolls
env = gym.make('MPIColls-v0')  
```

# Reset the env before starting
state = env.reset()

while not done:
    # TODO: env.render(mode=None)
    state, reward, done, infos = env.step(env.action_space.sample(), -1)

. . .
