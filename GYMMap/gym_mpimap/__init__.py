from gym.envs.registration import register

register(
    id='MPIMap-v0',
    entry_point='gym_mpimap.envs:MPIMap',
)
