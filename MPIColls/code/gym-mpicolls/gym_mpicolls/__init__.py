from gym.envs.registration import register

register(id='MPIColls-v0',
        entry_point='gym_mpicolls.envs:MPICollsEnv',
)
