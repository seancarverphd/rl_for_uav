from gym.envs.registration import register

register(
        id = 'field1d-v0',
        entry_point='gym_wizards.envs:Field1D',
        )
register(
        id = 'field2d-v0',
        entry_point='gym_wizards.envs:Field2D',
        )
## Add another register(...) for another environment
