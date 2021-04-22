import numpy as np
import gym

class Field1D(gym.Env):  # To use Gym inherit from gym.Env
    def __init__(self):
        self.opt = 5.
        self.peak = 10.
        self.left_bound = -20
        self.right_bound = 20
        self.observation_space = gym.spaces.Box(low=np.array((self.left_bound,), dtype=np.float32), high=np.array((self.right_bound,), dtype=np.float32))  # [-20,20]
        self.action_space = gym.spaces.Discrete(3)  # left, stay, right
        self.reset()

    def reset(self):
        self.x = 0
        self.n = 0  # Count number of steps
        return [self.x]

    def step(self, action):
        action -= 1
        assert action == -1 or action == 0 or action == 1
        self.x += action
        if self.x < self.left_bound:
            self.x = self.left_bound
        if self.x > self.right_bound:
            self.x = self.right_bound
        self.n += 1
        obs = [self.x]
        reward = -(self.x - self.opt)**2 + self.peak
        done = (self.n >= 15)
        info = {}
        return obs, reward, done, info

    def render(self):
        print(str(self.x) + " approaching " + str(self.opt))

