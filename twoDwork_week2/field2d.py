import numpy as np

class Field2D():  # To use Gym inherit from gym.Env
    def __init__(self):
        self.opt_x = 5.
        self.opt_y = 7.
        self.peak = 10.
        self.left_bound = -20
        self.right_bound = 20
        # Objects to define if using gym to pass to solver
        #    self.observation_space
        #    self.action_space
        self.reset()

    def reset(self):
        self.x = 0
        self.y = 0
        self.n = 0  # Count number of steps
        return [self.x, self.y]

    def convert_action_2d(self, action):
        assert isinstance(action, int)
        assert action >= 0
        assert action <= 8
        action_x = action % 3 - 1
        action_y = int(action / 3) - 1
        assert action_x == -1 or action_x == 0 or action_x == 1
        assert action_y == -1 or action_y == 0 or action_y == 1
        return action_x, action_y

    def reward_func(self):
        return -(self.x - self.opt_x)**2 -(self.y - self.opt_y)**2 + self.peak

    def step(self, action):
        action_x, action_y = self.convert_action_2d(action)
        self.x += action_x
        self.y += action_y
        if self.x < self.left_bound:
            self.x = self.left_bound
        if self.x > self.right_bound:
            self.x = self.right_bound
        if self.y < self.left_bound:
            self.y = self.left_bound
        if self.y > self.right_bound:
            self.y = self.right_bound
        self.n += 1
        obs = [self.x, self.y]
        reward = self.reward_func()
        done = (self.n >= 15)
        info = {}
        return obs, reward, done, info

    def render(self):
        print("["+ str(self.x) + ", "+ str(self.y) + "] approaching [" + str(self.opt_x) + ", " + str(self.opt_y) + "]")

