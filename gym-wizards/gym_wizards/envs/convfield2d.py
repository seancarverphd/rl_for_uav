import numpy as np
import gym

class Grid():
    def __init__(self):
        self.dx = 1
        self.dy = 1
        self.nx = 10
        self.ny = 8
        self.clear_grid()
        
    def clear_grid(self):
        self.grid = np.zeros([self.ny, self.nx])
        
    def add_xy(self, x, y):
        x1 = int(round(min(x/self.dx, self.nx-1)))
        y1 = int(round(min(y/self.dy, self.ny-1)))
        self.grid[y1,x1] += 1

class Field2D(gym.Env):  # To use Gym inherit from gym.Env
    def __init__(self):
        self.opt_x = 5.
        self.opt_y = 7.
        self.peak = 10.
        self.max_steps = 30
        self.left_bound = -20
        self.right_bound = 20
        self.lower_bound = -20
        self.upper_bound = 20
        self.observation_space = gym.spaces.Box(low=np.array((self.left_bound,self.lower_bound), dtype=np.float32), high=np.array((self.right_bound,self.upper_bound), dtype=np.float32))
        self.action_space = gym.spaces.Discrete(9)  # king's moves plus stay
        self.comms = Grid()
        self.reset()

    def reset(self):
        self.x = 0
        self.y = 0
        self.comms.clear_grid()
        self.comms.add_xy(self.x,self.y)
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
        if self.y < self.lower_bound:
            self.y = self.lower_bound
        if self.y > self.upper_bound:
            self.y = self.upper_bound
        self.n += 1
        self.comms.clear_grid()
        self.comms.add_xy(self.x,self.y)
        obs = self.comms.grid 
        reward = self.reward_func()
        done = (self.n >= self.max_steps)
        info = {}
        return obs, reward, done, info

    def render(self):
        print("Approaching [" + str(self.opt_x) + ", " + str(self.opt_y) + "]")
        print(self.comms.grid)
