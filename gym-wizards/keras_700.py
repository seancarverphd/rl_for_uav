import gym
from collections import deque
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import gym
class Field2D(gym.Env):  # To use Gym inherit from gym.Env
    def __init__(self):
        self.opt_x = 5.
        self.opt_y = 7.
        self.peak = -2.
        self.max_steps = 30
        self.left_bound = -20
        self.right_bound = 20
        self.lower_bound = -20
        self.upper_bound = 20
        self.observation_space = gym.spaces.Box(low=np.array((self.left_bound,self.lower_bound), dtype=np.float32), high=np.array((self.right_bound,self.upper_bound), dtype=np.float32))
        self.action_space = gym.spaces.Discrete(9)  # king's moves plus stay
        self.reset()
    def reset(self):
        self.x = 0
        self.y = 0
        self.n = 0  # Count number of steps
        return [self.x, self.y]
    def convert_action_2d(self, action):
        #assert isinstance(action, int)
        assert action >= 0
        assert action <= 8
        action_x = action % 3 - 1
        action_y = int(action / 3) - 1
        assert action_x == -1 or action_x == 0 or action_x == 1
        assert action_y == -1 or action_y == 0 or action_y == 1
        return action_x, action_y
    def reward_func(self):
        #return -(self.x - self.opt_x)**2 -(self.y - self.opt_y)**2 + self.peak
        # more peaked
        dx = np.abs(self.x-self.opt_x)
        dy = np.abs(self.y-self.opt_y)
        return (-3*dx) + (-3*dy)
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
        obs = [self.x, self.y]
        reward = self.reward_func()
        done = (self.n >= self.max_steps)
        info = {}
        return obs, reward, done, info
    def render(self):
        print("["+ str(self.x) + ", "+ str(self.y) + "] approaching [" + str(self.opt_x) + ", " + str(self.opt_y) + "]")
class Critic:
    def __init__(self, sess, action_dim, observation_dim):
        self.action_dim, self.observation_dim = action_dim, observation_dim
        # setting our created session as default session
        K.set_session(sess)
        self.model = self.create_model()
    def create_model(self):
        state_input = Input(shape=self.observation_dim)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(24, activation='relu')(state_h1)
        state_h3 = Dense(24, activation='relu')(state_h2)
        state_h4 = Dense(24, activation='relu')(state_h3)
        output = Dense(1, activation='linear')(state_h4)
        model = Model(inputs=state_input, outputs=output)
        model.compile(loss="mse", optimizer=Adam(lr=0.05))
        return model
class Actor:
    def __init__(self, sess, action_dim, observation_dim):
        self.action_dim, self.observation_dim = action_dim, observation_dim
        # setting the our created session as default session
        K.set_session(sess)
        self.sess = sess
        self.state_input, self.output, self.model = self.create_model()
        # Implementing
        # grad(J(actor_weights)) = sum_(t=1, T-1)[ grad(log(pi(at | st, actor_weights)) * Advantaged(st, at), actor_weights) ]
        # Placeholder for advantage values.
        self.advantages = tf.placeholder(tf.float32, shape=[None, action_dim])
        model_weights = self.model.trainable_weights
        # Adding small number inside log to avoid log(0) = -infinity
        log_prob = tf.math.log(self.output + 10e-10)
        # Multiply log by -1 to convert the optimization problem as minimization problem.
        # This step is essential because apply_gradients always do minimization.
        neg_log_prob = tf.multiply(log_prob, -1)
        # Calulate and update the weights of the model to optimize the actor
        actor_gradients = tf.gradients(neg_log_prob, model_weights, self.advantages)
        grads = zip(actor_gradients, model_weights)
        self.optimize = tf.train.AdamOptimizer(0.001).apply_gradients(grads)
    def create_model(self):
        state_input = Input(shape=self.observation_dim)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(24, activation='relu')(state_h1)
        output = Dense(self.action_dim, activation='softmax')(state_h2)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=adam)
        return state_input, output, model
    def train(self, X, y):
        self.sess.run(self.optimize, feed_dict={self.state_input:X, self.advantages:y})
# Hyperparameters
EPISODES = 1500
REPLAY_MEMORY_SIZE = 40000
MINIMUM_REPLAY_MEMORY = 10000
DISCOUNT = 0.9
EPSILON = 1
EPSILON_DECAY = 0.9999
MINIMUM_EPSILON = 0.03
MINIBATCH_SIZE = 32
VISUALIZATION = False
# Environment details
#env = gym.make('CartPole-v1').unwrapped
#action_dim = env.action_space.n
#observation_dim = env.observation_space.shape
env = Field2D()
action_dim = 9
observation_dim=2
# creating own session to use across all the Keras/Tensorflow models we are using
sess = tf.Session()
# sess = tf.compat.v1.Session() # for backward compatibility with TF 1
# Experience replay memory for stable learning
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
# Actor model to take actions
# state -> action
actor = Actor(sess, action_dim, observation_dim)
# Critic model to evaluate the acion taken by the actor
# state -> value of state V(s_t)
critic = Critic(sess, action_dim, observation_dim)
sess.run(tf.initialize_all_variables())
def train_advantage_actor_critic(replay_memory, actor, critic):
    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)
    X = []
    y = []
    advantages = np.zeros(shape=(MINIBATCH_SIZE, action_dim))
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state, done = sample
        if done:
            # If last state then advatage A(s, a) = reward_t - V(s_t)
            advantages[index][action] = reward - critic.model.predict(np.expand_dims(cur_state, axis=0))[0][0]
        else:
            # If not last state the advantage A(s_t, a_t) = reward_t + DISCOUNT * V(s_(t+1)) - V(s_t)
            next_reward = critic.model.predict(np.expand_dims(next_state, axis=0))[0][0]
            advantages[index][action] = reward + DISCOUNT * next_reward - \
                                        critic.model.predict(np.expand_dims(cur_state, axis=0))[0][0]
            # Updating reward to trian state value fuction V(s_t)
            reward = reward + DISCOUNT * next_reward
        X.append(cur_state)
        y.append(reward)
    X = np.array(X)
    y = np.array(y)
    y = np.expand_dims(y, axis=1)
    # Training Actor and Critic
    actor.train(X, advantages)
    critic.model.fit(X, y, batch_size=MINIBATCH_SIZE, verbose=0)
max_reward = -1000
for episode in range(EPISODES):
    cur_state = env.reset()
    done = False
    episode_reward = 0
    while not done and episode_reward < 1000:
        if VISUALIZATION:
            env.render()
        action = np.zeros(shape=(action_dim))
        if (np.random.uniform(0, 1) < EPSILON):
            # Taking random actions (Exploration)
            action[np.random.randint(0, action_dim)] = 1
        else:
            # Taking optimal action suggested by the actor (Exploitation)
            action = actor.model.predict(np.expand_dims(cur_state, axis=0))
            #action = action[0]
        # q-style
        action_taken = np.argmax(action)
        # smaple-style
        #action_taken = np.random.choice(list(range(9)), p=action)
        next_state, reward, done, _ = env.step(action_taken)
        episode_reward += reward
        if done:
            # Episode ends means we have lost the game. So, we are giving large negative reward.
            reward = 1
        # Recording experience to train the actor and critic
        replay_memory.append((cur_state, action_taken, reward, next_state, done))
        cur_state = next_state
        if len(replay_memory) < MINIMUM_REPLAY_MEMORY:
            continue
        # Training actor and critic
        train_advantage_actor_critic(replay_memory, actor, critic)
        # Decreasing the exploration probability
        if EPSILON > MINIMUM_EPSILON and len(replay_memory) >= MINIMUM_REPLAY_MEMORY:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(EPSILON, MINIMUM_EPSILON)
    # some bookkeeping
    #if (episode_reward > 400 and episode_reward > max_reward):
    #    actor.model.save_weights(str(episode_reward) + ".h5")
    max_reward = max(max_reward, episode_reward)
    print('Episodes:', episode, 'Episodic_Reweard:', episode_reward, 'Max_Reward_Achieved:', max_reward, 'EPSILON:',
          EPSILON)
## what's the policy look like now??
cur_state = env.reset()
for i in range(50):
    print(cur_state)
    action = actor.model.predict(np.expand_dims(cur_state, axis=0))
    action_taken = np.argmax(action)
    next_state, reward, done, _ = env.step(action_taken)
    cur_state = next_state

