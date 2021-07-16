#!/usr/bin/env python3
# adapted wholesale from https://keras.io/examples/rl/actor_critic_cartpole/ then adapted for PyTorch

import gym
import gym_wizards
import math
import numpy as np
import time
import torch

# ENV = "CartPole-v1" # "field1d-v0" # "field2d-v0"
ENV = "field1d-v0"
DEVICE = "cpu"
SEED = 3
HIDDEN_SIZE = 48 # size of hidden layer
LEARNING_RATE = 0.005
EPISODES = 5000  # Really this is number of singleton batches
STEPS_PER_EPISODE = 30  # Does not apply to CartPole (variable)
SET_STEPS = True  # True if environment has a self.max_steps attribute and you want to set it to STEPS_PER_EPISODE
GAMMA = .9  # Discount factor for rewards


## THE NEW NEURAL NETWORK WITH CONVOLUTIONS
class ConvModel(torch.nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__(n_actions)

        self.common = torch.nn.Sequential(
                torch.nn.Conv2D(1, 1, kernel_size=3, padding=1)  # One output channel?
                torch.nn.Tanh(),
                torch.nn.MaxPool2d(2)

                torch.nn.Conv2D(1, 1, kernel_size=3, padding=1)  # One output channel?
                torch.nn.Tanh(),
                torch.nn.MaxPool2d(2)
        )
        self.action = torch.nn.Linear(CONV_OUTPUT_SIZE, n_actions) # TODO What is CONV_OUTPUT_SIZE?  Fix this.
        self.critic = torch.nn.Linear(CONV_OUTPUT_SIZE, 1)  # TODO Fix this.

    def forward(self, inputs):
        common_out = self.common(inputs.unsqueeze(dim=1))
        return torch.nn.Softmax(dim=0)(self.action(common_out)[0][0]), self.critic(common_out)

                    
## THE OLD NEURAL NETWORK
class Model(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super(Model, self).__init__()

        self.common = torch.nn.Sequential(
            torch.nn.Linear(obs_size, HIDDEN_SIZE),
            torch.nn.ReLU(),
        )
        self.action = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_SIZE, n_actions),
            # torch.nn.Softmax(dim=0),  # Softmax taken in self.forward()
        )
        self.critic = torch.nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, inputs):
        common_out = self.common(inputs.unsqueeze(dim=1))
        return torch.nn.Softmax(dim=0)(self.action(common_out)[0][0]), self.critic(common_out)


class Episode():
    def __init__(self, parent):
        self.parent = parent
        self.env = parent.env
        self.model = parent.model
        self.obs_size = parent.obs_size 
        self.n_actions = parent.n_actions
        self.action_logprobs_v_history = []
        self.critic_value_v_history = []
        self.rewards_history = []
        self.state = self.env.reset()
        self.entropies = []
        self.episode_reward = 0
        self.done = False

    def choose_action(self, state):
        state_v = torch.FloatTensor([state])  # creates a tensor of shape=[1, n_obs]: list [state] needed for shape
        state_v = state_v.to(DEVICE)
        action_probs_v, critic_value_v = self.model(state_v)
        action_probs = action_probs_v.squeeze(dim=0).data.cpu().numpy()
        entropy = -sum(action_probs * np.log2(action_probs))
        action = np.random.choice(self.n_actions, p=np.squeeze(action_probs))
        return action, entropy, action_probs_v, critic_value_v

    def choose_max_action(self, state):
        state_v = torch.FloatTensor([state]).to(DEVICE)
        action_probs_v, critic_value_v = self.model(state_v)
        max_action = torch.max(action_probs_v)
        print(action_probs_v, max_action)

    def discount_and_standardize(self):
        # store the cumulative, discounted rewards
        cumulative_discounted_rewards = []
        discounted_sum = 0
        for r in self.rewards_history[::-1]:
            discounted_sum = r + GAMMA * discounted_sum
            cumulative_discounted_rewards.insert(0, discounted_sum)
        cumulative_discounted_rewards = np.array(cumulative_discounted_rewards)
        normalized_cumulative_discounted_rewards = (cumulative_discounted_rewards - np.mean(cumulative_discounted_rewards)) / (np.std(cumulative_discounted_rewards) + 0.000001)
        return normalized_cumulative_discounted_rewards.tolist()

    def run(self):
        while not self.done:  # better to let the environment count the steps as with some problems the number can be variable 
            # Take a step using the learned policy
            self.action, entropy, self.action_probs_v, self.critic_value_v = self.choose_action(self.state)
            self.entropies.append(entropy)
            self.action_logprobs_v_history.append(torch.log(self.action_probs_v[self.action]))
            self.critic_value_v_history.append(self.critic_value_v)
            self.state, self.reward, self.done, _ = self.env.step(self.action)
            self.rewards_history.append(self.reward)
            self.episode_reward += self.reward
        self.transformed_rewards = self.discount_and_standardize()
        self.history = zip(self.action_logprobs_v_history, self.critic_value_v_history, self.transformed_rewards)
        self.final_state = self.state
        self.mean_entropy = np.array(self.entropies).mean()

    def out_str(self):
        return str(self.final_state) + " Cumulative reward: " + str(self.episode_reward) + " Mean Entropy " + str(self.mean_entropy)

class Agent():
    def __init__(self):
        # load the environment and model
        self.env = gym.make(ENV)
        if SET_STEPS:
            self.env.max_steps = STEPS_PER_EPISODE
        self.obs_size = self.env.observation_space.shape[0] # 2 # just the x and y positions
        self.n_actions = self.env.action_space.n # 9  # kings moves plus stay-in-place
        self.model = Model(self.obs_size, self.n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.huber_loss = torch.nn.modules.loss.SmoothL1Loss()  # same as torch.nn.HuberLoss() but still available

    def proc_best(self, episode_reward, n):
        if self.best is None or self.best < episode_reward:
            self.best = episode_reward
            self.best_rewards.append(self.best)
            self.best_indicies.append(n)
            return "BEST"
        else:
            return ''

    def losses(self, episode):
        actor_losses = []
        critic_losses = []
        for log_prob, critic_val, transformed_reward in episode.history:
            '''At this point in history
                    the critic estimated that we would get a # total reward = `critic_val` in the future.
                    We took an action with log probability `log_prob` 
                    and ended up recieving a total reward = `transformed_reward`.
               The actor must be updated so that it predicts an action that leads to high rewards (compared to critic's estimate) with high probability.'''
            diff = transformed_reward - critic_val
            actor_losses.append(-log_prob * diff)  # actor loss
            # The critic must be updated so that it predicts a better estimate of the future rewards.
            critic_losses.append(self.huber_loss(torch.FloatTensor(critic_val[0][0]), torch.FloatTensor([transformed_reward])))
            '''======= FROM LAPIN --- Should we add entropy loss? ============
            ent_v = -(torch.log(2*math.pi*var_v) + 1)/2  # becomes pi log pi??  Check!
            entropy_loss_v = ENTROPY_BETA * ent_v.mean()
            loss_v = loss_policy_v + entropy_loss_v + loss_value_v'''
        return actor_losses, critic_losses

    def batch(self):
        self.best = None  # self.best will be best episode over all episodes in batch
        self.log_mean_entropies = []
        self.episode_rewards = []
        self.best_rewards = []
        self.best_indicies = []
        for n in range(EPISODES):
            self.optimizer.zero_grad()

            episode = Episode(self)  # self become parent inside Episode 
            episode.run()
            self.log_mean_entropies.append(np.log10(episode.mean_entropy))
            self.episode_rewards.append(episode.episode_reward)
            best_str = self.proc_best(episode.episode_reward, n)
            print(n, episode.out_str(), best_str) 

            # Loss function
            actor_losses, critic_losses = self.losses(episode)
            overall_loss_value = sum(actor_losses) + sum(critic_losses)

            # Backpropagation
            overall_loss_value.backward()
            self.optimizer.step()

def reseed(seed=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    t = time.time()
    reseed(SEED)
    # torch.manual_seed(SEED)
    # np.random.seed(SEED)
    A = Agent()
    A.batch()
    print(time.time() - t, "Seconds")
