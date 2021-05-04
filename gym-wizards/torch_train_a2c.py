#!/usr/bin/env python3
# adapted wholesale from https://keras.io/examples/rl/actor_critic_cartpole/ then adapted for PyTorch

import torch
import math
import numpy as np
import gym
import gym_wizards

# ENV = "CartPole-v1" # "field1d-v0" # "field2d-v0"
ENV = "field1d-v0"
DEVICE = "cpu"
HIDDEN_SIZE = 48 # size of hidden layer
LEARNING_RATE = 0.005
EPISODES = 500  # Really this is number of singleton batches
STEPS_PER_EPISODE = 30  # Does not apply to CartPole (variable)
SET_STEPS = True  # True if environment has a self.max_steps attribute and you want to set it to STEPS_PER_EPISODE
GAMMA = .9  # Discount factor for rewards


## CREATE THE NEURAL NETWORK
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

    def choose_action(self, state):
        state_v = torch.FloatTensor([state])  # TODO comment still good?:  creates a tensor of shape=[1, n_obs]: list from input needed for shape
        state_v = state_v.to(DEVICE)
        action_probs_v, critic_value_v = self.model(state_v)
        action_probs = action_probs_v.squeeze(dim=0).data.cpu().numpy()
        critic_value = critic_value_v.squeeze(dim=0).data.cpu().numpy()
        action = np.random.choice(self.n_actions, p=np.squeeze(action_probs))
        return action, action_probs, critic_value, action_probs_v, critic_value_v

    def episode(self):
        action_logprobs_history = []
        action_logprobs_v_history = []
        critic_value_history = []
        critic_value_v_history = []
        rewards_history = []
        state = self.env.reset()
        episode_reward = 0
        done = False  # TODO Uncomment for while not done
        while not done:  # better to let the environment count the steps as with some problems the number can be variable 
            # Take a step using the learned policy
            action, action_probs, critic_value, action_probs_v, critic_value_v = self.choose_action(state)
            action_logprobs_history.append(math.log(action_probs[action]))  # TODO indexing correct?
            action_logprobs_v_history.append(torch.log(action_probs_v[action]))
            critic_value_history.append(critic_value)
            critic_value_v_history.append(critic_value_v)
            state, reward, done, _ = self.env.step(action)
            rewards_history.append(reward)
            episode_reward += reward
        return state, episode_reward, rewards_history, action_logprobs_history, critic_value_history, critic_value_v_history


    def proc_best(self, episode_reward):
        if self.best is None or self.best < episode_reward:
            self.best = episode_reward
            return "BEST"
        else:
            return ''

    def print_episode_stats(self, final_state, episode_reward, best_episode_so_far_str):
        print(final_state, "Cumulative reward:", episode_reward, best_episode_so_far_str) # examine the final state of each episode plus more

    def discount_and_standardize(self, rewards_history):
        # store the cumulative, discounted rewards
        cumulative_discounted_rewards = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + GAMMA * discounted_sum
            cumulative_discounted_rewards.insert(0, discounted_sum)
        cumulative_discounted_rewards = np.array(cumulative_discounted_rewards)
        normalized_cumulative_discounted_rewards = (cumulative_discounted_rewards - np.mean(cumulative_discounted_rewards)) / (np.std(cumulative_discounted_rewards) + 0.000001)
        return normalized_cumulative_discounted_rewards.tolist()

    def batch(self):
        self.best = None  # self.best will be best episode over all episodes in batch
        for _ in range(EPISODES):
            self.optimizer.zero_grad()  # TODO Is this where it needs to be?

            final_state, episode_reward, rewards_history, action_logprobs_history, critic_value_history, critic_value_v_history = self.episode()

            ### FINISHED ONE EPISODE NOW PROCESS THAT EPISODE
            best_episode_so_far_str = self.proc_best(episode_reward)
            self.print_episode_stats(final_state, episode_reward, best_episode_so_far_str)
            transformed_rewards = self.discount_and_standardize(rewards_history)

            # Normalize the cumulative, discounted reward history

            history = zip(action_logprobs_history, critic_value_v_history, transformed_rewards)
            actor_losses = []
            critic_losses = []
            for log_prob, critic_val, transformed_reward in history:
                '''At this point in history
                        the critic estimated that we would get a # total reward = `critic_val` in the future.
                        We took an action with log probability `log_prob` 
                        and ended up recieving a total reward = `transformed_reward`.
                   The actor must be updated so that it predicts an action that leads to high rewards (compared to critic's estimate) with high probability.'''
                diff = transformed_reward - critic_val
                actor_losses.append(-log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of the future rewards.
                critic_losses.append(self.huber_loss(torch.FloatTensor(critic_val[0][0]), torch.FloatTensor([transformed_reward])))

                '''======= FROM LAPIN ============
                ent_v = -(torch.log(2*math.pi*var_v) + 1)/2
                entropy_loss_v = ENTROPY_BETA * ent_v.mean()
                loss_v = loss_policy_v + entropy_loss_v + loss_value_v'''

            # Backpropagation
            overall_loss_value = sum(actor_losses) + sum(critic_losses)
            overall_loss_value.backward()
            self.optimizer.step()
            ### NOW DO ANOTHER EPISODE

if __name__ == '__main__':
    A = Agent()
    A.batch()

