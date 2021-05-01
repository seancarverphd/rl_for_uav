#!/usr/bin/env python3
# adapted wholesale from https://keras.io/examples/rl/actor_critic_cartpole/
# then adapted for PyTorch

# imports
# import tensorflow_probability as tfp
# tfd = tfp.distributions
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import torch
import math
import numpy as np
import gym
import gym_wizards

# ENV = "CartPole-v1" # "field1d-v0" # "field2d-v0"
ENV = "field2d-v0"
STEPS_PER_EPISODE = 30  # Does not apply to CartPole (variable)
SET_STEPS = True  # True if environment has a self.max_steps attribute and you want to set it to STEPS_PER_EPISODE
DEVICE = "cpu"
HIDDEN_SIZE = 48 # size of hidden layer
ACTOR_LEARNING_RATE = 0.005
CRITIC_LEARNING_RATE = 0.005
EPISODES = 5000  # Really this is number of singleton batches
GAMMA = .9  # Discount factor for rewards

## CREATE THE NEURAL NETWORK

class Actor(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super(Actor, self).__init__()

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(obs_size, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, n_actions),
            # torch.nn.Softmax(dim=0),  # Softmax taken in self.forward()
        )

    def forward(self, inputs):
        actor_out = self.actor(inputs.unsqueeze(dim=1))
        return torch.nn.Softmax(dim=0)(actor_out[0][0])

class Critic(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super(Critic, self).__init__()

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(obs_size, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, inputs):
        critic_out = self.critic(inputs.unsqueeze(dim=1))
        return critic_out

# load the environment and model
env = gym.make(ENV)
if SET_STEPS:
    env.max_steps = STEPS_PER_EPISODE
obs_size = env.observation_space.shape[0] # 2 # just the x and y positions
n_actions = env.action_space.n # 9  # kings moves plus stay-in-place
actor = Actor(obs_size, n_actions)
critic = Critic(obs_size, n_actions)
actor_optimizer = torch.optim.SGD(actor.parameters(), lr=ACTOR_LEARNING_RATE)
critic_optimizer = torch.optim.SGD(critic.parameters(), lr=CRITIC_LEARNING_RATE)
huber_loss_actor = torch.nn.modules.loss.SmoothL1Loss()  # same as torch.nn.HuberLoss() but still available
huber_loss_critic = torch.nn.modules.loss.SmoothL1Loss()  # same as torch.nn.HuberLoss() but still available

action_probs_history = []
critic_value_history = []
rewards_history = []
best = None

for episode in range(EPISODES):
    state = env.reset()
    episode_reward = 0
    done = False  # TODO Uncomment for while not done
    while not done:  # better to let the environment count the steps as with some problems the number can be variable 
    # for step in range(STEPS_PER_EPISODE):
        # Take a step using the learned policy
        state_v = torch.FloatTensor([state])  # TODO comment still good?:  creates a tensor of shape=[1, n_obs]: list from input needed for shape
        state_v = state_v.to(DEVICE)
        action_probs_v, critic_value_v = model(state_v)
        action_probs = action_probs_v.squeeze(dim=0).data.cpu().numpy()  # TODO is the squeeze necessary?
        critic_value = critic_value_v.squeeze(dim=0).data.cpu().numpy()  # TODO is the squeeze necessary?
        critic_value_history.append(critic_value_v)
        action = np.random.choice(n_actions, p=np.squeeze(action_probs))  # sample action from probability distribution  # TODO squeeze necessary?
        action_probs_history.append(math.log(action_probs[action]))  # TODO indexing correct?
        state, reward, done, _ = env.step(action)
        rewards_history.append(reward)
        episode_reward += reward

    ### FINISHED ONE EPISODE NOW PROCESS THAT EPISODE
    if best is None or best < episode_reward:
        best = episode_reward
        beststr = "BEST"
    else:
        beststr = ''
    print(state, "Cumulative reward:", episode_reward, beststr) # examine the final state of each episode plus more

    # store the cumulative, discounted rewards
    cumulative_discounted_rewards = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + GAMMA * discounted_sum
        cumulative_discounted_rewards.insert(0, discounted_sum)

    # Normalize the cumulative, discounted reward history
    # TODO  Test.  I am suspicous of the next three lines.  Better, I think, to standardize batches than episodes
    cumulative_discounted_rewards = np.array(cumulative_discounted_rewards)
    normalized_cumulative_discounted_rewards = (cumulative_discounted_rewards - np.mean(cumulative_discounted_rewards)) / (np.std(cumulative_discounted_rewards) + 0.000001)
    normalized_cumulative_discounted_rewards = normalized_cumulative_discounted_rewards.tolist()

    history = zip(action_probs_history, critic_value_history, normalized_cumulative_discounted_rewards)
    actor_losses = []
    critic_losses = []
    actor_optimizer.zero_grad()  # TODO Is this where it needs to be?
    critic_optimizer.zero_grad()  # TODO Is this where it needs to be?
    for log_prob, critic_val, normed_cum_disc_rew in history:
        # At this point in history, the critic estimated that we would get a
        # total reward = `value` in the future. We took an action with log probability
        # of `log_prob` and ended up recieving a total reward = `ret`.
        # The actor must be updated so that it predicts an action that leads to
        # high rewards (compared to critic's estimate) with high probability.
        diff = normed_cum_disc_rew - critic_val
        actor_losses.append(-log_prob * diff)  # actor loss

        # The critic must be updated so that it predicts a better estimate of the future rewards.
        critic_losses.append(huber_loss(critic_val[0][0], torch.FloatTensor([normed_cum_disc_rew])))
        '''======= OLD ============
        optimizer.zero_grad()
        mu_v, var_v, value_v = net(states_v)
        loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
        adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
        log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
        loss_policy_v = -log_prob_v.mean()
        ent_v = -(torch.log(2*math.pi*var_v) + 1)/2
        entropy_loss_v = ENTROPY_BETA * ent_v.mean()
        loss_v = loss_policy_v + entropy_loss_v + loss_value_v
        loss_v.backward()
        optimizer.step()'''

    # Backpropagation
    # overall_loss_value = sum(actor_losses) + sum(critic_losses)
    sum_actor_losses = sum(actor_losses)
    sum_critic_losses = sum(critic_losses)
    # overall_loss_value.backward()
    sum_actor_losses.backward()
    sum_critic_losses.backward()
    # optimizer.step()
    actor_optimizer.step()
    critic_optimizer.step()

    # Reset env and clear the loss and reward history for next episode
    action_probs_history.clear()
    critic_value_history.clear()
    rewards_history.clear()
