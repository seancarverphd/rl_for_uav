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

ENV = "field2d-v0"
HIDDEN_SIZE = 48 # size of hidden layer
LEARNING_RATE = 0.005
EPISODES = 5000
STEPS_PER_EPISODE = 30
GAMMA = .9  # Discount factor for rewards

## CREATE THE NEURAL NETWORK

class Model(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(Model, self).__init__()

        self.common = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )
        self.action = nn.Sequential(
            nn.Linear(HID_SIZE, n_actions),
            nn.Softmax(dim=1),  # TODO Put in dimension to Softmax(dim=0 or dim=1)?
        )
        self.critic = nn.Linear(HID_SIZE, 1)

    def forward(self, inputs):
        common_out = self.common(inputs.unsqueeze(dim=1))  # TODO is unsqueeze needed`
        return self.action(common_out), self.value(common_out)

# inputs = layers.Input(shape=(num_inputs,))
# common = layers.Dense(HIDDEN_SIZE, activation="relu")(inputs)
# action = layers.Dense(num_actions, activation="softmax", name="action")(common)
# critic = layers.Dense(1,name="critic")(common)

# load the environment and model
env = gym.make(ENV)
obs_size = env.observation_space.shape[0] # 2 # just the x and y positions
n_actions = env.action_space.n # 9  # kings moves plus stay-in-place
model = Model(obs_size, n_actions)
# model = keras.Model(inputs=inputs, outputs=[action, critic])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = keras.optimizers.Adam(learning_rate=0.005)
huber_loss = torch.nn.HuberLoss()
# huber_loss = keras.losses.Huber()

action_probs_history = []
critic_value_history = []
rewards_history = []
best = None

for episode in range(EPISODES):
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape(persistent=False) as tape:
        for step in range(STEPS_PER_EPISODE):

            # Take a step using the learned policy
            curr_st_array = tf.convert_to_tensor(state)
            curr_st_array = tf.expand_dims(curr_st_array, 0)
            action_probs, critic_value = model(curr_st_array)
            critic_value_history.append(critic_value[0, 0])

            # sample action from probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            state, reward, done, _ = env.step(action)

            rewards_history.append(reward)
            episode_reward += reward

        if best is None or best < episode_reward:
            best = episode_reward
            beststr = "BEST"
        else:
            beststr = ''

        print(state, "Cumulative reward:", episode_reward, beststr) # casually examine the final state of each episode, should approach:  [5, 5]

        # store the cumulative, discounted rewards
        cumulative_discounted_rewards = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + GAMMA * discounted_sum
            cumulative_discounted_rewards.insert(0, discounted_sum)

        # Normalize the cumulative, discounted reward history
        cumulative_discounted_rewards = np.array(cumulative_discounted_rewards)
        normalized_cumulative_discounted_rewards = (cumulative_discounted_rewards - np.mean(cumulative_discounted_rewards)) / (np.std(cumulative_discounted_rewards) + 0.000001)
        normalized_cumulative_discounted_rewards = normalized_cumulative_discounted_rewards.tolist()

        history = zip(action_probs_history, critic_value_history, normalized_cumulative_discounted_rewards)
        actor_losses = []
        critic_losses = []
        for log_prob, critic_val, normed_cum_disc_rew in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = normed_cum_disc_rew - critic_val
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(critic_val, 0), tf.expand_dims(normed_cum_disc_rew, 0))
            )

        # Backpropagation
        overall_loss_value = sum(actor_losses) + sum(critic_losses)
        gradientss = tape.gradient(overall_loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradientss, model.trainable_variables))

        # Clear the loss and reward history for next episode
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

