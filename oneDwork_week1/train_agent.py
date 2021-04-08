# adapted wholesale from https://keras.io/examples/rl/actor_critic_cartpole/


import tensorflow_probability as tfp
tfd = tfp.distributions
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# import the environment
from oneDwork_week1.field1d import Field1D
env = Field1D()

try:
    del model
except:
    pass


# set up the model
# shared layers

num_hidden = 48 # size of hidden layer
num_inputs = 1 # just the x position
num_actions = 3  # left, stay in same place, right

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax", name="action")(common)
critic = layers.Dense(1,name="critic")(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.005)
huber_loss = keras.losses.Huber()


EPISODES = 500
STEPS_PER_EPISODE = 30
gamma = .9  # Discount factor for rewards

action_probs_history = []
critic_value_history = []
rewards_history = []

for episode in range(EPISODES):
    #print(episode)
    state = env.reset()
    episode_reward = 0
    #mean_reward = -5

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

            # re-map action from 0,1,2 to -1,0,1, as per environment's requirement
            movement = action - 1

            state, reward, done, _ = env.step(movement)

            rewards_history.append(reward)
            episode_reward += reward

        print(state)

        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 0.000001)
        returns = returns.tolist()

        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

