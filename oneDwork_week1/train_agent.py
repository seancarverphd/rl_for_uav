# adapted wholesale from https://keras.io/examples/rl/actor_critic_cartpole/

# imports
import tensorflow_probability as tfp
tfd = tfp.distributions
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# import the environment
from field1d import Field1D  # from oneDwork_week1.field1d import Field1D
env = Field1D()

# cleanup: delete the TF model if it is hanging around
try:
    del model
except:
    pass


# set up the model
num_hidden = 48 # size of hidden layer
num_inputs = 1 # just the x position
num_actions = 3  # left, stay in same place, right

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax", name="action")(common)
critic = layers.Dense(1,name="critic")(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.0005)
huber_loss = keras.losses.Huber()


EPISODES = 5000
STEPS_PER_EPISODE = 30
gamma = .8  # Discount factor for rewards

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

            # re-map action from 0,1,2 to -1,0,1, as per environment's requirement
            movement = action - 1

            state, reward, done, _ = env.step(movement)

            rewards_history.append(reward)
            episode_reward += reward

        #print(state) # casually examine the final state of each episode, should approach 5
        if episode % 50 == 0:
            env.render()

        # store the cumulative, discounted rewards
        cumulative_discounted_rewards = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            cumulative_discounted_rewards.insert(0, discounted_sum)

        # Normalize the cumulative, discounted reward history
        cumulative_discounted_rewards = np.array(cumulative_discounted_rewards)
        normalized_cumulative_discounted_rewards = (cumulative_discounted_rewards - np.mean(cumulative_discounted_rewards)) / (np.std(cumulative_discounted_rewards) + 0.000001)
        normalized_cumulative_discounted_rewards = normalized_cumulative_discounted_rewards.tolist()

        #cumulative_discounted_rewards = np.array(cumulative_discounted_rewards)
        #normalized_cumulative_discounted_rewards = cumulative_discounted_rewards / 100.0
        #normalized_cumulative_discounted_rewards = normalized_cumulative_discounted_rewards.tolist()

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
        gradients = tape.gradient(overall_loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Clear the loss and reward history for next episode
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

