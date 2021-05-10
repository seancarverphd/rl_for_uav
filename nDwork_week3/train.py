#!/usr/bin/env python3
import gym
import gym_wizards
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()  # should create a flat numpy.array shape=[n_obs]
    softmax = nn.Softmax(dim=1)  # Returns a function
    while True:
        obs_v = torch.FloatTensor([obs])  # creates a tensor of shape=[1, n_obs]: list from input needed for shape
        act_probs_v = softmax(net(obs_v))  # tensor shape=[1,n_act]: softmax makes them probs of actions
        act_probs = act_probs_v.data.numpy()[0]  # flat numpy.array shape=[n_act]: selecting [0] flattens from 2D-array
        action = np.random.choice(len(act_probs), p=act_probs)  # selects one action from discrete distribution
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))  # gets a list of rewards for all episodes in batch
    reward_bound = np.percentile(rewards, percentile)  # computes pth percentile of rewards (p adjustable; eg 70%)
    reward_mean = float(np.mean(rewards))  # computes mean of rewards 

    train_obs = []
    train_act = []
    for reward, steps in batch:  # batch is a list of episodes each a named tuple (undiscounted-reward, steps (list-EpisodeStep))
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))  # Extend list of all obs with reward above %-ile for this step
        train_act.extend(map(lambda step: step.action, steps))  # Extend list of all obs with reward above %-ile for this step
    # ^^^ Each obs is numpy array shape (n_obs,); train_obs is a list of these; likewise for act

    train_obs_v = torch.FloatTensor(train_obs)  # tensor shape [n_steps_above_threshold, n_obs]
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    # env = gym.make("CartPole-v0")
    env = gym.make("field1d-v0")

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()  # sets gradients to zero; otherwise accumulates from previous calculations
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()
