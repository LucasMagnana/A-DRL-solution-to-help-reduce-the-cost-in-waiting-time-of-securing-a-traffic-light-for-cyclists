import numpy as np

import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from NeuralNetworks import PPO_Model


def discount_rewards(rewards, gamma):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def gae(rewards, values, episode_ends, gamma, lam):
    """Compute generalized advantage estimate.
        rewards: a list of rewards at each step.
        values: the value estimate of the state at each step.
        episode_ends: an array of the same shape as rewards, with a 1 if the
            episode ended at that step and a 0 otherwise.
        gamma: the discount factor.
        lam: the GAE lambda parameter.
    """

    N = rewards.shape[0]
    T = rewards.shape[1]
    gae_step = np.zeros((N, ))
    advantages = np.zeros((N, T))
    for t in reversed(range(T - 1)):
        # First compute delta, which is the one-step TD error
        delta = rewards[:, t] + gamma * values[:, t + 1] * episode_ends[:, t] - values[:, t]
        # Then compute the current step's GAE by discounting the previous step
        # of GAE, resetting it to zero if the episode ended, and adding this
        # step's delta
        gae_step = delta + gamma * lam * episode_ends[:, t] * gae_step
        # And store it
        advantages[:, t] = gae_step
    return advantages



class PPOHyperParams :
    def __init__(self):
        self.LR = 0.01
        self.GAMMA = 0.99
        self.LAMBDA = 0.99
        self.EPSILON = 0.2

        self.EPISODE_COUNT = 30
        self.LEARNING_EP = 5
        self.K = 4

        self.COEFF_CRITIC_LOSS = 0.5
        self.COEFF_ENTROPY_LOSS = 0.01


class PPOAgent():

    def __init__(self, ob_space, ac_space, model_to_load):

        self.hyperParams = PPOHyperParams()

        self.model = PPO_Model(ob_space, ac_space)

        if(model_to_load != None):
            self.model.load_state_dict(torch.load(model_to_load))
            self.model.eval()

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperParams.LR)
        self.mse = torch.nn.MSELoss()

        self.ac_space = ac_space

        self.reset_batches()


    def reset_batches(self):
        self.batch_rewards = []
        self.batch_advantages = []
        self.batch_states = []
        self.batch_values = []
        self.batch_actions = []
        self.batch_selected_probs = []
        self.batch_done = []


    def learn(self):

        for k in range(self.hyperParams.K):
            self.optimizer.zero_grad()
            
            state_tensor = torch.tensor(self.batch_states)

            #print(state_tensor.tolist() == self.batch_states)

            advantages_tensor = torch.tensor(self.batch_advantages)
            old_selected_probs_tensor = torch.tensor(self.batch_selected_probs)

            old_values_tensor = torch.tensor(self.batch_values)
            rewards_tensor = torch.tensor(self.batch_rewards, requires_grad = True)
            rewards_tensor = rewards_tensor.float()
            # Actions are used as indices, must be 
            # LongTensor
            action_tensor = torch.LongTensor(self.batch_actions)
            action_tensor = action_tensor.long()

            
            # Calculate actor loss
            probs, values_tensor = self.model(state_tensor)
            selected_probs_tensor = torch.index_select(probs, 1, action_tensor).diag()
            values_tensor = values_tensor.flatten()

            loss = selected_probs_tensor/old_selected_probs_tensor*advantages_tensor
            clipped_loss = torch.clamp(selected_probs_tensor/old_selected_probs_tensor, 1-self.hyperParams.EPSILON, 1+self.hyperParams.EPSILON)*advantages_tensor

            loss_actor = -torch.min(loss, clipped_loss).mean()

            # Calculate critic loss
            value_pred_clipped = old_values_tensor + (values_tensor - old_values_tensor).clamp(-self.hyperParams.EPSILON, self.hyperParams.EPSILON)
            value_losses = (values_tensor - rewards_tensor) ** 2
            value_losses_clipped = (value_pred_clipped - rewards_tensor) ** 2

            loss_critic = 0.5 * torch.max(value_losses, value_losses_clipped)
            loss_critic = loss_critic.mean() 

            entropy_loss = torch.mean(torch.distributions.Categorical(probs = probs).entropy())

            loss = loss_actor + self.hyperParams.COEFF_CRITIC_LOSS * loss_critic -  self.hyperParams.COEFF_ENTROPY_LOSS * entropy_loss


            # Calculate gradients
            loss.backward()
            # Apply gradients
            self.optimizer.step()

        self.reset_batches()


    def memorize(self, ob_prec, val, action_probs, action, ob, reward, done):
        self.states.append(ob_prec)
        self.values.extend(val)
        self.rewards.append(reward)
        self.actions.append(action)
        self.selected_probs.append(action_probs[action])
        self.list_done.append(done)  

    def start_episode(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.values = []
        self.selected_probs = []
        self.list_done = []

    def end_episode(self):
        self.list_done[-1] = True
        self.batch_rewards.extend(discount_rewards(self.rewards, self.hyperParams.GAMMA))
        gaes = gae(np.expand_dims(np.array(self.rewards), 0), np.expand_dims(np.array(self.values), 0), np.expand_dims(np.array([not elem for elem in self.list_done]), 0), self.hyperParams.GAMMA, self.hyperParams.LAMBDA)
        self.batch_advantages.extend(gaes[0])
        self.batch_states.extend(self.states)
        self.batch_values.extend(self.values)
        self.batch_actions.extend(self.actions)
        self.batch_done.extend(self.list_done)
        self.batch_selected_probs.extend(self.selected_probs)


    def act(self, observation):
        # Get actions and convert to numpy array
        action_probs, val = self.model(torch.tensor(observation))
        action_probs = action_probs.detach().numpy()
        val = val.detach().numpy()
        action = np.random.choice(np.arange(self.ac_space), p=action_probs)

        return action, val, action_probs