import numpy as np

import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from NeuralNetworks import PPO_Model


def discount_rewards(rewards, list_done, gamma):
    r = []
    for reward, done in zip(reversed(rewards), reversed(list_done)):
        if done:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        r.insert(0, discounted_reward)
    r = np.array(r, dtype=np.single)
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
        self.LR = 0.001
        self.GAMMA = 0.99
        self.LAMBDA = 0.99
        self.EPSILON = 0.2

        self.EPISODE_COUNT = 30
        self.EP_LEARNING_START = 1
        self.LEARNING_EP = 5
        self.K = 4
        self.DECISION_COUNT = 750000

        self.COEFF_CRITIC_LOSS = 0.5
        self.COEFF_ENTROPY_LOSS = 0.01


class PPOAgent():

    def __init__(self, ob_space, ac_space, model_to_load, continuous_action_space=False):

        self.hyperParams = PPOHyperParams()

        self.continuous_action_space = continuous_action_space

        if(self.continuous_action_space):           
            self.model = PPO_Model(ob_space, ac_space, max_action=60)
            self.old_model = PPO_Model(ob_space, ac_space, max_action=60)
        else:
            self.model = PPO_Model(ob_space, ac_space)
            self.old_model = PPO_Model(ob_space, ac_space)

        if(model_to_load != None):
            self.model.load_state_dict(torch.load(model_to_load))
            self.model.eval()

        self.old_model.load_state_dict(self.model.state_dict())

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperParams.LR)
        self.mse = torch.nn.MSELoss()

        self.ac_space = ac_space

        

        if(self.continuous_action_space):
            self.action_std = torch.full((ac_space,), 1/120)

        self.num_decisions_made = 0

        self.tab_losses = []

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

            state_tensor = torch.tensor(self.batch_states)

            #print(state_tensor.tolist() == self.batch_states)

            advantages_tensor = torch.tensor(self.batch_advantages)
            old_selected_probs_tensor = torch.tensor(self.batch_selected_probs)

            old_values_tensor = torch.tensor(self.batch_values)

            rewards_tensor = torch.tensor(self.batch_rewards)
            # Normalizing the rewards:
            #rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-5)
            

            action_tensor = torch.tensor(self.batch_actions)

            
            # Calculate actor loss
            probs, values_tensor = self.model(state_tensor)
            values_tensor = values_tensor.flatten()

            if(self.continuous_action_space):
                probs=probs.flatten()
                action_std = self.action_std.expand_as(probs)
                mat_std = torch.diag_embed(action_std)
                dist = MultivariateNormal(probs, mat_std)
                selected_probs_tensor = dist.log_prob(action_tensor.flatten())
                ratios = torch.exp(selected_probs_tensor - old_selected_probs_tensor.detach())
                entropy_loss = dist.entropy().mean()

            else:
                # Actions are used as indices, must be 
                # LongTensor
                action_tensor = action_tensor.long()
                selected_probs_tensor = torch.index_select(probs, 1, action_tensor).diag()
                ratios = selected_probs_tensor/old_selected_probs_tensor
                entropy_loss = torch.distributions.Categorical(probs = probs).entropy().mean()

            
            #advantages_tensor = rewards_tensor - values_tensor.detach()   

            loss_actor = ratios*advantages_tensor
            clipped_loss_actor = torch.clamp(ratios, 1-self.hyperParams.EPSILON, 1+self.hyperParams.EPSILON)*advantages_tensor

            loss_actor = -(torch.min(loss_actor, clipped_loss_actor).mean())

            # Calculate critic loss
            '''value_pred_clipped = old_values_tensor + (values_tensor - old_values_tensor).clamp(-self.hyperParams.EPSILON, self.hyperParams.EPSILON)
            value_losses = (values_tensor - rewards_tensor) ** 2
            value_losses_clipped = (value_pred_clipped - rewards_tensor) ** 2

            loss_critic = 0.5 * torch.max(value_losses, value_losses_clipped)
            loss_critic = loss_critic.mean()''' 


            loss_critic = self.mse(values_tensor, rewards_tensor)
         

            loss = loss_actor + self.hyperParams.COEFF_CRITIC_LOSS * loss_critic -  self.hyperParams.COEFF_ENTROPY_LOSS * entropy_loss

            self.tab_losses.append(loss.item())

            print(loss)

            # Reset gradients
            self.optimizer.zero_grad()
            # Calculate gradients
            loss.backward()
            # Apply gradients
            self.optimizer.step()

        self.old_model.load_state_dict(self.model.state_dict())
        self.reset_batches()


    def memorize(self, ob_prec, val, action_probs, action, ob, reward, done):
        self.states.append(ob_prec)
        self.values.extend(val)
        self.rewards.append(reward)
        self.actions.append(action)
        if(self.continuous_action_space):
            self.selected_probs.append(action_probs)
        else:
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
        self.batch_rewards.extend(discount_rewards(self.rewards, self.list_done, self.hyperParams.GAMMA))
        gaes = gae(np.expand_dims(np.array(self.rewards), 0), np.expand_dims(np.array(self.values), 0), np.expand_dims(np.array([not elem for elem in self.list_done]), 0), self.hyperParams.GAMMA, self.hyperParams.LAMBDA)
        self.batch_advantages.extend(gaes[0])
        self.batch_states.extend(self.states)
        self.batch_values.extend(self.values)
        self.batch_actions.extend(self.actions)
        self.batch_done.extend(self.list_done)
        self.batch_selected_probs.extend(self.selected_probs)


    def act(self, observation):
        # Get actions and convert to numpy array
        action_probs, val = self.old_model(torch.tensor(observation))
        val = val.detach().numpy()
        if(self.continuous_action_space):
            std_mat = torch.diag(self.action_std)
            dist = MultivariateNormal(action_probs, std_mat)
            action = dist.sample().item()
            action_probs = dist.log_prob(action)
        else:
            action_probs = action_probs.detach().numpy()
            action = np.random.choice(np.arange(self.ac_space), p=action_probs)

        self.num_decisions_made += 1

        return action, val, action_probs