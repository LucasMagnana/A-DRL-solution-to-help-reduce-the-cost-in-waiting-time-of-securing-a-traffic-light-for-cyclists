import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical

import pickle


from NeuralNetworks import *

class PPOHyperParams : 
    def __init__(self):
        self.GAMMA = 0.99
        self.LR = 2.5e-4

        self.HIDDEN_SIZE_1 = 128
        self.HIDDEN_SIZE_2 = 128

        self.DECISION_COUNT = 1.5e6

        self.LAMBDA = 0.95
        self.EPSILON = 0.2

        self.K = 4
        self.NUM_ENV = 1
        self.BATCH_SIZE = 512//self.NUM_ENV
        self.NUM_MINIBATCHES = 4

        self.ENTROPY_COEFF = 0.01
        self.VALUES_COEFF = 0.5
        self.MAX_GRAD = 0.5
        self.GAMMA_GAE = 0.95


def gae(rewards, values, dones, num_steps, nextdones, nextob, actor, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - torch.tensor(nextdones).float()
            with torch.no_grad():
                _, nextvalues = actor(torch.tensor(nextob))
                nextvalues = nextvalues.flatten()
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages




class PPOAgent():

    def __init__(self, ob_space, ac_space, model_to_load=None, continuous_action_space=False, cnn=True):

        self.hyperParams = PPOHyperParams()

        self.continuous_action_space = continuous_action_space

        self.ac_space = ac_space
        self.ob_space = ob_space

        device = torch.device("cpu")

        if(self.continuous_action_space):
            self.model = PPO_Actor(ob_space, ac_space, hyperParams, max_action=ac_space)
        else:  
            self.model = ActorCritic(ob_space, ac_space, self.hyperParams, cnn=cnn, ppo=True).to(device)

        if(model_to_load != None):
            self.model.load_state_dict(torch.load(model_to_load))
            self.model.eval()


        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperParams.LR, eps=1e-5)
        self.mse = torch.nn.MSELoss()

        

        if(self.continuous_action_space):
            self.action_std = torch.full((ac_space,), 1/100)

        self.num_decisions_made = 0

        self.e_loss = []
        self.p_loss = []
        self.v_loss = []
        self.lr = []

        self.tab_losses = []


        self.batch_rewards = torch.zeros((self.hyperParams.BATCH_SIZE, self.hyperParams.NUM_ENV))
        self.batch_states = torch.zeros((self.hyperParams.BATCH_SIZE, self.hyperParams.NUM_ENV) + self.ob_space)
        self.batch_values = torch.zeros((self.hyperParams.BATCH_SIZE, self.hyperParams.NUM_ENV))
        self.batch_actions = torch.zeros((self.hyperParams.BATCH_SIZE, self.hyperParams.NUM_ENV))
        self.batch_log_probs = torch.zeros((self.hyperParams.BATCH_SIZE, self.hyperParams.NUM_ENV))
        self.batch_dones = torch.zeros((self.hyperParams.BATCH_SIZE, self.hyperParams.NUM_ENV))
    



    def memorize(self, ob_prec, action, ob, reward, done, infos):
        val = infos[0]
        action_probs = infos[1]
        env = 0
        step = self.num_decisions_made%self.hyperParams.BATCH_SIZE
        self.batch_states[step] = torch.Tensor(ob_prec)
        self.batch_values[step] = val
        self.batch_rewards[step] = torch.tensor(reward).view(-1)
        self.batch_actions[step] = action
        self.batch_log_probs[step] = action_probs
        if(self.hyperParams.NUM_ENV == 1):
            done = [done]
        self.batch_dones[step] = torch.Tensor(done)


    def act(self, observation):
        with torch.no_grad():
            if(self.continuous_action_space):
                action_exp, action_std = self.model(torch.tensor(observation))
                std_mat = torch.diag(action_std)
                dist = MultivariateNormal(action_exp, std_mat)
            else:
                observation = np.expand_dims(observation, axis=0)
                action_probs, val = self.model(torch.tensor(observation))
                dist = Categorical(logits=action_probs)

            action = dist.sample()
            self.num_decisions_made += 1

            return action.item(), [val.flatten(), dist.log_prob(action)]



    
    def learn(self, next_dones, next_obs):
        next_obs = np.expand_dims(next_obs, axis=0)
        gaes = gae(self.batch_rewards, self.batch_values, self.batch_dones, self.hyperParams.BATCH_SIZE, next_dones, next_obs, self.model, self.hyperParams.GAMMA, self.hyperParams.LAMBDA)
        returns = gaes + self.batch_values
        # flatten the batch
        batch_states = self.batch_states.reshape((-1,) + self.ob_space)
        batch_logprobs = self.batch_log_probs.reshape(-1)
        batch_actions = self.batch_actions.reshape(-1)
        batch_advantages = gaes.reshape(-1)
        batch_returns = returns.reshape(-1)
        #print(batch_returns.shape, batch_returns.mean().item())
        batch_values = self.batch_values.reshape(-1)


        # Optimizing the policy and value network
        batch_inds = np.arange(len(batch_states))
        clipfracs = []
        for epoch in range(self.hyperParams.K):
            np.random.shuffle(batch_inds)
            for start in range(0, len(batch_states), len(batch_states)//self.hyperParams.NUM_MINIBATCHES):                
                end = start + len(batch_states)//self.hyperParams.NUM_MINIBATCHES
                m_batch_inds = batch_inds[start:end]
                action_probs, newvalue = self.model(batch_states[m_batch_inds])
                probs = Categorical(logits=action_probs)
                newlogprob = probs.log_prob(batch_actions.long()[m_batch_inds])
                entropy = probs.entropy()
                logratio = newlogprob - batch_logprobs[m_batch_inds]
                ratio = logratio.exp()
                norm_batch_advantages = batch_advantages[m_batch_inds]
                norm_batch_advantages = (norm_batch_advantages - norm_batch_advantages.mean()) / (norm_batch_advantages.std() + 1e-8)
        
                # Policy loss
                unclipped_policy_loss = -norm_batch_advantages * ratio
                clipped_policy_loss = -norm_batch_advantages * torch.clamp(ratio, 1 - self.hyperParams.EPSILON, 1 + self.hyperParams.EPSILON)
                e = unclipped_policy_loss
                policy_loss = torch.max(unclipped_policy_loss, clipped_policy_loss)
                policy_loss = policy_loss.mean()

                # Value loss
                newvalue = newvalue.view(-1)
                value_loss_unclipped = (newvalue - batch_returns[m_batch_inds]) ** 2
                values_clipped = batch_values[m_batch_inds] + torch.clamp(
                    newvalue - batch_values[m_batch_inds],
                    -self.hyperParams.EPSILON,
                    self.hyperParams.EPSILON,
                )
                value_loss_clipped = (values_clipped - batch_returns[m_batch_inds]) ** 2
                value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
                value_loss = 0.5 * value_loss_max.mean()
    

                entropy_loss = entropy.mean()

                loss = policy_loss - self.hyperParams.ENTROPY_COEFF * entropy_loss + value_loss * self.hyperParams.VALUES_COEFF


                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.hyperParams.MAX_GRAD)
                self.optimizer.step()

        self.p_loss.append(policy_loss.item())
        self.v_loss.append(value_loss.item())
        self.e_loss.append(entropy_loss.item())
        self.tab_losses.append(loss.item())