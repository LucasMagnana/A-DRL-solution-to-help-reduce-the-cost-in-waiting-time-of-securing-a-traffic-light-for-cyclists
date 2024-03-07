import torch
from torch import nn
from torch.autograd import Variable
from math import *
import numpy as np


def layer_init(layer, ppo=False, std=np.sqrt(2), bias_const=0.0):
    if(ppo):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    else:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    return layer


class Actor(nn.Module):

    def __init__(self, size_ob, size_action, hyperParams, cnn=False): #for saved hyperparameters
        super(Actor, self).__init__()

        self.cnn = cnn

        if(cnn):
            self.network = CNN_layers(size_ob, hyperParams, ac=False)
        else:
            l1 = layer_init(nn.Linear(np.array(size_ob).prod(), hyperParams.HIDDEN_SIZE_1))

            l2 = layer_init(nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2))

            activation = nn.ReLU()

            self.network = nn.Sequential(
                l1,
                activation,
                l2,
                activation
            )

        self.actor = layer_init(nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action), std=0.01)



    def forward(self, ob):
        features = self.network(ob)

        if(self.cnn):
            qvalues = self.actor(features[1])

        else:
            qvalues = self.actor(features)

        return qvalues


class ActorCritic(nn.Module):

    def __init__(self, size_ob, size_action, hyperParams, cnn=False, ppo=False, max_action=-1): #for saved hyperparameters
        super(ActorCritic, self).__init__()

        self.cnn = cnn
        self.ppo = ppo
        self.max_action = max_action

        if(cnn):
            self.network = CNN_layers(size_ob, hyperParams, ppo)
        else:
            l1 = layer_init(nn.Linear(np.array(size_ob).prod(), 64), ppo)

            l2 = layer_init(nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2), ppo)

            if(ppo):
                activation = nn.Tanh()
            else:
                activation = nn.ReLU()

            self.network = nn.Sequential(
                l1,
                activation,
                l2,
                activation
            )

        self.actor = layer_init(nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action), ppo, std=0.01)

        self.critic = layer_init(nn.Linear(hyperParams.HIDDEN_SIZE_2, 1), ppo, std=1)

        if(max_action>0):
            self.stds = nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action)
            torch.nn.init.kaiming_normal_(self.stds.weight, nonlinearity="relu")



    def forward(self, ob):
        features = self.network(ob)

        if(self.cnn):
            values = self.critic(features[0])
            advantages = self.actor(features[1])

        else:
            advantages = self.actor(features)
            values = self.critic(features)

        if(self.ppo):
            if(self.max_action>0):
                stds = self.stds(features)
                return nn.functional.tanh(advantages)*self.max_action, nn.functional.sigmoid(stds), values
            else:
                return advantages, values
        else:
            return values + (advantages - advantages.mean())


class CNN_layers(nn.Module):

    def __init__(self, size_ob, hyperParams, ppo=False, ac=True): #for saved hyperparameters
        super(CNN_layers, self).__init__()

        if(ac):
            ol_size = hyperParams.HIDDEN_SIZE_2*2
        else:
            ol_size = hyperParams.HIDDEN_SIZE_2

        self.hidden_size = hyperParams.HIDDEN_SIZE_2

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(size_ob[0], 16, 2), ppo),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 16, 2), ppo),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            layer_init(nn.Linear(1728, ol_size), ppo),
            nn.ReLU())


    def forward(self, ob):
        features = self.cnn(ob.float())
        return torch.split(features, self.hidden_size, dim=1)