from random import sample
from random import * 
from torch.nn import MSELoss

import copy
import numpy as np
import torch

from NeuralNetworks import Actor

class DDQNHyperParams :
    def __init__(self):
        self.BUFFER_SIZE = 10000 
        self.ALPHA = 0.05 #
        self.GAMMA = 0.99
        self.LR = 0.01
        self.BATCH_SIZE = 64

        self.HIDDEN_SIZE = 16
        self.ACT_INTER = 16

        self.EPISODE_COUNT = 36000
        self.MAX_STEPS = 1000
        self.LEARNING_START = 0

        self.EPSILON = 1.0
        self.MIN_EPSILON = 0
        self.EPSILON_DECAY = self.EPSILON/(self.EPISODE_COUNT*4/5)

class DDQNAgent(object):
    def __init__(self, observation_space, action_space, cuda=False, actor_to_load=None):

        self.hyperParams = DDQNHyperParams()
        
        if(actor_to_load != None): #use the good hyper parameters (loaded if it's a test, written in the code if it's a training)
            self.hyperParams.EPSILON = 0

        self.action_space = action_space   

        self.buffer = [] #replay buffer of the agent
        self.buffer_max_size = self.hyperParams.BUFFER_SIZE

        self.alpha = self.hyperParams.ALPHA
        self.epsilon = self.hyperParams.EPSILON
        self.gamma = self.hyperParams.GAMMA

        self.device = torch.device("cuda" if cuda else "cpu")

        self.actor = Actor(observation_space, action_space, self.hyperParams).to(self.device) #for cartpole

        self.batch_size = self.hyperParams.BATCH_SIZE


        if(actor_to_load != None): #if it's a test, use the loaded NN
            self.actor.load_state_dict(torch.load(actor_to_load, map_location=self.device))
            self.actor.eval()
        
        self.actor_target = copy.deepcopy(self.actor) #a target network is used to make the convergence possible (see papers on DRL)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), self.hyperParams.LR) # smooth gradient descent

        self.observation_space = observation_space

        


    def act(self, observation):
        #return self.action_space.sample()
        observation = torch.tensor(observation, device=self.device)
        tens_qvalue = self.actor(observation) #compute the qvalues for the observation
        tens_qvalue = tens_qvalue.squeeze()
        rand = random()
        if(rand > self.epsilon): #noise management
            _, indices = tens_qvalue.max(0) #finds the index of the max qvalue
            return indices.item() #return it
        return randint(0, tens_qvalue.size()[0]-1) #choose a random action

    def sample(self):
        if(len(self.buffer) < self.batch_size):
            return sample(self.buffer, len(self.buffer))
        else:
            return sample(self.buffer, self.batch_size)

    def memorize(self, ob_prec, action, ob, reward, done):
        if(len(self.buffer) > self.buffer_max_size): #delete the first element if the buffer is at max size 
            self.buffer.pop(0)
        self.buffer.append([ob_prec, action, ob, reward, not(done)])

    def learn(self, n_iter=None):

        #previous noise decaying method, works well with cartpole
        '''if(self.epsilon > self.hyperParams.MIN_EPSILON):
            self.epsilon *= self.hyperParams.EPSILON_DECAY
        else:
            self.epsilon = 0'''

        
        #actual noise decaying method, works well with the custom env
        self.epsilon -= self.hyperParams.EPSILON_DECAY
        if(self.epsilon<self.hyperParams.MIN_EPSILON):
            self.epsilon=self.hyperParams.MIN_EPSILON

        loss = MSELoss()

        spl = self.sample()  #create a batch of experiences

        spl=list(zip(*spl))

        tens_state=torch.tensor(spl[0], device=self.device)

        tens_action=torch.tensor(spl[1], device=self.device)
        tens_action=tens_action.long()

        tens_state_next=torch.tensor(spl[2], device=self.device)

        tens_reward=torch.tensor(spl[3], device=self.device)

        tens_done=torch.tensor(spl[4], device=self.device)

        tens_qvalue = self.actor(tens_state) #compute the qvalues for all the actual states

        tens_qvalue = torch.index_select(tens_qvalue, 1, tens_action).diag() #select the qvalues corresponding to the chosen actions

        '''
        # Simple DQN
        tens_next_qvalue = self.actor_target(tens_state_next) #compute all the qvalues for all the "next states"
        (tens_next_qvalue, _) = torch.max(tens_next_qvalue, 1) #select the max qvalues for all the next states'''
        
        # Double DQN
        tens_next_qvalue = self.actor(tens_state_next) #compute all the qvalues for all the "next states" with the ppal network
        (_, tens_next_action) = torch.max(tens_next_qvalue, 1) #returns the indices of the max qvalues for all the next states(to choose the next actions)
        tens_next_qvalue = self.actor_target(tens_state_next) #compute all the qvalues for all the "next states" with the target network
        tens_next_qvalue = torch.index_select(tens_next_qvalue, 1, tens_next_action).diag() #select the qvalues corresponding to the chosen next actions

        self.optimizer.zero_grad() #reset the gradient
        tens_loss = loss(tens_qvalue, tens_reward+(self.gamma*tens_next_qvalue)*tens_done) #calculate the loss
        tens_loss.backward() #compute the gradient
        self.optimizer.step() #back-propagate the gradient

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()): #updates the target network
            target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )