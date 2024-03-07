
import random

from torch.nn import HuberLoss
from torchrl import data

import numpy as np

import copy
import numpy as np
import torch

from NeuralNetworks import ActorCritic

class DQNHyperParams :
    def __init__(self):
        self.BUFFER_SIZE = 25000 
        self.TARGET_UPDATE = 7500
        self.GAMMA = 0.99
        self.LR = 0.001
        self.BATCH_SIZE = 128

        self.HIDDEN_SIZE_1 = 128
        self.HIDDEN_SIZE_2 = 128

        self.DECISION_COUNT = 1.5e6
        self.LEARNING_START = 10000
        self.LEARN_EVERY = 3600

        self.START_EPSILON = 1.0
        self.FIRST_MIN_EPSILON = 0.1
        self.FIRST_EPSILON_DECAY = (self.START_EPSILON-self.FIRST_MIN_EPSILON)/(5e4-self.LEARNING_START)
        self.MIN_EPSILON = 0.01
        self.SECOND_EPSILON_DECAY = (self.FIRST_MIN_EPSILON-self.MIN_EPSILON)/5e4


class DQNAgent(object):
    def __init__(self, ob_space, ac_space, test=False, double=True, duelling=True, PER=False, cuda=False, model_to_load=None, cnn=True):

        self.hyperParams = DQNHyperParams()

        self.ac_space = ac_space 

        self.epsilon = self.hyperParams.START_EPSILON
        self.epsilon_decay_value = self.hyperParams.FIRST_EPSILON_DECAY
        
        self.gamma = self.hyperParams.GAMMA
        self.test = False

        self.num_decisions_made = 0

        self.device = torch.device("cuda" if cuda else "cpu")

        self.duelling = duelling
        self.cnn = cnn
        if(self.duelling):
            self.model = ActorCritic(ob_space, ac_space, self.hyperParams, cnn=cnn).to(self.device) 
        else:
            self.model = Actor(ob_space, ac_space, self.hyperParams, cnn=cnn).to(self.device) #for cartpole


        if(model_to_load != None): #if it's a test, use the loaded NN
            self.epsilon = 0.01
            actor = torch.load(model_to_load, map_location=self.device)
            self.model.load_state_dict(actor)
            self.model.eval()
            self.test = True
        else:
            self.model_target = copy.deepcopy(self.model) #a target network is used to make the convergence possible (see papers on DRL)

            self.optimizer = torch.optim.Adam(self.model.parameters(), self.hyperParams.LR) # smooth gradient descent

            self.ob_space = ob_space

            self.double = double

            self.update_target = 0

            self.tab_max_q = []
            self.tab_losses = []

            self.buffer_size = int(self.hyperParams.BUFFER_SIZE)

            if(cnn):
                ob_dtype = torch.uint8
            else:
                ob_dtype = torch.float

            self.batch_prec_states = torch.zeros((self.buffer_size,) + self.ob_space, dtype=ob_dtype)
            self.batch_actions = torch.zeros(self.buffer_size)
            self.batch_states = torch.zeros((self.buffer_size,) + self.ob_space, dtype=ob_dtype)
            self.batch_rewards = torch.zeros(self.buffer_size)
            self.batch_dones = torch.zeros(self.buffer_size)

            self.b_inds = np.arange(self.buffer_size)

            self.num_transition_stored = 0

            self.loss = HuberLoss()



    def epsilon_decay(self):
        self.epsilon -= self.epsilon_decay_value
        if(self.epsilon_decay_value == self.hyperParams.FIRST_EPSILON_DECAY and self.epsilon<self.hyperParams.FIRST_MIN_EPSILON ):
            self.epsilon = self.hyperParams.FIRST_MIN_EPSILON
            self.epsilon_decay_value = self.hyperParams.SECOND_EPSILON_DECAY
        if(self.epsilon < self.hyperParams.MIN_EPSILON):
            self.epsilon = self.hyperParams.MIN_EPSILON

  
        


    def act(self, observation):
        if(not self.test):
            if(self.num_transition_stored >= self.hyperParams.LEARNING_START):
                self.epsilon_decay()
            self.update_target += 1  
        
        observation = np.expand_dims(observation, axis=0)
        observation = torch.tensor(observation, device=self.device)
        tens_qvalue = self.model(observation) #compute the qvalues for the observation
        tens_qvalue = tens_qvalue.squeeze()
        rand = random.random()
        if(rand > self.epsilon): #noise management
            _, indices = tens_qvalue.max(0) #finds the index of the max qvalue
            action = indices #return it
        else:
            action = torch.tensor(random.randint(0, tens_qvalue.size()[0]-1)) #choose a random action
        self.num_decisions_made += 1
        return action.item(), []


    def memorize(self, ob_prec, action, ob, reward, done, infos):
        self.batch_prec_states[self.num_transition_stored%self.buffer_size] = torch.Tensor(np.array(ob_prec))
        self.batch_actions[self.num_transition_stored%self.buffer_size] = action
        self.batch_states[self.num_transition_stored%self.buffer_size] = torch.Tensor(np.array(ob))
        self.batch_rewards[self.num_transition_stored%self.buffer_size] = torch.tensor(reward).view(-1)
        self.batch_dones[self.num_transition_stored%self.buffer_size] = not(done)

        self.num_transition_stored += 1
  

    def learn(self, n_iter=None):


        m_batch_inds = np.random.choice(self.b_inds[:min(self.num_transition_stored, self.buffer_size)],\
        size=self.hyperParams.BATCH_SIZE, replace=False)
        
        tens_state = self.batch_prec_states[m_batch_inds]
        tens_state_next = self.batch_states[m_batch_inds]
        tens_action = self.batch_actions[m_batch_inds].long()
        tens_reward = self.batch_rewards[m_batch_inds].float()
        tens_done = self.batch_dones[m_batch_inds].bool()

        tens_qvalue = self.model(tens_state) #compute the qvalues for all the actual states

        tens_qvalue = torch.index_select(tens_qvalue, 1, tens_action).diag() #select the qvalues corresponding to the chosen actions


        
        if(self.double):
            # Double DQN
            tens_next_qvalue = self.model(tens_state_next) #compute all the qvalues for all the "next states" with the ppal network
            (_, tens_next_action) = torch.max(tens_next_qvalue, 1) #returns the indices of the max qvalues for all the next states(to choose the next actions)
            tens_next_qvalue = self.model_target(tens_state_next) #compute all the qvalues for all the "next states" with the target network
            tens_next_qvalue = torch.index_select(tens_next_qvalue, 1, tens_next_action).diag() #select the qvalues corresponding to the chosen next actions          
        else:
            # Simple DQN
            tens_next_qvalue = self.model_target(tens_state_next) #compute all the qvalues for all the "next states"
            (tens_next_qvalue, _) = torch.max(tens_next_qvalue, 1) #select the max qvalues for all the next states
            

        self.optimizer.zero_grad() #reset the gradient
        tens_loss = self.loss(tens_qvalue, tens_reward+(self.gamma*tens_next_qvalue)*tens_done) #calculate the loss
        tens_loss.backward() #compute the gradient
        self.optimizer.step() #back-propagate the gradient

        self.tab_max_q.append(torch.max(tens_qvalue).item())
        self.tab_losses.append(torch.max(tens_loss).item())

        #print(tens_loss.item(), torch.max(tens_qvalue))
        
        if(self.update_target/self.hyperParams.TARGET_UPDATE > 1):
            self.model_target.load_state_dict(self.model.state_dict())
            self.update_target = 0