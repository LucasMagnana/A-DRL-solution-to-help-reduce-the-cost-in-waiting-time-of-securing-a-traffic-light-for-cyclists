
import random
from random import sample, random, randint

from torch.nn import MSELoss
from torchrl import data

import numpy as np

import copy
import numpy as np
import torch

from NeuralNetworks import Actor, DuellingActorCNN, DuellingActor

class DQNHyperParams :
    def __init__(self):
        self.BUFFER_SIZE = 25000 
        self.ALPHA = 0.05 #
        self.GAMMA = 0.99
        self.LR = 0.001
        self.BATCH_SIZE = 128

        self.HIDDEN_SIZE = 16
        self.ACT_INTER = 16

        self.UPDATE_TARGET = 7500

        self.DECISION_COUNT = 750000
        self.MAX_STEPS = 1000
        self.DECISION_CT_LEARNING_START = 10000
        self.LEARNING_EP = 1

        self.EPSILON = 1.0
        self.MIN_EPSILON = 0.05
        self.EPSILON_DECAY = self.EPSILON/(self.DECISION_COUNT)

class DQNAgent(object):
    def __init__(self, observation_space, action_space, test=False, double=False, duelling=False, PER=False, cnn=None, cuda=False, model_to_load=None):

        self.hyperParams = DQNHyperParams()
        
        if(model_to_load != None): #use the good hyper parameters (loaded if it's a test, written in the code if it's a training)
            self.hyperParams.EPSILON = 0 #self.hyperParams.MIN_EPSILON

        self.action_space = action_space 
        
        self.PER = PER
        if(not test):
            if(self.PER):  
                self.buffer = data.PrioritizedReplayBuffer(int(self.hyperParams.BUFFER_SIZE), 0.1, 0.2)
            else:
                self.buffer = data.ReplayBuffer(int(self.hyperParams.BUFFER_SIZE))

        self.alpha = self.hyperParams.ALPHA
        self.epsilon = self.hyperParams.EPSILON
        self.gamma = self.hyperParams.GAMMA

        self.device = torch.device("cuda" if cuda else "cpu")

        if(len(observation_space) > 1):
            self.cnn = True
        else:
            self.cnn = False

        self.duelling = duelling
        if(self.duelling):
            if(self.cnn):
                self.model = DuellingActorCNN(observation_space, action_space).to(self.device) #for cartpole
            else:
                self.model = DuellingActor(observation_space, action_space).to(self.device) #for cartpole
        else:
            self.model = Actor(observation_space, action_space).to(self.device) #for cartpole

        self.batch_size = self.hyperParams.BATCH_SIZE


        if(model_to_load != None): #if it's a test, use the loaded NN
            self.model.load_state_dict(torch.load(model_to_load, map_location=self.device))
            self.model.eval()
        
        self.model_target = copy.deepcopy(self.model) #a target network is used to make the convergence possible (see papers on DRL)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.hyperParams.LR) # smooth gradient descent

        self.observation_space = observation_space

        self.double = double

        self.num_decisions_made = 0

        self.tab_losses = []        
        


    def act(self, observation):
        #actual noise decaying method, works well with the custom env
        self.epsilon -= self.hyperParams.EPSILON_DECAY
        if(self.epsilon<self.hyperParams.MIN_EPSILON):
            self.epsilon=self.hyperParams.MIN_EPSILON
        self.num_decisions_made += 1

        if(self.num_decisions_made%self.hyperParams.UPDATE_TARGET == 0):
            self.model_target.load_state_dict(self.model.state_dict())
        
        observation = torch.tensor(observation, device=self.device)
        tens_qvalue = self.model(observation) #compute the qvalues for the observation
        tens_qvalue = tens_qvalue.squeeze()
        #print(tens_qvalue)
        rand = random()
        if(rand > self.epsilon): #noise management
            _, indices = tens_qvalue.max(0) #finds the index of the max qvalue
            return indices.item() #return it
        return randint(0, tens_qvalue.size()[0]-1) #choose a random action

    def sample(self):
        if(len(self.buffer) < self.batch_size):
            return self.buffer.sample(len(self.buffer))
        else:
            return self.buffer.sample(self.batch_size)

    def memorize(self, ob_prec, action, ob, reward, done):
        experience = copy.deepcopy(ob_prec).flatten()
        experience = np.append(experience, action)
        experience = np.append(experience, ob.flatten())
        experience = np.append(experience, reward)
        experience = np.append(experience, not(done))
        self.buffer.add(torch.FloatTensor(experience, device=self.device))   

    def learn(self, n_iter=None):
        
        #previous noise decaying method, works well with cartpole
        '''if(self.epsilon > self.hyperParams.MIN_EPSILON):
            self.epsilon *= self.hyperParams.EPSILON_DECAY
        else:
            self.epsilon = 0'''

        loss = MSELoss()

        spl = self.sample()  #create a batch of experiences
        if(self.PER):
            datas = spl[1]
            spl = spl[0]


        spl = torch.split(spl, [np.prod(self.observation_space), 1, np.prod(self.observation_space), 1, 1], dim=1)

        tens_state = spl[0]

        if(self.cnn):
            tens_state = tens_state.view(spl[1].shape[0], self.observation_space[0], self.observation_space[1], self.observation_space[2])
            

        tens_action = spl[1].squeeze().long()

        
        tens_state_next = spl[2]
        
        if(self.cnn):
            tens_state_next = tens_state_next.view(spl[1].shape[0], self.observation_space[0], self.observation_space[1], self.observation_space[2])

        tens_reward = spl[3].squeeze()

        tens_done = spl[4].squeeze().bool()

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
        tens_loss = loss(tens_qvalue, tens_reward+(self.gamma*tens_next_qvalue)*tens_done) #calculate the loss
        print()
        print("loss : ",tens_loss)
        self.tab_losses.append(tens_loss.item())
        tens_loss.backward() #compute the gradient
        self.optimizer.step() #back-propagate the gradient

        '''for target_param, param in zip(self.model_target.parameters(), self.model.parameters()): #updates the target network
            target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )'''