import torch
from torch import nn
from torch.autograd import Variable

def shape_after_conv_and_flatten(input_shape, conv):
    return int((input_shape[1] + 2*conv.padding[0] - conv.dilation[0]*(conv.kernel_size[0] - 1) -1)/conv.stride[0]+1)*\
    int((input_shape[2] + 2*conv.padding[1] - conv.dilation[1]*(conv.kernel_size[1] - 1) -1)/conv.stride[1]+1)*conv.out_channels

class Actor(nn.Module):

    def __init__(self, size_ob, size_action, max_action=1, tanh=False): #for saved hyperparameters
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(size_ob[0], 16, 2)
        out_shape = shape_after_conv_and_flatten(size_ob[2], self.conv1)
        self.out = nn.Linear(out_shape, size_action)
        self.max_action = max_action
        self.tanh = tanh

    def forward(self, ob):
        ob = ob.float()
        out = nn.functional.relu(self.conv1(ob))
        if(len(out.shape) == 3):
            out = torch.flatten(out)
        elif(len(out.shape) == 4):
            out = torch.flatten(out, start_dim=1)
        if(self.tanh):
            return torch.tanh(self.out(out)*self.max_action)
        else:
            return self.out(out)*self.max_action



class DuellingActor(nn.Module):

    def __init__(self, size_ob, size_action, max_action=1, tanh=False): #for saved hyperparameters
        super(DuellingActor, self).__init__()

        self.conv1 = nn.Conv2d(size_ob[0], 16, 2)
        out_shape = shape_after_conv_and_flatten(size_ob[2], self.conv1)

        self.advantage_out = nn.Linear(out_shape, size_action)

        self.value_out = nn.Linear(out_shape, 1)

        self.max_action = max_action
        self.tanh = tanh

    def forward(self, ob):
        ob = ob.float()
        features = nn.functional.relu(self.conv1(ob))
        if(len(features.shape) == 3):
            features = torch.flatten(features)
        elif(len(features.shape) == 4):
            features = torch.flatten(features, start_dim=1)

        values = self.value_out(features)

        advantages = self.advantage_out(features)

        return values + (advantages - advantages.mean())


        


class PPO_Model(nn.Module):
    def __init__(self, size_ob, size_action):
        super(PPO_Model, self).__init__()

        self.conv1 = nn.Conv2d(size_ob[0], 16, 2)
        out_shape = shape_after_conv_and_flatten(size_ob, self.conv1)
        self.out = nn.Linear(out_shape, 32)

        self.actor = nn.Sequential(
            nn.Linear(32, size_action),
            nn.Softmax(dim=-1))

        self.critic = nn.Linear(32, 1)
    
    def forward(self, ob):
        ob = ob.float()
        features = nn.functional.relu(self.conv1(ob))
        if(len(features.shape) == 3):
            features = torch.flatten(features)
        elif(len(features.shape) == 4):
            features = torch.flatten(features, start_dim=1)
        features = nn.functional.relu(self.out(features))
        return self.actor(features), self.critic(features)

