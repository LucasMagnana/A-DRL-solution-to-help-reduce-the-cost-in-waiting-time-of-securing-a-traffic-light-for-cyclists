import torch
from torch import nn
from torch.autograd import Variable




class Actor(nn.Module):

    def __init__(self, size_ob, size_action, hyperParams, max_action=1, tanh=False): #for saved hyperparameters
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 2)
        out_shape = self.shape_after_conv_and_flatten(size_ob, self.conv1)
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

    def shape_after_conv_and_flatten(self, w_input, conv):
        return int((conv.in_channels + 2*conv.padding[0] - conv.dilation[0]*(conv.kernel_size[0] - 1) -1)/conv.stride[0]+1)*\
        int((w_input + 2*conv.padding[1] - conv.dilation[1]*(conv.kernel_size[1] - 1) -1)/conv.stride[1]+1)*conv.out_channels

