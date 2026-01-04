import torch
import torch.nn as nn
import numpy as np
import random

from model.base_model import FNN
from model.base_model import CNN
from distributions.general import Normal
from utils.utils import *

class QNet(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(QNet,self).__init__()
        self.fc=FNN(state_dim)
        self.qnet=nn.Linear(32,action_dim)
    
    def forward(self,state):
        output=self.fc(state)
        output=self.qnet(output)
        return output
    
    def make_action(self,q_values,epsilon=0.1,method='greedy'):
        if method=='greedy':
            return torch.argmax(q_values,dim=-1)
        elif method=='epsilon_greedy':
            if random.random()<epsilon:
                return torch.randint(low=0,high=q_values.size(-1),size=(1,))
            else:
                return torch.argmax(q_values,dim=-1)