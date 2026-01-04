import torch
import torch.nn as nn
import numpy as np
import random

from model.base_model import FNN
from model.base_model import CNN
from distributions.general import Normal
from utils.utils import *

class ActorCritic(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(ActorCritic,self).__init__()
        self.fc=FNN(input_dim)
        self.actor=nn.Linear(32,output_dim)
        self.critic=nn.Linear(32,1)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,x):
        x=self.fc(x)
        #actor
        actor_output=self.actor(x)
        #critic
        critic_value=self.critic(x)
        #prob
        actor_prob=self.softmax(actor_output)

        res={
            'actor_output':actor_output,
            'critic_value':critic_value,
            'actor_prob':actor_prob
        }
        return res
    
    def make_action(self,res,method='max'):
        if method=='max':
            return torch.argmax(res['actor_prob'],dim=-1)
        elif method=='random':
            return np.random.choice(a=len(self.output_dim),p=res['actor_prob'])
        else:
            raise ValueError('No this sampling method!')
        
    
class Actor(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Actor,self).__init__()
        self.fc=FNN(input_dim)
        self.actor_reparam=nn.Linear(32,output_dim*2)
        self.actor_simple=nn.Linear(32,output_dim)
    
    def forward(self,x):
        x=self.fc(x)
        output_reparam=self.actor_reparam(x)
        output_simple=self.actor_simple(x)
        return {
            'output_reparam':output_reparam,
            'output_simple':output_simple
        }
    
    def make_action(self,res,method='simple',sample_func='Normal'):
        if method=='simple':
            out=res['output_simple']
            out=tanh_torch(out)
            return {'action':out,
                    'log_prob':0}

        elif method=='reparam':
            out=res['output_reparam']
            
            if sample_func=='Normal':
                try:
                    sample_dist=Normal(out[:,0].mean(),out[:,1].mean())
                except:
                    sample_dist=Normal(out[0],out[1])
                out=sample_dist.transform_data(sample_dist.sample(),method='tanh')
                return out
        
        else:
            raise ValueError("No sub method!{simple,reparam} are availabel!")
    
class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__() 
        
        self.fc=FNN(state_dim+action_dim)
        self.output=nn.Linear(in_features=32,out_features=1)

    def forward(self,state,action):
        if isinstance(action,dict):
            action=action['action']
        sa=torch.cat([state,action],dim=-1)
        q=self.output(self.fc(sa))
        return q
    


    


            


