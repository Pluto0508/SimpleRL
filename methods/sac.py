import numpy as np
import gymnasium as gym
from gym import Env
from typing import Optional
import torch
from torch import optim
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import math

from model.AC import Actor,Critic
from model.buffer import PPOBuffer
from distributions.general import Normal
from utils.utils import *

class SACTrainer:
    def __init__(self,
                 actor: Optional[Actor]=None,
                 env: Optional[Env]=None,
                 train_epoch=1000,
                 buffer_size=100):
        
        self.actor=actor
        self.critic0=None
        self.critic1=None
        self.critic_target0=None
        self.critic_target1=None
        self.action_dim=None
        self.state_dim=None
        self.train_epoch=train_epoch
        self.env=env
        self.buffer=PPOBuffer(buffer_size)

        self.optimizer=None
        self.gamma=0.9
        self.clip_epsilon=0.2
        self.temp=0.9
        self.alpha=0.02

        self.actor_loss=[]
        self.critic_loss=[]
    
    def init_model(self,state_dim,action_dim):
        self.state_dim=state_dim
        self.action_dim=action_dim

        self.actor=Actor(state_dim,action_dim)
        self.critic0=Critic(state_dim,action_dim)
        self.critic1=Critic(state_dim,action_dim)
        self.critic_target0=deep_clone_model(self.critic0)
        self.critic_target1=deep_clone_model(self.critic1)

        self.env=gym.make('MountainCarContinuous-v0',render_mode='human')
        self.optimizer_actor=optim.Adam(self.actor.parameters(),lr=0.001)
        self.optimizer_critic0=optim.Adam(self.critic0.parameters(),lr=0.001)
        self.optimizer_critic1=optim.Adam(self.critic0.parameters(),lr=0.001)
        return self
    
    def sample(self,sample_epoch):
        self.buffer.clear()
        self.actor.eval()

        with torch.no_grad():
            while self.buffer.current_size<self.buffer.size:
                state=self.env.reset()
                state=torch.tensor(state[0])
                for _ in range(sample_epoch):
                    res=self.actor(state)
                    action=self.actor.make_action(res)
                    next_state,reward,done,_,_=self.env.step(deep_clone_tensor(action['action']).numpy())
                    next_state=torch.tensor(next_state)
                    reward=torch.tensor(reward)
                    self.env.render()
                    self.buffer.push(deep_clone_tensor(state),deep_clone_tensor(action['action']),deep_clone_tensor(next_state),reward,done,action['log_prob'],None )
                    if done:
                        break
                    state=next_state
    
    def soft_update(self,alpha=0.05):
        for param_target,param in zip(self.critic_target0.parameters(),self.critic0.parameters()):
            param_target.data.copy_(alpha*param.data+(1-alpha)*param_target.data)
        
        for param_target,param in zip(self.critic_target1.parameters(),self.critic1.parameters()):
            param_target.data.copy_(alpha*param.data+(1-alpha)*param_target.data)


    def train(self,batch_size=4,sample_method='normal'):
        self.actor.train()
        self.critic0.train()
        self.critic1.train()

        for _ in range(self.train_epoch):

            for idx,sample in enumerate(self.buffer.sample(batch_size)):
                for key,value in sample.items():
                    if key in ['state','next_state','action','reward']:
                        sample[key]=torch.stack(value)
                    
                #update q
                with torch.no_grad():
                    res_actor=self.actor(sample['next_state'])
                    out_action=self.actor.make_action(res_actor)
                    q_target0=self.critic_target0(sample['next_state'],out_action['action'])
                    q_target1=self.critic_target1(sample['next_state'],out_action['action'])
                    q_target=torch.min(torch.cat([q_target0,q_target1],dim=-1),dim=-1).values-self.alpha*torch.tensor(out_action['log_prob'])
                    q_target=sample['reward']+self.gamma*q_target
                
                #loss critic
                q_current0=self.critic0(sample['state'],sample['action'])
                q_current1=self.critic1(sample['state'],sample['action'])
                loss0=nn.MSELoss()(q_current0,q_target0)
                loss1=nn.MSELoss()(q_current1,q_target1)

                #update critic
                self.optimizer_critic0.zero_grad()
                loss0.backward()
                self.optimizer_critic0.step()

                self.optimizer_critic1.zero_grad()
                loss1.backward()
                self.optimizer_critic1.step()

                #update actor
                res_actor_new=self.actor(sample['state'])
                out_action_new=self.actor.make_action(res_actor_new)
                q_new0=self.critic0(sample['state'],out_action_new['action'])
                q_new1=self.critic1(sample['state'],out_action_new['action'])
                q_new=torch.min(torch.cat([q_new0,q_new1],dim=-1),dim=-1).values
                loss_actor=(torch.tensor(out_action_new['log_prob'])-q_new).mean()
                
                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                self.optimizer_actor.step()

                if idx%50==0:
                    print(f'critic_loss:{loss0}  {loss1}-------actor_loss:{loss_actor}')
            
            self.soft_update()

def sac_sample(epoch=4):
    trainer=SACTrainer(train_epoch=2)
    trainer.init_model(2,1)
    
    for i in range(epoch):
        trainer.sample(sample_epoch=2)
        trainer.train()