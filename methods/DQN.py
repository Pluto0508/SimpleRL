import torch
import torch.nn as nn
import math
from typing import Optional
import gymnasium as gym
from gymnasium import Env

from utils.utils import *
from model.base_model import FNN
from model.QNetwork import QNet
from model.buffer import DQNBuffer

class DQNTrainer:
    def __init__(self,
                 qnet: Optional[QNet]=None,
                 env: Optional[Env]=None,
                 train_epoch=1000,
                 buffer_size=100):
        
        self.qnet=qnet
        self.train_epoch=train_epoch
        
        self.env=env
        self.buffer=DQNBuffer(buffer_size)
        
        self.optimizer=None
        self.gamma=0.6
    
    def init_model(self,state_dim,action_dim):
        self.qnet=QNet(state_dim,action_dim)
        self.env=gym.make('MountainCar-v0',render_mode='human')

        self.optimizer=torch.optim.Adam(self.qnet.parameters(),lr=0.001)
    
    def sample(self,sample_epoch):
        self.buffer.clear()
        self.qnet.eval()

        while self.buffer.current_size<self.buffer.size:
            state=self.env.reset()
            state=torch.tensor(state[0])

            with torch.no_grad():
                for _ in range(sample_epoch):
                    res=self.qnet(state)
                    action=self.qnet.make_action(res,method='epsilon_greedy')
                    next_state,reward,done,_,_=self.env.step(action.detach().item())
                    next_state=torch.tensor(next_state)
                    reward=torch.tensor(reward+reward_model(state,next_state))
                    self.env.render()
                    self.buffer.push(
                        {'state':deep_clone_tensor(state),
                         'action':action,
                         'next_state':next_state,
                         'reward':reward,
                         'done':done})
                    
                    if done:
                        break
                    state=next_state
    
    def soft_update(self,alpha=0.05):
        for param_target,param in zip(self.qnet_target.parameters(),self.qnet.parameters()):
            param_target.data.copy_(alpha*param.data+(1-alpha)*param_target.data)

    def train(self):
        self.qnet.train()

        for _ in range(self.train_epoch):
            self.buffer.prepare(shuffle=True)

            for idx,sample in enumerate(self.buffer.sample()):
                #compute target q
                with torch.no_grad():
                    q_next=self.qnet(sample['next_state']).max(dim=-1).values
                    q_target=sample['reward']+self.gamma*q_next
                
                #compute current q
                q_current=self.qnet(sample['state']).gather(-1,sample['action'])
                

                loss=nn.MSELoss()(q_current,q_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if idx%50==0:
                    print(f'training loss is:{loss}')
            

def reward_model(state,next_state):
    if state[0]>0:
        position_reward=state[0]
    else:
        position_reward=-state[0]

    speed_reward=math.fabs(state[1])
    
    if state[0]*next_state[0]>0 and math.fabs(state[0])<math.fabs(next_state[0]):
        speed_up_reward=(math.fabs(next_state[0])-math.fabs(state[0]))+0.5*math.fabs(state[0])
    else:
        speed_up_reward=0
    
    return 0.9*position_reward+speed_up_reward+speed_reward


def dqn_sample(epoch=10):
    trainer=DQNTrainer(train_epoch=2,buffer_size=200)
    trainer.init_model(2,3)
    for i in range(epoch):
        trainer.sample(sample_epoch=2)
        trainer.train()
    
            





                

        