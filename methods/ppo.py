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

from model.AC import ActorCritic
from model.buffer import PPOBuffer

class PPOTrainer:
    def __init__(self,
                 agent: Optional[ActorCritic]=None,
                 env: Optional[Env]=None,
                 train_epoch=1000,
                 buffer_size=100):
        
        self.agent=agent
        self.train_epoch=train_epoch
        self.env=env
        self.buffer=PPOBuffer(buffer_size)

        self.optimizer=None
        self.gamma=0.9
        self.clip_epsilon=0.2

        self.actor_loss=[]
        self.critic_loss=[]
    
    def init_model(self,input_dim,output_dim):
        self.agent=ActorCritic(input_dim=input_dim,output_dim=output_dim).float()
        self.env=gym.make('MountainCar-v0',render_mode='human')
        self.optimizer=optim.Adam(self.agent.parameters(),lr=0.001)
        return self
    
    def sample(self,sample_epoch):
        self.buffer.clear()
        self.agent.eval()

        while self.buffer.current_size<self.buffer.size:
            state=self.env.reset()
            state=torch.tensor(state[0])
            with torch.no_grad():
                for _ in range(sample_epoch):
                    res=self.agent(state)
                    action=self.agent.make_action(res)
                    next_state,reward,done,_,_=self.env.step(action.detach().numpy())
                    action=action.detach().clone().requires_grad_(False)
                    next_state=torch.tensor(next_state)
                    reward=torch.tensor(reward)
                    self.env.render()
                    self.buffer.push(state.detach().clone(),action,next_state.detach().clone(),reward,done,torch.log(res['actor_prob']).detach().clone(),res['critic_value'].detach().clone())
                    if done:
                        break
                    state=next_state
    
    def compute_return(self,rewards):
        R=0
        returns=[0]*len(rewards)
        for i in range(len(rewards)-1,-1,-1):
            R=rewards[i]+self.gamma*R
            returns[i]=R

        return returns

    def train(self,batch_size=4):
        self.agent.train()
        
        for _ in range(self.train_epoch):

            for idx,sample in enumerate(self.buffer.sample(batch_size)):
                for key,value in sample.items():
                    if key in ['state','next_state','action','reward','log_prob','critic_value']:
                        sample[key]=torch.stack(value)
                    
                #compute return
                returns=self.compute_return(sample['reward'])
                returns=torch.tensor(returns)

                #advantage
                advantage=returns-sample['critic_value']
                
                #forward
                res_new=self.agent(sample['state'])
                action_new=self.agent.make_action(res_new)
                log_prob_new=torch.log(res_new['actor_prob'])
                ratios=torch.exp(log_prob_new-sample['log_prob'])
                ratios=torch.gather(ratios,1,action_new.detach().unsqueeze(1))

                #ppo clip
                surr1=ratios*advantage
                surr2=torch.clamp(ratios,1-self.clip_epsilon,1+self.clip_epsilon)*advantage
                actor_loss=-torch.min(surr1,surr2).mean()

                #critic_loss
                critic_loss=nn.MSELoss()(res_new['critic_value'],returns.unsqueeze(1))

                #entropy
                dist = torch.distributions.Categorical(logits=res_new['actor_output'])
                entropy_loss=dist.entropy().mean()

                #total
                self.actor_loss.append(actor_loss)
                self.critic_loss.append(critic_loss)
                total_loss=actor_loss+0.5*critic_loss-0.01*entropy_loss
                print(total_loss)
                #step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(),0.5)
                self.optimizer.step()
                

    def evaluate(self):
        pass

    def plot_res(self):
        x=list(range(len(self.critic_loss)))
        print(self.actor_loss)
        print(self.critic_loss)


def ppo_sample():
    trainer=PPOTrainer(train_epoch=1)
    trainer.init_model(input_dim=2,output_dim=3)
    
    for i in range(30):
        trainer.sample(sample_epoch=1)
        trainer.train()
    trainer.env.close()