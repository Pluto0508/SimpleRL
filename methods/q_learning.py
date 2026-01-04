import numpy as np
import torch

from games.cliff import CliffWalkingEnv
from methods.base_class import cliff_base
from model.QNetwork import QNet
from model.buffer import PPOBuffer
from utils.utils import *


class Q_learning(cliff_base):
    def __init__(self, env):
        super().__init__(env)
    
    def q_learning(self,episodes=500,alpha=0.1,gamma=0.9,epsilon=0.1):
        
        for episode in range(episodes):
            state=self.env.reset()
            total_rewards=0
            done=False

            while not done:
                if np.random.random()<epsilon:
                    action=np.random.choice(len(self.env.actions))
                else:
                    action=np.argmax(self.q_table[state[0],state[1]])

                next_state,reward,done=self.env.step(action)
                total_rewards+=reward

                #update q_table
                next_best_action=np.argmax(self.q_table[next_state[0],next_state[1]])
                td_target=reward+gamma*self.q_table[next_state[0],next_state[1],next_best_action]
                td_error=td_target-self.q_table[state[0],state[1],action]
                self.q_table[state[0],state[1],action]+=alpha*td_error

                state=next_state
            
            self.rewards.append(total_rewards)
        
        return self.q_table,self.rewards

class DQLTrainer:
    def __init__(self,train_epoch=2,buffer_size=100):
        self.qnet=None
        self.qnet_target=None
        self.buffer=None

        self.gamma=0.6
        self.buffer_size=buffer_size
        self.train_epoch=train_epoch

        self.optimizer_qnet=None
        
    
    def init_model(self,state_dim,action_dim):
        self.qnet=QNet(state_dim,action_dim)
        self.qnet_target=deep_clone_model(self.qnet)
        self.optimizer_qnet=torch.optim.Adam(self.qnet.parameters(),lr=0.001)
        self.buffer=PPOBuffer(self.buffer_size)
    
    def soft_update(self,alpha=0.05):
        for param_target,param in zip(self.qnet_target.parameters(),self.qnet.parameters()):
            param_target.data.copy_(alpha*param.data+(1-alpha)*param_target.data)
    
    def train(self,batch_size=4):
        self.qnet.train()

        for i in range(self.train_epoch):
            self.buffer.prepare(shuffle=True)

            for idx,sample in enumerate(self.buffer.sample(batch_size)):
                for key,value in sample.items():
                    if key in ['state','next_state','action','reward']:
                        sample[key]=torch.stack(value)
            
                #target q
                with torch.no_grad():
                    q_next_state=self.qnet(sample['next_state'])
                    action_next=self.qnet.make_action(q_next_state)
                    q_next_state=self.qnet_target(sample['next_state']).gather(-1,action_next)
                    q_target=sample['reward']+self.gamma*q_next_state
                
                #current q
                q_current=self.qnet(sample['state']).gather(-1,sample['action'])
                loss=torch.nn.MSELoss()(q_current,q_target)

                #update q
                self.optimizer_qnet.zero_grad()
                loss.backward()
                self.optimizer_qnet.step()

                if idx%50==0:
                    print(f'loss:{loss}')
            
            self.soft_update()




            







    

