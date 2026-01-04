import torch
import torch.nn as nn
import gymnasium as gym
from pettingzoo.mpe import simple_speaker_listener_v4

from methods.q_learning import DQLTrainer
from utils.utils import *

class IQLTrainer:
    def __init__(self,train_epoch=4,agent_names=None):
        self.agents=None
        self.agent_nums=None
        self.env=None
        self.agent_names=agent_names

        self.train_epoch=train_epoch
    
    def init_model(self,state_dim,action_dim):
        self.env=simple_speaker_listener_v4.parallel_env(
            continuous_actions=False,
            max_cycles=25,
            render_mode='human'  
        )
        self.agent_nums=2
        self.agents=[]
        for i in range(self.agent_nums):
            agent=DQLTrainer()
            agent.init_model(state_dim[i],action_dim[i])
            self.agents.append(agent)
    
    def train(self,batch_size=4):
        for i in range(self.train_epoch):
            state=self.env.reset()
            state=state[0]
            episode_reward=0
            done=False

            while not done:
                actions={}
                for i in range(self.agent_nums):
                    q=self.agents[i].qnet(torch.tensor(state[self.agent_names[i]]))
                    action=self.agents[i].qnet.make_action(q).detach().numpy()
                    actions[self.agent_names[i]]=action
                
                next_state,reward,done,_,_=self.env.step(actions)
                self.env.render()
                episode_reward+=0

                for i in range(self.agent_nums):
                    self.agents[i].buffer.push(
                        state[self.agent_names[i]],
                        actions[self.agent_names[i]],
                        next_state[self.agent_names[i]],
                        reward[self.agent_names[i]],
                        done[self.agent_names[i]],
                        None,
                        None
                    )

                    self.agents[i].train()
                
                
                state=next_state
                
                print(episode_reward)


def iql_sample(epoch=2):
    trainer=IQLTrainer(agent_names=['speaker_0','listener_0'])
    trainer.init_model(state_dim=[3,11],action_dim=[3,5])
    for i in range(epoch):
        trainer.train()

