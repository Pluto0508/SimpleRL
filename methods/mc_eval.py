from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from games.card_game import BlackjackEnv
from .base_class import card_game_base

class MC_Eval(card_game_base):
    def __init__(self,env,policy=None,lamabda=0.9):
        super().__init__(env)

        self.policy=policy if policy is not None else self.simple_policy
        self.returns=defaultdict(list)
        self.visit_counts=defaultdict(int)
        self.lamabda=lamabda
    
    def generate_trajectory(self):
        trajectory=[]
        state=self.env.reset()

        while not self.env.done:
            action=self.policy(state)
            next_state,reward,done,_=self.env.step(action)
            trajectory.append((state,action,reward))
            state=next_state
        
        return trajectory
    
    def mc_eval(self,num_episodes=1000):
        for _ in range(num_episodes):
            trajectory=self.generate_trajectory()
            G=0

            for t in range(len(trajectory)-1,-1,-1):
                state,action,reward=trajectory[t]
                G=self.lamabda*G+reward
                self.returns[state].append(G)
                self.V[state]=np.mean(self.returns[state])
                self.visit_counts[state]+=1

            if _%100==0:
                print(f'{_} episodes completed.')
    
    
def mc_eval_sample():
    env=BlackjackEnv()
    sample=MC_Eval(env)
    sample.mc_eval()
    sample.print_statistics()
    #print(sample.V)





        
