from collections import defaultdict
import numpy as np

from games.card_game import BlackjackEnv
from .base_class import card_game_base

class TD_learning(card_game_base):
    def __init__(self,env,alpha=0.01,decay_factor=0.9):
        super().__init__(env)
        
        self.env=env
        self.alpha=alpha
        self.decay_factor=decay_factor
        self.returns=defaultdict(float)

    def TD_zero_update(self,state,reward,next_state,done):

        if done:
            td_target=reward
        else:
            td_target=reward+self.decay_factor*self.V[next_state]
        
        td_error=td_target-self.V[state]
        self.V[state]+=self.alpha*td_error
    
    def TD_n_update(self,state,reward,next_state,done):
        pass

    def TD_zero(self,max_iters=20):

        for _ in range(max_iters):
            #prepare
            state=self.env.reset()
            done=False

            while not done:
                action=self.simple_policy(state)
                next_state,reward,done,_=self.env.step(action)
                self.visit_counts[next_state]+=1
                self.TD_zero_update(state,reward,next_state,done)

                state=next_state

def td_zero_sample():
    env=BlackjackEnv()
    model=TD_learning(env)
    model.TD_zero()
    model.print_statistics()

    


