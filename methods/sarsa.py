from games.cliff import CliffWalkingEnv
from base_class import cliff_base

import numpy as np

class Sarsa(cliff_base):
    def __init__(self, env):
        super().__init__(env)
    
    def select_action(self,state,epsilon):

        if np.random.random()<epsilon:
            action=np.random.choice(len(self.env.actions))
        else:
            action=np.argmax(self.q_table[state[0],state[1]])

        return action
    
    def sarsa(self,episodes=500,alpha=0.1,gamma=0.9,epsilon=0.1):
        
        for episode in range(episodes):
            state=self.env.reset()
            done=False
            total_rewards=0
            
            action=self.select_action(state,epsilon)
            
            while not done:
                next_state,reward,done=self.env.step(action)
                total_rewards+=reward

                #update q_table
                next_action=self.select_action(next_state,epsilon)
                td_target=reward+gamma*self.q_table[next_state[0],next_state[1],next_action]
                td_error=td_target-self.q_table[state[0],state[1],action]
                self.q_table[state[0],state[1],action]+=alpha*td_error

                state=next_state
                action=next_action
            
            self.rewards.append(total_rewards)
        
        return self.q_table,self.rewards

                



