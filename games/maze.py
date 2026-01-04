import numpy as np

class Maze:
    def __init__(self):
        self.size=4
        self.actions=['up', 'down', 'left', 'right']
        self.goal=(3,3)
        self.trap=(1,1)
        self.gama=0.9
        self.rewards=np.zeros((self.size,self.size))
        self.rewards[self.goal]=10
        self.rewards[self.trap]=-10
        self.default_reward = -1

    def get_next_state(self,state,action):
        if state==self.goal or state==self.trap:
            return state
        
        if np.random.random()<0.8:
            intend_action=action
        else:
            intend_action=np.random.choice([a for a in self.actions if a!=action])

        row,col=state
        if intend_action=='up':
            row=max(row-1,0)
        elif intend_action=='down':
            row=min(row+1,self.size-1)
        elif intend_action=='left':
            col=max(col-1,0)
        else:
            col=min(col+1,self.size-1)
        
        return (row,col)