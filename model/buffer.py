from typing import Optional
import numpy as np
import logging
import torch
import random
#set seed
class Buffer:
    def __init__(self,buffer_size=100):
        self.size=buffer_size
        self.current_size=0
    
    def clear(self):
        pass
    
    def push(self,data):
        pass
    
    def prepare(self,shuffle=False):
        pass

    def sample(self):
        pass

class DQNBuffer(Buffer):
    def __init__(self,buffer_size,container: Optional[list]=None):
        super(DQNBuffer,self).__init__(buffer_size)
        self.container=[]
    
    def clear(self):
        self.container=[]
        self.current_size=0
        return self

    def push(self,data):
        if self.size<=self.current_size:
            logging.warning('Out of size of buffer! Randomly drop a sample point!')
            idx=np.random.randint(low=0,high=self.size)
        else:
            self.container.append(data)
            self.current_size+=1
        return self
    
    def prepare(self,shuffle=False):
        self.container=np.array(self.container)
        if shuffle:
            slice=np.random.choice(a=len(self.container),size=len(self.container),replace=False)
            self.container=self.container[slice]
        self.container=self.container.tolist()

    def sample(self):
        for i in range(0,self.size):
            yield self.container[i]
    
    def __len__(self):
        return len(self.container)
    
    def __getitem__(self,idx):
        return self.container[idx]

class PPOBuffer(Buffer):
    def __init__(self,buffer_size):
        super(PPOBuffer,self).__init__(buffer_size)
        self.state=[]
        self.action=[]
        self.next_state=[]
        self.reward=[]
        self.done=[]
        self.log_prob=[]
        self.res=[]
    
    def change_data(self,state,action,next_state,reward,done,log_prob,res,idx=None):
        if idx==None:
            self.state=state
            self.action=action
            self.next_state=next_state
            self.reward=reward
            self.done=done
            self.log_prob=log_prob
            self.res=res
        elif idx==-1:
            self.state.append(state)
            self.action.append(action)
            self.next_state.append(next_state)
            self.reward.append(reward)
            self.done.append(done)
            self.log_prob.append(log_prob)
            self.res.append(res)
    
    def clear(self):
        self.change_data([],[],[],[],[],[],[])
        self.current_size=0
        return self
    
    def push(self,state,action,next_state,reward,done,log_prob,res):
        if self.size<=self.current_size:
            raise ValueError('Out of size of buffer! Randomly drop a sample point!')
        else:
            self.change_data(state,action,next_state,reward,done,log_prob,res,idx=-1)
            self.current_size+=1
        return self
    
    def prepare(self,shuffle=False):
        if shuffle:
            random.shuffle(self.state)
            random.shuffle(self.action)
            random.shuffle(self.next_state)
            random.shuffle(self.reward)
            random.shuffle(self.done)
            random.shuffle(self.log_prob)
            random.shuffle(self.res)

    def sample(self,batch_size=4):
        for i in range(0,self.size-batch_size+1,batch_size):
            if i+batch_size<=self.size:
                yield {
                    'state':self.state[i:i+batch_size],
                    'action':self.action[i:i+batch_size],
                    'next_state':self.next_state[i:i+batch_size],
                    'reward':self.reward[i:i+batch_size],
                    'done':self.done[i:i+batch_size],
                    'log_prob':self.log_prob[i:i+batch_size],
                    'critic_value':self.res[i:i+batch_size]
                }


def test_buffer():
    buffer=Buffer(buffer_size=8)
    for i in range(8):
        buffer.push(i)
    
    buffer.prepare(shuffle=True)
    for item in buffer.sample():
        print(item)

    buffer.clear()
    print(len(buffer))

if __name__=='__main__':
    test_buffer()
