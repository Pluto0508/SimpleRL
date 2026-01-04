import numpy as np

from games.grid import Grid

def value_iteration(grid,theta=0.01):
    policy=np.random.choice(grid.actions,size=(grid.size,grid.size))
    V=np.zeros((grid.size,grid.size))

    #value iteration
    while True:
        delta=0
        print(f"Value Iteration\n Value Function:\n {V}\n")
        for i in range(grid.size):
            for j in range(grid.size):
                v=V[i,j]
                action_values=[]
                for action in grid.actions:
                    next_state=grid.get_next_state((i,j),action)
                    reward=grid.get_reward(next_state)
                    action_values.append(reward+grid.gama*V[next_state])
                V[i,j]=max(action_values)
                delta=max(delta,abs(V[i,j]-v))
        
        if delta<theta:
            break
    
    #derive policy
    for i in range(grid.size):
        for j in range(grid.size):
            action_values=[]
            for action in grid.actions:
                next_state=grid.get_next_state((i,j),action)
                reward=grid.get_reward(next_state)
                action_values.append((reward+grid.gama*V[next_state],action))
            policy[i,j]=max(action_values,key=lambda x:x[0])[1]
    
    return policy,V

if __name__ == "__main__":
    grid=Grid()
    policy,V=value_iteration(grid)
    print("Optimal Value Function:")
    print(V)
    print("Optimal Policy:")
    print(policy)
            

