import numpy as np
import logging

from games.maze import Maze
    
def policy_evaluation(maze,theta=0.01):
    policy=np.random.choice(maze.actions,size=(maze.size,maze.size))
    V=np.zeros((maze.size,maze.size))
    iteration=0

    while True:
        print(f"Policy Iteration - Iteration {iteration}\n Policy:\n {policy}\n Value Function:\n {V}\n")
        #policy evaluation
        while True:
            delta=0
            for i in range(maze.size):
                for j in range(maze.size):
                    v=V[i,j]
                    action=policy[i,j]
                    next_state=maze.get_next_state((i,j),action)
                    reward=maze.get_reward(next_state)
                    V[i,j]=reward+maze.gama*V[next_state]
                    delta=max(delta,abs(V[i,j]-v))
            
            if delta<theta:
                break
        
        #policy improvement
        policy_stable=True
        for i in range(maze.size):
            for j in range(maze.size):
                old_action=policy[i,j]
                action_values=[]
                for action in maze.actions:
                    next_state=maze.get_next_state((i,j),action)
                    reward=maze.get_reward(next_state)
                    action_values.append(reward+maze.gama*V[next_state])
                best_action=maze.actions[np.argmax(action_values)]
                policy[i,j]=best_action
                if old_action!=best_action:
                    policy_stable=False
        
        iteration+=1
        
        if policy_stable:
            break

    return policy,V

if __name__=="__main__":
    maze=Maze()
    optimal_policy,optimal_value=policy_evaluation(maze)
    print(f"optimal_policy: {optimal_policy} \n optimal_value: {optimal_value}")