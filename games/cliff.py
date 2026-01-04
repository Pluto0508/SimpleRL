class CliffWalkingEnv:
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = [(3, i) for i in range(1, 11)]
        self.actions = ['up', 'right', 'down', 'left']
        self.state = self.start
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        row, col = self.state
        
        if action == 'up':
            row = max(row - 1, 0)
        elif action == 'right':
            col = min(col + 1, self.cols - 1)
        elif action == 'down':
            row = min(row + 1, self.rows - 1)
        elif action == 'left':
            col = max(col - 1, 0)
            
        self.state = (row, col)
        
        if self.state in self.cliff:
            reward = -100
            done = True
            self.state = self.start
        elif self.state == self.goal:
            reward = 0
            done = True
        else:
            reward = -1
            done = False
            
        return self.state, reward, done