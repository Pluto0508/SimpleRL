class Grid:
    def __init__(self):
        self.size = 3
        self.actions = ['up', 'down', 'left', 'right']
        self.center = (1, 1)
        self.traps=(1,2)
        self.gama = 0.9
        
    def get_next_state(self, state, action):
        row, col = state
        if action == 'up':
            row = max(row - 1, 0)
        elif action == 'down':
            row = min(row + 1, self.size - 1)
        elif action == 'left':
            col = max(col - 1, 0)
        elif action == 'right':
            col = min(col + 1, self.size - 1)
        return (row, col)
    
    def get_reward(self, state):
        if state == self.center:
            return 1
        if state==self.traps:
            return -1
        return -0.1