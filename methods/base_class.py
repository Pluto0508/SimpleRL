from collections import defaultdict
import numpy as np

class card_game_base:
    def __init__(self,env):
        self.env=env
        self.V=defaultdict(float)
        self.visit_counts=defaultdict(int)

    def simple_policy(self,state):
        player_value,_,_=state
        return 1 if player_value<17 else 0
    
    def print_statistics(self):
        print(f"\n评估统计:")
        print(f"总状态数: {len(self.V)}")
        print(f"平均访问次数: {np.mean(list(self.visit_counts.values())):.2f}")
        print(f"最大访问次数: {max(self.visit_counts.values())}")
        print(f"最小价值: {min(self.V.values()):.3f}")
        print(f"最大价值: {max(self.V.values()):.3f}")
        print(f"平均价值: {np.mean(list(self.V.values())):.3f}")

class cliff_base:
    def __init__(self,env):
        self.env=env
        self.q_table=np.zeros((env.rows,env.cols,len(env.actions)))
        self.rewards=[]

class ac_base:
    pass
    