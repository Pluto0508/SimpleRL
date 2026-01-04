import numpy as np
import random
from collections import defaultdict

class BlackjackEnv:
    """21点游戏环境"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置游戏状态"""
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card(), self.draw_card()]
        self.done = False
        
        return self.get_state()
    
    def _create_deck(self):
        """创建一副扑克牌"""
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
    
    def draw_card(self):
        """从牌堆中抽一张牌"""
        if len(self.deck) == 0:
            self.deck = self._create_deck()
            random.shuffle(self.deck)
        return self.deck.pop()
    
    def get_card_value(self, card):
        """获取单张牌的值"""
        if card == 1:
            return 11  # A初始为11
        elif card >= 10:
            return 10
        else:
            return card
    
    def get_hand_value(self, hand):
        """计算手牌总值，处理A的两种情况"""
        value = sum(self.get_card_value(card) for card in hand)
        num_aces = sum(1 for card in hand if card == 1)
        
        while value > 21 and num_aces > 0:
            value -= 10  # 将A从11变为1
            num_aces -= 1
        
        return value
    
    def get_state(self):
        """获取当前状态：(玩家点数, 庄家明牌, 是否有可用A)"""
        player_value = self.get_hand_value(self.player_hand)
        dealer_showing = self.get_card_value(self.dealer_hand[0])
        usable_ace = 1 if any(card == 1 and self.get_card_value(card) == 11 
                    for card in self.player_hand) else 0
        
        return (player_value, dealer_showing, usable_ace)
    
    def step(self, action):
        """执行动作"""
        if self.done:
            raise ValueError("游戏已结束")
        
        if action == 1:  # 要牌
            self.player_hand.append(self.draw_card())
            player_value = self.get_hand_value(self.player_hand)
            
            if player_value > 21:  # 爆牌
                self.done = True
                return self.get_state(), -1, True, {}
            else:
                return self.get_state(), 0, False, {}
        
        else:  # 停牌
            return self.dealer_play()
    
    def dealer_play(self):
        """庄家按规则玩牌"""
        while self.get_hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self.draw_card())
        
        player_value = self.get_hand_value(self.player_hand)
        dealer_value = self.get_hand_value(self.dealer_hand)
        self.done = True
        
        if dealer_value > 21 or player_value > dealer_value:
            reward = 1
        elif player_value < dealer_value:
            reward = -1
        else:
            reward = 0
            
        return self.get_state(), reward, True, {}
