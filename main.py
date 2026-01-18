import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import FrameStackObservation, ReshapeObservation

from pyboy import PyBoy

from collections import namedtuple, deque
import random
import math
from enum import Enum


class Actions(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    # DOWN = 4
    JUMP = 4
    FIRE = 5
    LEFT_PRESS = 6
    RIGHT_PRESS = 7
    JUMP_PRESS = 8
    LEFT_RUN = 9
    RIGHT_RUN = 10
    LEFT_JUMP = 11
    RIGHT_JUMP = 12

# 強制使用 GPU（如可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class MarioEnv(gym.Env):
    def __init__(self, pyboy):
        super().__init__()
        self.pyboy = pyboy
        self.mario = self.pyboy.game_wrapper
        
        # 根據 README.md 設定合理的最大值 (索引 0-11, 共 12 個)
        high = np.array([
            99,         # 0: lives_left
            999999,     # 1: score
            400,        # 2: time_left
            2601,       # 3: level_progress
            99,         # 4: coins
            4,          # 5: world
            3,          # 6: stage
            81,         # 7: mario x position
            134,        # 8: mario y position
            57,         # 9: game over
            4,          # 10: power state
            2,          # 11: mario jump routine
        ], dtype=np.float32)
        
        low = np.array([
            0,          # 0: lives_left
            0,          # 1: score
            0,          # 2: time_left
            250,        # 3: level_progress
            0,          # 4: coins
            1,          # 5: world
            1,          # 6: stage
            0,          # 7: mario x position
            0,          # 8: mario y position
            0,          # 9: game over
            0,          # 10: power state
            0,          # 11: mario jump routine
        ], dtype=np.float32)
        
        self.action_space = Discrete(len(Actions))
        self.observation_space = spaces.Box(low, high, shape=(len(low),), dtype=np.float32)
        
        self.mario.start_game()
        self.mario.set_lives_left(10)
        
        
        self.prev_progress = 0  # 新增：追蹤上一步進度
        self.prev_score = 0  # 新增：追蹤上一步分數

    def step(self, action):
        
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        

        # if action == Actions.NOOP.value:
        #     pass
        # elif action == Actions.LEFT_PRESS.value:
        #     self.pyboy.button_press("left")
        # elif action == Actions.RIGHT_PRESS.value:
        #     self.pyboy.button_press("right")
        # elif action == Actions.LEFT.value:
        #     self.pyboy.button("left")
        # elif action == Actions.RIGHT.value:
        #     self.pyboy.button("right")
        # elif action == Actions.UP.value:
        #     self.pyboy.button("up")
        # elif action == Actions.JUMP_PRESS.value:
        #     self.pyboy.button_press("b")
        #     self.pyboy.button_press("a")
        #     self.pyboy.button_press("right")
        # elif action == Actions.JUMP.value:
        #     self.pyboy.button("a")
        # elif action == Actions.FIRE.value:
        #     self.pyboy.button("b")
        # elif action == Actions.LEFT_RUN.value:
        #     self.pyboy.button_press("b")
        #     self.pyboy.button_press("left")
        # elif action == Actions.RIGHT_RUN.value:
        #     self.pyboy.button_press("b")
        #     self.pyboy.button_press("right")
        # elif action == Actions.LEFT_JUMP.value:
        #     self.pyboy.button_press("a")
        #     self.pyboy.button_press("left")
        # elif action == Actions.RIGHT_JUMP.value:
        #     self.pyboy.button_press("a")
        #     self.pyboy.button_press("right")
        
        # 定義動作映射字典：鍵為 action 值，值為 lambda 函式執行按鈕操作
        action_map = {
            Actions.NOOP.value: lambda: None,  # 無操作
            Actions.LEFT_PRESS.value: lambda: self.pyboy.button_press("left"),
            Actions.RIGHT_PRESS.value: lambda: self.pyboy.button_press("right"),
            Actions.LEFT.value: lambda: self.pyboy.button("left"),
            Actions.RIGHT.value: lambda: self.pyboy.button("right"),
            Actions.UP.value: lambda: self.pyboy.button("up"),
            # Actions.DOWN.value: lambda: self.pyboy.button("down"),
            Actions.JUMP_PRESS.value: lambda: (
                self.pyboy.button_press("b"),
                self.pyboy.button_press("a"),
                self.pyboy.button_press("right")
            ),  # 多個操作用 tuple 包起來
            Actions.JUMP.value: lambda: self.pyboy.button("a"),
            Actions.FIRE.value: lambda: self.pyboy.button("b"),
            Actions.LEFT_RUN.value: lambda: (
                self.pyboy.button_press("b"),
                self.pyboy.button_press("left")
            ),
            Actions.RIGHT_RUN.value: lambda: (
                self.pyboy.button_press("b"),
                self.pyboy.button_press("right")
            ),
            Actions.LEFT_JUMP.value: lambda: (
                self.pyboy.button_press("a"),
                self.pyboy.button_press("left")
            ),
            Actions.RIGHT_JUMP.value: lambda: (
                self.pyboy.button_press("a"),
                self.pyboy.button_press("right")
            ),
        }

        # 執行對應的動作（若 action 不在字典中，會拋 KeyError）
        action_map[action]()
        
        reward = self.calculate_reward()

        # advance emulator for frame_skip ticks, accumulate reward
        
        info = self.pyboy.game_wrapper
        # 根據 README.md 的 terminated 條件
        terminated = (self.live() == 0 or 
                     self.pyboy.memory[0xC0A4] != 0 or 
                     self.progress() == 2601)  # 死亡、game over 或達到終點
        truncated = self.time() == 0  # 時間耗盡

        state = self.get_obs()
        # extra terminal bonus
        if terminated:
            self.mario.reset_game()
            
        return state, reward, terminated, truncated, info
    
    def live(self):
        return self.mario.lives_left
    
    def score(self):
        return self.mario.score
    
    def time(self):
        return self.mario.time_left
    
    def progress(self):
        return self.mario.level_progress
    
    def coins(self):
        return self.mario.coins
    
    def world(self):
        return self.mario.world[0]
    
    def stage(self):
        return self.mario.world[1]
    
    def calculate_reward(self):
        reward = 0.0
        
        # 1. 進度獎勵 (最重要) - 鼓勵向右前進
        current_progress = self.progress()
        progress_delta = current_progress - self.prev_progress
        reward += progress_delta * 0.1  # 每前進一個單位給 0.1 分
        self.prev_progress = current_progress
        
        # 2. 分數獎勵 - 鼓勵吃金幣、踩敵人
        current_score = self.score()
        score_delta = current_score - self.prev_score
        reward += score_delta * 0.001  # 分數變化轉換為獎勵
        self.prev_score = current_score
        
        # 3. 時間懲罰 - 避免拖延
        # reward -= 0.01  # 每 tick 輕微懲罰
        
        # 修正後
        if not hasattr(self, 'prev_lives'):
            self.prev_lives = self.live()

        if self.live() < self.prev_lives:
            reward -= 50.0  # 失去生命時扣分
        self.prev_lives = self.live()
        
        # 5. 站著不動懲罰 (檢測 x 位置是否改變)
        current_x = self.pyboy.memory[0xC202]
        if not hasattr(self, 'prev_x'):
            self.prev_x = current_x
        
        if current_x == self.prev_x:
            reward -= 0.05  # 沒移動就扣分
        self.prev_x = current_x
        
        # 6. 金幣獎勵
        if not hasattr(self, 'prev_coins'):
            self.prev_coins = self.coins()
        
        coins_delta = self.coins() - self.prev_coins
        if coins_delta > 0:
            reward += 5.0 * coins_delta
        self.prev_coins = self.coins()
        
        return reward

    def reset(self, *, seed=42, options=None):
        super().reset(seed=seed)
    
        # 初始化所有追蹤變數
        self.prev_progress = self.progress()
        self.prev_score = self.score()
        self.prev_x = self.pyboy.memory[0xC202]
        self.prev_coins = self.coins()
        self.prev_lives = self.live()  # 初始化 prev_lives
        
        state = self.get_obs()
        info = self.pyboy.game_wrapper
        return state, info
    
    def get_obs(self):
        # 根據 README.md，返回 12 個觀察值 (索引 0-11)
        return np.array([
            self.live(),                # 0: lives_left
            self.score(),               # 1: score
            self.time(),                # 2: time_left
            self.progress(),            # 3: level_progress
            self.coins(),               # 4: coins
            self.world(),               # 5: world
            self.stage(),               # 6: stage
            self.pyboy.memory[0xC202],  # 7: mario x position
            self.pyboy.memory[0xC201],  # 8: mario y position
            self.pyboy.memory[0xC0A4],  # 9: game over
            self.pyboy.memory[0xFF99],  # 10: power state
            self.pyboy.memory[0xC207],  # 11: mario jump routine
        ], dtype=np.float32)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

pyboy = PyBoy("rom.gb", window="SDL2", sound_volume=0)
env = MarioEnv(pyboy)
print(env.observation_space.shape)

BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05          # 保留 5% 隨機
EPS_DECAY = 5000000      # 約 120~150 萬步才降到 0.1 左右
TAU = 0.001
LR = 6.25e-5

state, info = env.reset()
print("Initial observation:", state)

n_observations = len(state)
n_actions = env.action_space.n

print("Observation space:", n_observations)
print("Action space:", n_actions)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[np.random.randint(n_actions)]], device=device, dtype=torch.long)
            

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
    
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while pyboy.tick():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        if done:
            break
    pyboy.stop()

