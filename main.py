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
        
        # 假設新觀察形狀為 (3, 7)，重新定義 low 和 high
        # 基於原來的範圍，分組為 3 行，每行 7 個元素
        # 注意：調整值以匹配您的實際範圍（例如，rom_table 的 high 可能不同）
        # 統一為 (3, 7)，補充 rom_table 至 7 個元素（例如，添加 0 或移除一個）
        low = np.array([
            [0, 0, 0, 250, 0, 1, 1],      # game_state: lives, score, time, progress, coins, world, stage
            [0, 0, 0, 0, 0, 0, 0],        # mario_state: x_pos, y_pos, game_over, power_state, jump_routine, pose, timer=
            [0, 0, 0, 0, 0, 0, 0]         # rom_table: 記憶體值
        ], dtype=np.float32)
        
        high = np.array([
            [99, 999999, 400, 2601, 99, 4, 3],  # game_state (7)
            [81, 134, 57, 4, 2, 5, 255],        # mario_state (7)
            [255, 255, 255, 255, 255, 255, 255]  # rom_table (補充至 7)
        ], dtype=np.float32)
        
        self.action_space = Discrete(len(Actions))
        self.observation_space = spaces.Box(low, high,  shape=low.shape, dtype=np.float32)
        
        self.mario.start_game()
        self.mario.set_lives_left(10)
        
        # 初始化 reward 追蹤變數
        self.prev_progress = 0
        self.prev_power_state = 0

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
        # terminated(Rest)：生命值=0 或 時間=0 時重置遊戲
        terminated = (self.live() == 0 or self.time() == 0)
        # terminated(Stop)：進度=2601 時停止遊戲
        truncated = (self.progress() == 2601)
        
        
        truncated = False  # 不使用 truncated 信號

        state = self.get_obs()
        
        if truncated:
            self.pyboy.stop()
            
        # 根據終止類型決定是否重置
        if terminated:
            self.mario.reset_game()
            self.mario.set_lives_left(10)
        # 如果是 terminated_stop 則不重置，遊戲真正結束
            
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
        
        # 根據 README.md 的 Reward System
        
        # 1. Game Over 懲罰: if pyboy.memory[0xC0A4] == 0x39
        if self.pyboy.memory[0xC0A4] == 0x39:
            reward -= 100
        
        # 2. 死亡/重生懲罰: if pyboy.memory[0xFFA6] == 0x90
        if self.pyboy.memory[0xFFA6] == 0x90:
            reward -= 100
        
        # 3. 進度獎勵: if level_progress + 1
        current_progress = self.progress()
        if not hasattr(self, 'prev_progress'):
            self.prev_progress = current_progress
        
        progress_delta = current_progress - self.prev_progress
        if progress_delta > 0:
            reward += progress_delta  # 每前進一個單位給 +1 分
        self.prev_progress = current_progress
        
        # 4. 變大/超級球獎勵: if pyboy.memory[0xFF99] == 0x02
        if not hasattr(self, 'prev_power_state'):
            self.prev_power_state = self.pyboy.memory[0xFF99]
        
        current_power_state = self.pyboy.memory[0xFF99]
        # 當從小變大時（0x00 → 0x02）給獎勵
        if current_power_state == 0x02 and self.prev_power_state != 0x02:
            reward += 100
        self.prev_power_state = current_power_state
        
        return reward

    def reset(self, *, seed=42, options=None):
        super().reset(seed=seed)
        
        # 添加重置
        self.mario.set_lives_left(10)
        
        # 初始化追蹤變數
        self.prev_progress = self.progress()
        self.prev_power_state = self.pyboy.memory[0xFF99]
        
        state = self.get_obs()
        info = self.mario
        return state, info
    
    def get_obs(self):
        game_state = [self.live(), self.score(), self.time(), self.progress(), self.coins(), self.world(), self.stage()]  # 7 個
        mario_state = [self.pyboy.memory[0xC202], self.pyboy.memory[0xC201], self.pyboy.memory[0xC0A4], 
                    self.pyboy.memory[0xFF99], self.pyboy.memory[0xC207], self.pyboy.memory[0xC203], self.pyboy.memory[0xFFA6]]  # 7 個
        rom_table = [self.pyboy.memory[0xD100], self.pyboy.memory[0xD101], self.pyboy.memory[0xD102], 
                    self.pyboy.memory[0xD103], self.pyboy.memory[0xD106], self.pyboy.memory[0xD108], 0]  # 補充至 7 個（添加 0）
        return np.array([game_state, mario_state, rom_table], dtype=np.float32)  # 現在形狀一致 (3, 7)


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
# print(env.observation_space.shape)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05          # 保留 5% 隨機
EPS_DECAY = 5000000      # 約 120~150 萬步才降到 0.1 左右
TAU = 0.001
LR = 6.25e-5

state, info = env.reset()
# print("Initial observation:", env.reset()[0])
# print("Observation shape:", env.reset()[0].shape)

n_observations = env.reset()[0].shape[0]
n_actions = env.action_space.n

# print("Observation space:", n_observations)
# print("Action space:", n_actions)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # 輸入: (batch, 3, 1, 7)
        self.conv1 = nn.Conv2d(n_observations, 16, kernel_size=(1, 3), stride=1)  # 輸出: (batch, 16, 1, 5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), stride=1)  # 輸出: (batch, 32, 1, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 1 * 3, 128)  # 壓平後的全連接
        self.fc2 = nn.Linear(128, n_actions)   # 輸出 Q 值

    def forward(self, x):
        x = x.view(-1, 3, 1, 7)  # Reshape 為 (batch, 3, 1, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 初始化網路
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(100000)  # 經驗回放緩衝區

# print(policy_net)
# print(target_net)

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
    
# 添加訓練迴圈
for episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    while pyboy.tick():  # 每集 100 步
        action = select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if done:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

        # 儲存經驗
        memory.push(state, action, next_state, reward)
        state = next_state

        # 優化模型
        optimize_model()

        if done:
            break
        
        # print(state)

        # 更新目標網路
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)


