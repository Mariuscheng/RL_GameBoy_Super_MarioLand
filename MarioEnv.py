from pyboy import PyBoy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import FlattenObservation, NumpyToTorch

from enum import Enum
import numpy as np

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
from PIL import Image

# from cyberbrain import trace

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    # "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(device)

class Actions(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    JUMP = 5
    FIRE = 6
    LEFT_PRESS = 7
    RIGHT_PRESS = 8
    JUMP_PRESS = 9
    LEFT_RUN = 10
    RIGHT_RUN = 11
    LEFT_JUMP = 12
    RIGHT_JUMP = 13
    
# class ARROW_Function(Enum):
#     NOOP = 0
#     LEFT = 1
#     RIGHT = 2
#     UP = 3
#     DOWN = 4
#     LEFT_PRESS = 5
#     RIGHT_PRESS = 6
    
# class A_FUNCTION(Enum):
#     NOOP = 0
#     BUTTON_A = 1  # Jump

# class B_FUNCTION(Enum):
#     NOOP = 0
#     BUTTON_B = 1  # RUN OR FIRE

# Create action groups for easier reference
# DIRECTION_ACTIONS = [ARROW_Function.NOOP, ARROW_Function.LEFT, ARROW_Function.RIGHT, ARROW_Function.UP, ARROW_Function.DOWN, ARROW_Function.LEFT_PRESS, ARROW_Function.RIGHT_PRESS]
# A_FUNCTION_ACTIONS = [A_FUNCTION.NOOP, A_FUNCTION.BUTTON_A]
# B_FUNCTION_ACTIONS = [B_FUNCTION.NOOP, B_FUNCTION.BUTTON_B]


class MarioEnv(gym.Env):
    
    def __init__(self, pyboy):
        # super().__init__(PyBoy)
        self.pyboy = pyboy
    
        # Define action space using the number of actions in each group
        # self.n_directions = len(DIRECTION_ACTIONS)
        # self.a_functions = len(A_FUNCTION_ACTIONS)
        # self.b_functions = len(B_FUNCTION_ACTIONS)
        # self.action_space = MultiDiscrete(np.array([self.n_directions, self.a_functions, self.b_functions]), seed=42)
        self.action_space = Discrete(len(Actions), seed=42)

        # Define observation space
        self.observation_space = Box(low=0, high=255, shape=(16, 20), dtype=np.int32) # 假設距離的值範圍是 0-255
        

    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        # Move the agent
        if action == Actions.NOOP.value:
            pass
        elif action == Actions.LEFT_PRESS.value:
            self.pyboy.button_press("left")
        elif action == Actions.RIGHT_PRESS.value:
            self.pyboy.button_press("right")
        elif action == Actions.LEFT.value:
            self.pyboy.button("left")
        elif action == Actions.RIGHT.value:
            self.pyboy.button("right")
        elif action == Actions.UP.value:
            self.pyboy.button("up")
        # elif action == Actions.DOWN.value:
        #     self.pyboy.button("down")   
        elif action == Actions.JUMP_PRESS.value:
            self.pyboy.button_press("b")
            self.pyboy.button_press("a")
            self.pyboy.button_press("right")
        elif action == Actions.JUMP.value:
            self.pyboy.button("a")   
        elif action == Actions.FIRE.value:
            self.pyboy.button("b")
        elif action == Actions.LEFT_RUN.value:
            self.pyboy.button_press("b")
            self.pyboy.button_press("left")
        elif action == Actions.RIGHT_RUN.value:
            self.pyboy.button_press("b")
            self.pyboy.button_press("right")
        elif action == Actions.LEFT_JUMP.value:
            self.pyboy.button_press("a")
            self.pyboy.button_press("left")
        elif action == Actions.RIGHT_JUMP.value:
            self.pyboy.button_press("a")
            self.pyboy.button_press("right")

        # # Move the agent
        # if action == ARROW_Function[0].NOOP.value:
        #     pass
        # elif action == ARROW_Function.LEFT_PRESS.value:
        #     self.pyboy.button_press("left")
        # elif action == ARROW_Function.RIGHT_PRESS.value:
        #     self.pyboy.button_press("right")
        # elif action == ARROW_Function.LEFT.value:
        #     self.pyboy.button("left")
        # elif action == ARROW_Function.RIGHT.value:
        #     self.pyboy.button("right")
        # elif action == ARROW_Function.UP.value:
        #     self.pyboy.button("up")
        # elif action == ARROW_Function.DOWN.value:
        #     self.pyboy.button("down")
            
        # # A button actions for the agent    
        # if action == A_FUNCTION.NOOP.value:
        #     pass
        # elif action == A_FUNCTION.BUTTON_A.value:
        #     self.pyboy.button_press("a")
        
        # # B button actions for the agent    
        # if action == B_FUNCTION.NOOP.value:
        #     pass
        # elif action == B_FUNCTION.BUTTON_B.value:
        #     self.pyboy.button("b")
            
        self.pyboy.tick()
                
        terminated = self.pyboy.game_wrapper.level_progress > 2601
        
        
        reward=self.pyboy.game_wrapper.score

        state=self._get_obs()
        
        
        info = {}
        truncated = False

        return state, reward, terminated, truncated, info

    
    def reset(self, seed=42, options=None):
        super().reset(seed=seed)
        #self.pyboy.game_wrapper.reset_game()

        state=self._get_obs()
            
        info = {}
            
        return state, info

    
    def render(self):
        self.pyboy.tick()

    
    def close(self):
        self.pyboy.stop()

    
    def _get_obs(self):
        return self.pyboy.game_area()
        



    
pyboy = PyBoy("rom.gb", window="SDL2", sound_volume=0) 
env = MarioEnv(pyboy)

# flatten the array
env = FlattenObservation(env)
env = NumpyToTorch(env)

mario = pyboy.game_wrapper

mario.start_game(world_level=(1,1))

# mario.game_area_mapping(pyboy.game_wrapper.mapping_minimal, 0)

state, info = env.reset()

print(state.to(device))
# tensor([339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 
#         320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 
#         300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 321, 322, 321, 322, 323, 300, 300, 300, 
#         300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 324, 325, 326, 325, 326, 327, 300, 300, 300, 
#         300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 
#         300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 
#         300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 
#         300, 300, 300, 300, 300, 300, 300, 300, 310, 350, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 
#         300, 300, 300, 300, 300, 300, 300, 310, 300, 300, 350, 300, 300, 300, 300, 300, 300, 300, 300, 300, 
#         300, 300, 300, 300, 300, 129, 310, 300, 300, 300, 300, 350, 300, 300, 300, 300, 300, 300, 300, 300, 
#         300, 300, 300, 300, 300, 310, 300, 300, 300, 300, 300, 300, 350, 300, 300, 300, 300, 300, 300, 300, 
#         300, 300, 310, 350, 310, 300, 300, 300, 300, 306, 307, 300, 300, 350, 300, 300, 300, 300, 300, 300, 
#         300, 368, 369, 300,   0,   1, 300, 306, 307, 305, 300, 300, 300, 300, 350, 300, 300, 300, 300, 300, 
#         310, 370, 371, 300,  16,  17, 300, 305, 300, 305, 300, 300, 300, 300, 300, 350, 300, 300, 300, 300,
#         352, 352, 352, 352, 352, 352, 352, 352, 352, 352, 352, 352, 352, 352, 352, 352, 352, 352, 352, 352, 
#         353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353], device='cuda:0', dtype=torch.int32)

# Get the total 320 number of elements in the tensor
print("flatten observation space :", torch.tensor(env.observation_space.shape)) #(320, )


print("action_space.sample :", torch.tensor(env.action_space.sample())) #1


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# class ReplayMemory(object):

#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)

#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(Transition(*args))

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)

# 優先級經驗回放
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def push(self, transition, td_error):
        max_priority = max(self.priorities) if self.memory else 1.0
        self.memory.append(transition)
        self.priorities.append(max_priority ** self.alpha)

        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, torch.tensor(weights, dtype=torch.float32, device=device), indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) ** self.alpha

    def __len__(self):
        return len(self.memory)

# class DQN(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.layer1 = nn.Linear(n_observations,  512)
#         self.act1 = nn.Tanh()
#         self.layer2 = nn.Linear(512, 256)
#         self.act2 = nn.Tanh()
#         self.layer3 = nn.Linear(256, 128)
#         self.act3 = nn.Tanh()
#         self.layer4 = nn.Linear(128, n_actions)

#     def forward(self, x):
#         x = F.relu(self.act1(self.layer1(x)))
#         x = F.relu(self.act2(self.layer2(x)))
#         x = F.relu(self.act3(self.layer3(x)))
#         return self.layer4(x)

# 定義分布式 Q 網絡
class RainbowDQN(nn.Module):
    def __init__(self, n_observations, n_actions, atom_size, v_min, v_max):
        super(RainbowDQN, self).__init__()
        self.n_actions = n_actions
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, atom_size).to(device)

        self.fc1 = nn.Linear(n_observations, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_actions * atom_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, self.n_actions, self.atom_size)
        x = F.softmax(x, dim=-1)
        return x

    def get_q_values(self, x):
        dist = self.forward(x)
        q_values = torch.sum(dist * self.support, dim=2)
        return q_values        

# 定義參數
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.0001
ATOM_SIZE = 51
V_MIN = -10
V_MAX = 10
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005

# Get number of actions from gym action space
n_actions = env.action_space.n
print("n_actions :", n_actions) #14

n_observations = torch.tensor(env.observation_space.shape)
print("n_observations :", n_observations) #320

channel = state.ndim
print("n_observations chennel :", channel) #1

# policy_net = DQN(n_observations, n_actions).to(device)
# target_net = DQN(n_observations, n_actions).to(device)

policy_net = RainbowDQN(n_observations, n_actions, ATOM_SIZE, V_MIN, V_MAX).to(device)
target_net = RainbowDQN(n_observations, n_actions, ATOM_SIZE, V_MIN, V_MAX).to(device)

print(f"Model structure: {policy_net}\n\n")
print("policy_net :", policy_net)
print("target_net :", target_net)

for name, param in policy_net.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

target_net.load_state_dict(policy_net.state_dict())
torch.save(policy_net.state_dict(), 'model_weights.pth')
target_net.eval()

# optimizer = torch.optim.SGD(policy_net.parameters(), lr=LR, momentum=0.9)
# memory = ReplayMemory(100000)

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = PrioritizedReplayMemory(100000)
print(optimizer)




# def select_action(state, EPS_START):
#     if random.random() < EPS_START:
#         return torch.tensor([[np.random.randint(n_actions)]], device=device, dtype=torch.long)
#     else: 
#         q_values = policy_net(state)
#         return torch.argmax(q_values).item()
  
# 選擇動作
def select_action(state, epsilon):
    if random.random() < epsilon:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            q_values = policy_net.get_q_values(state)
            return torch.argmax(q_values).view(1, 1)            

# def optimize_model():
#     if len(memory) < BATCH_SIZE:
#         return
        
#     # 從經驗回放中採樣批次數據
#     transitions = memory.sample(BATCH_SIZE)
#     batch = Transition(*zip(*transitions))

#     # 創建非終止狀態的掩碼
#     non_final_mask = torch.tensor(
#         tuple(map(lambda s: s is not None, batch.next_state)),
#         device=device,
#         dtype=torch.bool
#     )
    
#     # 收集非終止的下一狀態
#     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
#     # 將當前狀態批次轉換為張量
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat([torch.tensor(r).unsqueeze(0) for r in batch.reward]).to(device)

#     # 計算當前狀態的 Q 值
#     state_action_values = policy_net(state_batch).gather(1, action_batch)

#     # 初始化下一狀態的值
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
#     # 計算下一狀態的目標 Q 值
#     with torch.no_grad():
#         # 使用策略網絡選擇最佳動作
#         # best_actions = policy_net(non_final_next_states).max(1)[1]
#         best_actions = policy_net(non_final_next_states).argmax(dim=1, keepdim=True)
#         # 使用目標網絡評估這些動作
#         next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, best_actions).squeeze()
#         # next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, best_actions.unsqueeze(1)).squeeze()

#     # 計算期望的 Q 值
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch

#     # 計算 Huber 損失
#     # criterion = nn.MSELoss()
#     # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
#     loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

#     # 優化模型
#     optimizer.zero_grad()
#     loss.backward()
    
#     # 梯度裁剪以防止梯度爆炸
#     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    
#     optimizer.step()
    
# 優化模型
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions, weights, indices = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat([torch.tensor(r).unsqueeze(0) for r in batch.reward]).to(device)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # 計算分布式 Q 值
    with torch.no_grad():
        next_dist = target_net(non_final_next_states)
        next_q_values = policy_net.get_q_values(non_final_next_states)
        best_actions = torch.argmax(next_q_values, dim=1)
        next_dist = next_dist[range(len(non_final_next_states)), best_actions]

        t_z = reward_batch.unsqueeze(1) + GAMMA * target_net.support.unsqueeze(0)
        t_z = t_z.clamp(V_MIN, V_MAX)
        b = (t_z - V_MIN) / (V_MAX - V_MIN) * (ATOM_SIZE - 1)
        l = b.floor().long()
        u = b.ceil().long()

        m = torch.zeros(BATCH_SIZE, ATOM_SIZE, device=device)
        for i in range(BATCH_SIZE):
            m[i].scatter_add_(0, l[i], next_dist[i] * (u[i] - b[i]))
            m[i].scatter_add_(0, u[i], next_dist[i] * (b[i] - l[i]))

    dist = policy_net(state_batch)
    dist = dist[range(BATCH_SIZE), action_batch.squeeze()]  # 修正形狀
    log_p = torch.log(dist)
    loss = -(m * log_p).sum(dim=1) * weights
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 更新優先級
    td_errors = (m - dist).abs().sum(dim=1).detach().cpu().numpy()
    memory.update_priorities(indices, td_errors)
    
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

rewards_per_episode = []
steps_done = 0

# Goomba = pyboy.get_sprite_by_tile_identifier([144])
epsilon = EPS_START
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    # selected_mario_pos = torch.tensor([[244, 245, 264, 265]])
    # mario_pos = state[selected_mario_pos]
    
    mario_score = mario.score
    mario_coins = mario.coins
    mario_lives = mario.lives_left
    
    
    total_reward = 0
    
    for t in count():
        action = select_action(state, EPS_START)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        
        info = mario
        
        # 獲取遊戲狀態
        Goomba = pyboy.memory[0XD100] == 0 
        flatten_Goomba = pyboy.memory[0XD100] == 1
        Nokobon = pyboy.memory[0XD100] == 4 
        Nokobon_bomb = pyboy.memory[0XD100] == 5
        Powerup_Status_Samll = pyboy.memory[0xFF99] == 0x00 
        Powerup_Status_big = pyboy.memory[0xFF99] == 0x02
        Die = pyboy.memory[0XFFA6] == 0x90 
        
        # 計算獎勵
        reward = 0
        
        # 基礎獎勵：存活獎勵
        reward += 1
        
        # 金幣獎勵
        if mario.coins > mario_coins :
            reward += 100
            mario_coins = mario.coins
            
        # 擊敗敵人獎勵
        if flatten_Goomba is True :
            reward += 100
            
        if Nokobon_bomb is True :
            reward += 100
            
        # 變身獎勵
        if Powerup_Status_big is True :
            reward += 1000
            
        # 死亡懲罰
        if Die is True :
            reward -= 500
            
        # 完成關卡獎勵
        if mario.level_progress >= 2601:
            reward += 5000
            print("level complete")
            pyboy.stop()
            break
            
        # 更新狀態
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # # 存儲轉換到記憶中
        # memory.push(state, action, next_state, reward)
        memory.push((state, action, next_state, reward), td_error=1.0)
        
        # 更新狀態
        state = next_state
        total_reward += reward
        
        # 如果生命值為0，結束遊戲
        if mario.lives_left == 0:
            mario.reset_game()
            break

        # 執行優化步驟
        optimize_model()

        # 軟更新目標網路
        for key in policy_net.state_dict():
            target_net.state_dict()[key] = TAU * policy_net.state_dict()[key] + (1 - TAU) * target_net.state_dict()[key]

        steps_done += 1
        
    # 更新 epsilon
    epsilon = max(EPS_END, epsilon * math.exp(-1.0 / EPS_DECAY))

    print(f"Episode {i_episode + 1}/{num_episodes}, Total Reward: {total_reward}")
