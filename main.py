import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import FlattenObservation


from pyboy import PyBoy

from collections import namedtuple, deque
import random
import math
from enum import Enum

# ----------- Enum Actions ----------- #
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

# ----------- Device ----------- #
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ----------- Mario Env ----------- #
class MarioEnv(gym.Env):
    def __init__(self, pyboy):
        super().__init__()
        self.pyboy = pyboy
        self.action_space = Discrete(len(Actions))
        self.observation_space = Box(
            low=0, 
            high=255, 
            shape=(16,20), dtype=np.uint32)

    

    def step(self, action):
        
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        

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

        self.pyboy.tick()
        
        info = self.pyboy.game_wrapper
        reward = self.calculate_reward()
        observation = self.get_obs()
        game_state = self.game_state()

        # 檢查 lives <= 0
        terminated = game_state["lives"] <= 0
        truncated = False
        if terminated:
            observation, info = self.reset()  # 重置環境
            reward -= 100  # 死亡額外懲罰
        
        return observation, reward, terminated, truncated, info
    
    def calculate_reward(self):

        total_reward = 0
        level_progress = 251

        if (self.pyboy.memory[0xFFA6] == 0x90) == True:
            total_reward -= 500

        if self.pyboy.game_wrapper.level_progress > level_progress:
            total_reward += 100

        return total_reward

    def game_state(self):
        info = {
            "mario_x": self.pyboy.memory[0xC202],
            "mario_y": self.pyboy.memory[0xC201],
            "mario_level_progress": self.pyboy.game_wrapper.level_progress,
            "score": self.pyboy.game_wrapper.score,
            "lives": self.pyboy.game_wrapper.lives_left,
            "time": self.pyboy.game_wrapper.time_left,
            "level": self.pyboy.game_wrapper.world[1],
            "world": self.pyboy.game_wrapper.world,
            "is_dead": bool(self.pyboy.memory[0xFFA6] == 0x90)
        }
        return info

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.get_obs()
        info = self.game_state()
        return observation, info
    
    def get_obs(self):
        return self.pyboy.game_area()
    
    
        
def normalize_game_state(game_state):
    state_vector = torch.tensor([
        game_state["mario_x"] / 255.0,  # 假設 x 範圍 0-255
        game_state["mario_y"] / 255.0,  # 假設 y 範圍 0-255
        game_state["mario_level_progress"] / 1000.0,  # 假設進度 0-1000
        game_state["score"] / 10000.0,  # 假設分數 0-10000
        game_state["lives"] / 5.0,  # 假設生命 0-5
        game_state["time"] / 400.0,  # 假設時間 0-400
        game_state["level"] / 10.0,  # 假設關卡 0-10
        game_state["world"][0] / 8.0,  # 假設世界 1-8
        float(game_state["is_dead"]),  # 0 或 1
    ], dtype=torch.float32).to(device)
    return state_vector

        
"================setup================"
pyboy = PyBoy("rom.gb", window="SDL2", sound_volume=50)

env = MarioEnv(pyboy)

mario = pyboy.game_wrapper
mario.start_game()

observation, info = env.reset()

# [[339 339 339 339 339 339 339 339 339 339 339 339 339 339 339 339 339 339
#   339 339]
#  [320 320 320 320 320 320 320 320 320 320 320 320 320 320 320 320 320 320
#   320 320]
#  [300 300 300 300 300 300 300 300 300 300 300 300 321 322 321 322 323 300
#   300 300]
#  [300 300 300 300 300 300 300 300 300 300 300 324 325 326 325 326 327 300
#   300 300]
#  [300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300
#   300 300]
#  [300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300
#   300 300]
#  [300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300
#   300 300]
#  [300 300 300 300 300 300 300 300 310 350 300 300 300 300 300 300 300 300
#   300 300]
#  [300 300 300 300 300 300 300 310 300 300 350 300 300 300 300 300 300 300
#   300 300]
#  [300 300 300 300 300 129 310 300 300 300 300 350 300 300 300 300 300 300
#   300 300]
#  [300 300 300 300 300 310 300 300 300 300 300 300 350 300 300 300 300 300
#   300 300]
#  [300 300 310 350 310 300 300 300 300 306 307 300 300 350 300 300 300 300
#   300 300]
#  [300 368 369 300   0   1 300 306 307 305 300 300 300 300 350 300 300 300
#   300 300]
#  [310 370 371 300  16  17 300 305 300 305 300 300 300 300 300 350 300 300
#   300 300]
#  [352 352 352 352 352 352 352 352 352 352 352 352 352 352 352 352 352 352
#   352 352]
#  [353 353 353 353 353 353 353 353 353 353 353 353 353 353 353 353 353 353
#   353 353]]


game_shape = env.observation_space.shape
num_actions = len(Actions)
game_state = env.game_state()

print(game_state, len(game_state)) #9
print("Observation shape:", game_shape) #(16, 20)
print("Number of actions:", num_actions) #14


# 改進的 DQN 模型（整合 game_state）
class DQN(nn.Module):
    def __init__(self, input_shape=(16,20), in_channels=1, num_actions=num_actions, fc_units=128, state_dim=len(game_state)):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.m = nn.Flatten()

        # 使用 dummy tensor 計算 conv 層輸出 flatten 大小（自動處理 game_shape）
        with torch.no_grad():
            h, w = input_shape
            dummy = torch.zeros(1, in_channels, h, w)
            out = F.relu(self.conv1(dummy))
            out = self.pool1(out)
            out = F.relu(self.conv2(out))
            conv_output_size = out.numel() // out.shape[0]

        self.fc1 = nn.Linear(conv_output_size + state_dim, fc_units)  # 拼接 game_state
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(fc_units, num_actions)

    def forward(self, x, state_vector):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.m(x)
        x = torch.cat((x, state_vector), dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# 初始化模型和優化器
model = DQN(input_shape=game_shape, in_channels=1, num_actions=num_actions, fc_units=128, state_dim=len(game_state)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練循環
replay_buffer = deque(maxlen=10000)
Transition = namedtuple('Transition', ('state', 'game_state', 'action', 'next_state', 'next_game_state', 'reward', 'done'))
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
gamma = 0.99


# 依 lives 控制 episode，死亡才算一個 episode
episode = 0
max_episodes = 100000  # 可自行調整最大訓練次數
while episode < max_episodes:
    state, game_state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    game_state_vector = normalize_game_state(game_state).unsqueeze(0)
    total_reward = 0
    done = False
    prev_lives = game_state["lives"]
     # 新增：準確率計數
    step_count = 0
    agree_count = 0
    
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = model(state, game_state_vector)
                pred_action = q_values.argmax().item()
                
            # epsilon-greedy：若隨機採樣，action 為隨機；否則使用模型預測
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = pred_action
                
            # 更新準確率計數
            step_count += 1
            if action == pred_action:
                agree_count += 1

        next_state, reward, terminated, truncated, info = env.step(action)
        next_game_state = env.game_state()
        done = terminated or truncated
        total_reward += reward

        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        next_game_state_vector = normalize_game_state(next_game_state).unsqueeze(0)
        replay_buffer.append(Transition(state, game_state_vector, action, next_state, next_game_state_vector, reward, done))

        if len(replay_buffer) >= 64:
            batch = random.sample(replay_buffer, 64)
            states, game_states, actions, next_states, next_game_states, rewards, dones = zip(*batch)
            states = torch.cat(states).to(device)
            game_states = torch.cat(game_states).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            next_states = torch.cat(next_states).to(device)
            next_game_states = torch.cat(next_game_states).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            q_values = model(states, game_states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = model(next_states, next_game_states).max(1)[0]
            targets = rewards + (1 - dones) * gamma * next_q_values

            loss = F.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 檢查 lives 是否減少，若減少則結束本 episode
        if next_game_state["lives"] < prev_lives:
            # 若 lives 等於 0，重置 mario（使用者要求）
            if next_game_state["lives"] == 0:
                mario.reset_game()

            episode += 1
            # 輸出加上準確率
            accuracy = (agree_count / step_count) if step_count > 0 else 0.0
            print(f"Episode {episode}, Total Reward: {total_reward}, Accuracy: {accuracy*100:.2f}% ({agree_count}/{step_count})")
            break

        state = next_state
        game_state_vector = next_game_state_vector
        prev_lives = next_game_state["lives"]
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
pyboy.stop()    
        
