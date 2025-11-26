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

class MarioEnv(gym.Env):
    def __init__(self, pyboy):
        super().__init__()
        self.pyboy = pyboy
        self.action_space = Discrete(len(Actions))
        # change observation_space to include channel dimension (H,W,C=1)
        self.observation_space = Box(low=0, high=255, shape=(16, 20), dtype=np.uint32)
        # how many emulator ticks to advance per action
        
        
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
        
        total_reward = 0

        # advance emulator for frame_skip ticks, accumulate reward
        
        info = self.pyboy.game_wrapper
        terminated = False
        truncated = False
        
        # accumulate reward at each tick (game_state may change)
        total_reward += self.calculate_reward()
        # check level progress or game over
        if self.pyboy.game_wrapper.level_progress == 2601:
                terminated = True

        observation = self.get_obs()
        # extra terminal bonus
        if terminated:
            pyboy.stop()

        return observation, total_reward, terminated, truncated, info
    
    def game_state(self):
        # 讀取記憶值並計算進度（避免引用未定義的變數）
        mario_x = self.pyboy.memory[0xC202]
        mario_y = self.pyboy.memory[0xC201]
        mario_direction = self.pyboy.memory[0xC20D]

        # level_progress 以遊戲 wrapper 的 level_progress 做差異計算
        end_progress = 2601
        new_progress = end_progress - self.pyboy.game_wrapper.level_progress

        score = self.pyboy.game_wrapper.score
        lives = self.pyboy.game_wrapper.lives_left
        time_left = self.pyboy.game_wrapper.time_left
        lost_life = self.pyboy.memory[0xC0A3] == 255
        world = self.pyboy.game_wrapper.world[0]
        stage = self.pyboy.game_wrapper.world[1]
        is_dead_from_pit_enemy = self.pyboy.memory[0xFFA6] == 0x90
        power_up = self.pyboy.memory[0xFF99]
        game_over = self.pyboy.memory[0xC0A4] == 0x39

        enemy_id = self.pyboy.memory[0xD100]
        enemy_HP = self.pyboy.memory[0xD101]
        enemy_x = self.pyboy.memory[0xD103]
        enemy_y = self.pyboy.memory[0xD102]

        game_state = {
            "mario": {
                "x": mario_x,
                "y": mario_y,
                "direction": mario_direction,
                "score": score,
                "lives": lives,
                "time": time_left,
                "lost_life": lost_life,
                "world": world,
                "stage": stage,
                "level_progress": new_progress,
                "is_dead_from_pit_enemy": is_dead_from_pit_enemy,
                "power_up": power_up,
                "game_over": game_over,
            },
            "enemy": {
                "id": enemy_id,
                "HP": enemy_HP,
                "x": enemy_x,
                "y": enemy_y,
            }
        }

        return game_state
    
    def calculate_reward(self):
        total_reward = 0
        game_state = self.game_state()
        
        # 每步微小負獎勵，鼓勵快速通關
        total_reward -= 0.05

        # 死亡懲罰（生命減少）
        current_lives = game_state["mario"]["lives"]
        if current_lives < self.prev_lives:
            total_reward -= 50  # 懲罰要夠大
        self.prev_lives = current_lives

        # 掉坑懲罰
        if game_state["mario"]["is_dead_from_pit_enemy"]:
            total_reward -= 30
            

        # 遊戲結束懲罰
        if game_state["mario"]["game_over"]:
            total_reward -= 100
            self.pyboy.game_wrapper.reset_game()
            self.pyboy.game_wrapper.set_lives_left(10)

        # 前進獎勵（level_progress減少）
        current_progress = game_state["mario"]["level_progress"]
        if current_progress < self.prev_progress:
            total_reward += 5
        self.prev_progress = current_progress

        # 通關獎勵
        if current_progress == 0:
            total_reward += 1000

        # 右移獎勵（可選，避免太高）
        if game_state["mario"]["direction"] == 1:
            total_reward += 0.2

        return total_reward

    def reset(self, *, seed=42, options=None):
        super().reset(seed=seed)
        self.prev_progress = 2601
        self.prev_lives = 10
        
        observation = self.get_obs()
        info = self.pyboy.game_wrapper
        return observation, info
    
    def get_obs(self):
        self.pyboy.game_wrapper.game_area_mapping(pyboy.game_wrapper.mapping_minimal, 0)
        return self.pyboy.game_wrapper.game_area()


pyboy = PyBoy('rom.gb', window="SDL2", sound_volume=0)

env = MarioEnv(pyboy)
env = FrameStackObservation(env, stack_size=4)
# env = ReshapeObservation(env, shape=(1, 4, 16, 20))

mario = pyboy.game_wrapper

mario.start_game()
mario.set_lives_left(10)

BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05          # 保留 5% 隨機
EPS_DECAY = 300000      # 約 120~150 萬步才降到 0.1 左右
TAU = 0.001
LR = 6.25e-5

observation, _ = env.reset()
observation = torch.tensor(observation)


# print(observation)
# tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 3, 3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 3, 3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
#         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]],
#        dtype=torch.uint32)

# new_observation = torch.stack((observation, observation), dim=0)
# print(new_observation.shape)
# print(new_observation.size()[0])

state = observation[0]
# print(state)

observation_shape = observation.shape #(4, 16, 20)
# print(new_observation.shape)

# print("Observation shape:", observation_shape)

n_observations = observation.shape[0]
n_actions = env.action_space.n
# print(n_observations)
# print(n_actions)

class Cnn(nn.Module):
    """
    通用 CNN，輸入 (B, C, H, W)。會自動計算全連接層輸入大小。
    用法：
      model = Cnn(n_observations, n_actions).to(device)
      x = obs_to_tensor(observation, device)   # x shape -> (1, C, H, W)
      out = model(x)                           # out shape -> (1, n_actions)
    """
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_observations, 32, kernel_size=3, padding=1),   # 32x32 → 32x32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 16x16
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → 8x8
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # → 8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 直接變成 1x1（比 Flatten + FC 更好）
        )
        self.classifier = nn.Linear(64, n_actions)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 或 x.squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x
    
policy_net = Cnn(n_observations, n_actions).to(device)
target_net = Cnn(n_observations, n_actions).to(device)
policy_net.eval()


# Replay memory and optimizer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
memory = deque([], maxlen=10000)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
steps_done = 0


def select_action(state):
    global steps_done
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None

    transitions = random.sample(memory, BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # mask for non-final next states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

    # state tensors: each state should be shape (1, C, H, W)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Build non_final_next_states with correct shape (N_non_final, C, H, W)
    if non_final_mask.any():
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    else:
        # create empty tensor with same channel/space dims as state_batch
        non_final_next_states = torch.empty((0,) + state_batch.shape[1:], device=device)

    # Compute Q(s_t, a)
    # policy_net(state_batch) -> (B, n_actions); gather -> (B,1)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_mask.any():
        with torch.no_grad():
            # target_net(non_final_next_states) -> (N_non_final, n_actions)
            next_q = target_net(non_final_next_states)
            # take max over actions -> (N_non_final,)
            next_state_vals_non_final = next_q.max(1).values
        # Place computed next state values into the appropriate positions
        next_state_values[non_final_mask] = next_state_vals_non_final

    # Compute expected Q values (shape (B,))
    # reward_batch may be shape (B,1) or (B,), ensure shape (B,)
    expected_state_action_values = reward_batch.view(-1) + (GAMMA * next_state_values)

    # Compute loss between state_action_values (B,1) and expected (B,)
    loss = F.smooth_l1_loss(state_action_values.view(-1), expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping (use norm clipping for stability)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
    optimizer.step()

    # Soft update of target network
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

    return loss.item()



if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    observation, info = env.reset()
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
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
        memory.append(Transition(state, action, next_state, reward))

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
        
    pyboy.stop()

