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
        self.observation_space = Box(low=0, high=255, shape=(16,20), dtype=np.uint8)
        self.last_score = 0
        self.last_enemy_id = pyboy.memory[0xD100]

    # def get_state(self):
    #     def safe_val(sprite, attr):
    #         return float(getattr(sprite, attr)) if sprite.on_screen else -1.0
    #     return [
    #         safe_val(self.pyboy.get_sprite(i), 'x') if i in [3, 4, 5, 6, 20, 21] else 0 for i in range(3, 22)
    #         for attr in ['x', 'y']
    #     ][:12] + [
    #         float(self.pyboy.memory[0xFFA6] == 0x90),
    #         # float(self.pyboy.memory[0xD100] == 0x01),
    #         float(self.pyboy.memory[0xD100] == 0x05),
    #         float(self.pyboy.memory[0xD100] == 0x0F),
    #         float(self.pyboy.memory[0xC201] > 134),
    #         float(self.pyboy.game_wrapper.level_progress),
    #         float(self.pyboy.game_wrapper.lives_left),
    #         float(self.pyboy.game_wrapper.score),
    #         float(self.pyboy.game_wrapper.world[0]),
    #         float(self.pyboy.game_wrapper.world[1]),
    #         float(self.pyboy.game_wrapper.time_left),
    #         self.pyboy.get_tile(144).tile_identifier,
            
            
    #     ]

    

    def step(self, action):
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
        
        info = pyboy.game_wrapper

        # 判斷敵人剛被消滅
        current_enemy_id = pyboy.memory[0xD100]
        self.enemy_just_killed = (
            self.last_enemy_id in [0x01, 0x05, 0x04, 0x0E, 0x0F, 0x46] and current_enemy_id == 0xFF
        )
        self.last_enemy_id = current_enemy_id

        reward = self.calc_reward()

        # obs = torch.tensor(self.get_state(), dtype=torch.float32, device=device)
        observation = self.get_obs()

        truncated = False
        terminated = False
        
        if pyboy.memory[0xFFA6] == 0x90 or pyboy.memory[0xC0A4] == 0x39:
            terminated = True
        elif pyboy.game_wrapper.level_progress == 2601:
            terminated = True
            self.pyboy.stop()

        return observation, reward, terminated, truncated, info
    
    def get_obs(self):
        # 取得遊戲狀態
        game_state = [
            pyboy.game_wrapper.level_progress,
            pyboy.game_wrapper.lives_left,
            pyboy.game_wrapper.score,
            pyboy.game_wrapper.world[0],
            pyboy.game_wrapper.world[1],
            pyboy.game_wrapper.time_left
        ]

        # 取得 Mario 狀態
        mario_state = [
            pyboy.memory[0xC202],
            pyboy.memory[0xC201],
            pyboy.memory[0xC20A],
            pyboy.memory[0xFFA6],
            pyboy.memory[0xFF99],
            pyboy.memory[0xFFB5],
        ]

        # 取得物件資訊（敵人/道具）
        object_info = np.array(pyboy.memory[0xD100:0xD108], dtype=np.float32)

        # 計算距離
        distances = [object_info[3] - mario_state[0], object_info[2] - mario_state[1]]

        # 是否無敵
        no_enemy = [object_info[0] == 0xFF]

        return np.concatenate([
            np.array(game_state, dtype=np.float32),
            np.array(mario_state, dtype=np.float32),
            object_info,
            np.array(distances, dtype=np.float32),
            np.array(no_enemy, dtype=np.float32)
        ])
    
    def calc_reward(self):
        reward = pyboy.game_wrapper.score - self.last_score
        self.last_score = pyboy.game_wrapper.score
        reward += 5  # 存活獎勵
        if pyboy.memory[0xFFA6] == 0x90:
            reward -= 200
        if pyboy.memory[0xFFA6] == 0x50:
            reward += 100
        if getattr(self, 'enemy_just_killed', False):
            reward += 200
        if pyboy.game_wrapper.level_progress == 2601:
            reward += 500
        if pyboy.memory[0xC0A4] == 0x39:
            reward -= 100
        reward = np.clip(reward, -1000, 2000)
        return reward
    
    def reset(self, seed=42, options=None):
        super().reset(seed=seed)
        self.last_enemy_id = pyboy.memory[0xD100]
        # obs = torch.tensor(self.get_state(), dtype=torch.float32, device=device)
        observation = self.get_obs()
        info = pyboy.game_wrapper
        return observation, info
        


# ----------- Replay Memory ----------- #
Transition = namedtuple('Transition', ('observation', 'action', 'next_observation', 'reward', 'terminated'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


    
# ----------- Setup ----------- #
BATCH_SIZE = 256
GAMMA = 0.995
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
    
pyboy = PyBoy('rom.gb', window="SDL2", sound_volume=0)
env = MarioEnv(pyboy)

mario = pyboy.game_wrapper
mario.start_game()

observation, info = env.reset()
# print(observation.flatten())

# flatten_observation = torch.tensor(observation, dtype=torch.float32, device=device).flatten()
# print(f"Flattened observation : {flatten_observation}")
n_observations = observation.size
n_actions = env.action_space.n

# print(f"Observation size: {n_observations}, Action space: {n_actions}")

# ----------- DQN Model ----------- #
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_observations, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)

# print(f"Policy network: {policy_net}")

# ----------- Select Action ----------- #
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # 修正：隨機產生合法 action index
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)
        

# print(f"Select action function: {select_action(observation)}")

# ----------- Optimize Model ----------- #
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_observation)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32, device=device).flatten().unsqueeze(0)
                                       for s in batch.next_observation if s is not None]) if any(s is not None for s in batch.next_observation) else torch.empty(0, n_observations, device=device)
    state_batch = torch.cat([torch.tensor(s, dtype=torch.float32, device=device).flatten().unsqueeze(0)
                             for s in batch.observation])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        if len(non_final_next_states) > 0:
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# ----------- reward_history ----------- #
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
plt.ion()

def plot_rewards(reward_history, num_episodes):
    axs.clear()
    rewards = [r.item() if hasattr(r, 'item') else float(r) for r in reward_history]
    axs.plot(range(1, num_episodes + 1), rewards, marker='o', linestyle='-')
    axs.set_xlabel('Episode')
    axs.set_ylabel('Reward')
    axs.set_title('Episode Reward Trend')
    axs.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


    
# ----------- Training Loop ----------- #
num_episodes = 600 if torch.cuda.is_available() or torch.backends.mps.is_available() else 50

# print(f"開始訓練，共 {num_episodes} 個 episodes")
# print(f"使用設備: {device}")
# print(f"觀察空間大小: {n_observations}, 動作空間大小: {n_actions}")
reward_history = []
for episode in range(num_episodes):
    # print(f"Episode {episode + 1}/{num_episodes}")
    observation, info = env.reset()
    
    # 1. flatten 並轉 tensor
    observation = torch.tensor(observation, dtype=torch.float32, device=device).flatten().unsqueeze(0)
    episode_done = False
    while pyboy.tick():
        # 2. select_action 輸入 tensor
        action = select_action(observation)
        action_value = action.item()

        # 3. env.step 回傳 observation 也要 flatten 並轉 tensor
        next_obs, reward, terminated, truncated, info = env.step(action_value)
        reward = torch.tensor([reward], device=device)
        terminated = torch.tensor([terminated], device=device)
        truncated = torch.tensor([truncated], device=device)

        episode_done = (terminated | truncated).item()

        if not episode_done:
            next_observation = torch.tensor(next_obs, dtype=torch.float32, device=device).flatten().unsqueeze(0)
            observation = next_observation
        else:
            next_observation = None

        # 存儲原始 numpy 陣列而不是 tensor
        obs_np = observation.cpu().numpy().squeeze()
        next_obs_np = next_observation.cpu().numpy().squeeze() if next_observation is not None else None
        
        memory.push(obs_np, action, next_obs_np, reward, terminated)
        

        optimize_model()

        # target_net 軟更新
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key].data.copy_(
                policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1.0 - TAU)
            )

        # if episode_done:
        #     print(f"Episode {episode + 1} 完成，獎勵總和: {reward.item():.2f}")
        #     break
        
        # if mario.lives_left <= 0:
        if pyboy.memory[0xC0A4] == 0x39:
            reward_history.append(reward)  # reward 是 torch.tensor([reward], device=device)
            plot_rewards(reward_history, len(reward_history))
            mario.set_lives_left(10)
            mario.reset_game()
           # print(f"Episode {episode + 1} 生命歸零，重置遊戲")
            
            
    pyboy.stop()