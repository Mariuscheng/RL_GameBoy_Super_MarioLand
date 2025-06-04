import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
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
        self.observation_space = Box(low=np.array([0]*12 + [0]*5 + [0, 0, 0, 1, 1, 0], dtype=np.float32),
                                     high=np.array([255]*12 + [1]*5 + [3000, 10, 999999, 8, 4, 400], dtype=np.float32))
        self.last_progress = 0

    def get_state(self):
        def safe_val(sprite, attr):
            return float(getattr(sprite, attr)) if sprite.on_screen else -1.0
        return [
            safe_val(self.pyboy.get_sprite(i), 'x') if i in [3, 4, 5, 6, 20, 21] else 0 for i in range(3, 22)
            for attr in ['x', 'y']
        ][:12] + [
            float(self.pyboy.memory[0xFFA6] == 0x90),
            float(self.pyboy.memory[0xD100] == 0x01),
            float(self.pyboy.memory[0xD100] == 0x05),
            float(self.pyboy.memory[0xD100] == 0x0F),
            float(self.pyboy.memory[0xC201] > 134),
            float(self.pyboy.game_wrapper.level_progress),
            float(self.pyboy.game_wrapper.lives_left),
            float(self.pyboy.game_wrapper.score),
            float(self.pyboy.game_wrapper.world[0]),
            float(self.pyboy.game_wrapper.world[1]),
            float(self.pyboy.game_wrapper.time_left)
        ]

    def reset(self, seed=42, options=None):
        super().reset(seed=seed)
        self.last_progress = 0
        obs = torch.tensor(self.get_state(), dtype=torch.float32, device=device)
        return obs, self.pyboy.game_wrapper

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

        progress = self.pyboy.game_wrapper.level_progress
        reward = progress - self.last_progress
        self.last_progress = progress

        if self.pyboy.memory[0xD100] in [0x01, 0x05, 0x0F]:
            reward += 100
        if self.pyboy.game_wrapper.lives_left <= 0:
            reward -= 500

        obs = torch.tensor(self.get_state(), dtype=torch.float32, device=device)
        terminated = self.pyboy.game_wrapper.level_progress > 2601
        truncated = False

        return obs, reward, terminated, truncated, self.pyboy.game_wrapper

# ----------- DQN Model ----------- #
class DQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

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

# ----------- Select Action ----------- #
steps_done = 0

def select_action(observation):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if observation.dim() == 1:
        observation = observation.unsqueeze(0)

    if random.random() < eps_threshold:
        return torch.tensor([[np.random.randint(n_actions)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            q_values = policy_net(observation)
            return q_values.argmax(dim=1).view(1, 1)

# ----------- Optimize ----------- #
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_observation)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_observation if s is not None])

    state_batch = torch.cat(batch.observation)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    done_batch = torch.cat(batch.terminated)

    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze()
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_values = (next_state_values * GAMMA) + reward_batch * (1 - done_batch.float())
    loss = nn.SmoothL1Loss()(state_action_values, expected_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# ----------- Setup ----------- #
BATCH_SIZE = 256
GAMMA = 0.995
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

pyboy = PyBoy("rom.gb", window="SDL2", sound_volume=0, cgb=True)
env = MarioEnv(pyboy)
mario = pyboy.game_wrapper
mario.start_game(world_level=(1, 1))
mario.set_lives_left(10)

observation, info = env.reset()
obs_size = observation.shape[0]
n_actions = env.action_space.n

policy_net = DQN(obs_size, n_actions).to(device)
target_net = DQN(obs_size, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)

# ----------- Training Loop ----------- #
num_episodes = 600 if torch.cuda.is_available() or torch.backends.mps.is_available() else 50

for episode in range(num_episodes):
    observation, info = env.reset()
    if observation.dim() == 1:
        observation = observation.unsqueeze(0)

    episode_done = False
    while not episode_done:
        pyboy.tick()

        action = select_action(observation)
        action_value = action.item()

        next_obs, reward, terminated, truncated, info = env.step(action_value)
        reward = torch.tensor([reward], device=device)
        terminated = torch.tensor([terminated], device=device)
        truncated = torch.tensor([truncated], device=device)

        episode_done = (terminated | truncated).item()

        if not episode_done:
            next_observation = next_obs.unsqueeze(0) if next_obs.dim() == 1 else next_obs
        else:
            next_observation = None

        memory.push(observation, action, next_observation, reward, terminated)
        observation = next_observation

        optimize_model()

        for key in policy_net.state_dict():
            target_net.state_dict()[key] = policy_net.state_dict()[key] * TAU + target_net.state_dict()[key] * (1.0 - TAU)

        if mario.lives_left == 0:
            mario.reset_game()
