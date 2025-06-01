from pyboy import PyBoy
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import FrameStackObservation
from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from enum import Enum

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

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class MarioEnv(gym.Env):
    def __init__(self, pyboy):
        super().__init__()
        self.pyboy = pyboy
        
        # state = self.get_state()
        
        self.action_space = Discrete(len(Actions))
        self.observation_space = Box(low=256, high=256, shape=(20,18), dtype=np.uint16)

    def send_input(self, event):
        self.pyboy.send_input(event)

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
        
        observation = torch.tensor(self.get_state())
        
        reward = self.pyboy.game_wrapper.score
        
        terminated = self.pyboy.game_wrapper.level_progress > 2601
        
        info = self.pyboy.game_wrapper
        
        truncated = False
        
        return observation, reward, terminated, truncated, info
    
    def get_state(self):
        
        self.pyboy.get_sprite(3)
        self.pyboy.get_sprite(4)
        self.pyboy.get_sprite(5)
        self.pyboy.get_sprite(6)
        self.pyboy.get_sprite(20)
        self.pyboy.get_sprite(20)
        self.pyboy.get_sprite(21)
        pyboy.memory[0XD100]

        return [
            pyboy.get_sprite(3).x if pyboy.get_sprite(3).on_screen else False,
            pyboy.get_sprite(3).y if pyboy.get_sprite(3).on_screen else False,
            pyboy.get_sprite(4).x if pyboy.get_sprite(4).on_screen else False,
            pyboy.get_sprite(4).y if pyboy.get_sprite(4).on_screen else False,
            pyboy.get_sprite(5).x if pyboy.get_sprite(5).on_screen else False,
            pyboy.get_sprite(5).y if pyboy.get_sprite(5).on_screen else False,
            pyboy.get_sprite(6).x if pyboy.get_sprite(6).on_screen else False,
            pyboy.get_sprite(6).y if pyboy.get_sprite(6).on_screen else False,
            pyboy.get_sprite(20).x if pyboy.get_sprite(20).on_screen else False,
            pyboy.get_sprite(20).y if pyboy.get_sprite(20).on_screen else False,
            pyboy.get_sprite(21).x if pyboy.get_sprite(21).on_screen else False,
            pyboy.get_sprite(21).y if pyboy.get_sprite(21).on_screen else False,
            pyboy.memory[0XD100]
        ]
         
    
    def state(self):
        return self.get_state()
    
    def reset(self, seed=42, options=None):
        super().reset(seed=seed)
        
        observation = torch.tensor(self.get_state(), dtype=torch.float32)
        info = self.pyboy.game_wrapper
        
        return observation, info

pyboy = PyBoy("rom.gb", window="SDL2", sound_volume=0)
env = MarioEnv(pyboy)
# env = FrameStackObservation(env, stack_size=4)


mario = pyboy.game_wrapper

mario.start_game(world_level=(1, 1))
#mario.game_area_mapping(pyboy.game_wrapper.mapping_minimal, 0)
mario.set_lives_left(10)

# 獲取 game_area
game_area = pyboy.game_area()
print(game_area.shape)  # 假設輸出：(20, 18)
print(game_area.dtype)  # 假設輸出：uint8
print(game_area.min(), game_area.max())

# for i in range(1):
#     print(pyboy.get_sprite(3))
#     print(pyboy.get_sprite(3).x if pyboy.get_sprite(3).on_screen else None)
#     print(pyboy.get_sprite(3).y if pyboy.get_sprite(3).on_screen else None)
#     print(pyboy.get_sprite(4))
#     print(pyboy.get_sprite(5))
#     print(pyboy.get_sprite(6))
#     print(pyboy.get_sprite(20))
#     print(pyboy.get_sprite(21))

#print(env.get_state())
print(env.observation_space)

observation, info = env.reset()

print(observation)

n_observations = len(observation)
n_actions = env.action_space.n

print(n_observations)

print(n_actions)

Transition = namedtuple('Transition',
                        ('observation', 'action', 'next_observation', 'reward'))


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

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
policy_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)

print(policy_net)

steps_done = 0

def select_action(observation):
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
            return policy_net(observation).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[np.random.choice(n_actions)]], device=device, dtype=torch.long)
    
print(select_action(observation))


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
                                          batch.next_observation)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_observation
                                                if s is not None])
    state_batch = torch.cat(batch.observation)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
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
    
    observation, info = env.reset()
    observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    
    while pyboy.tick():
        action = select_action(observation)
        observation_, reward, done, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = torch.tensor([done], device=device)
        
        if mario.lives_left == 0:
            mario.reset_game()
            
        if pyboy.memory[0XFFA6] == 0X90:
            reward += 0
            
        if pyboy.memory[0XD100] == 0X01 or pyboy.memory[0XD100] == 0X05:
            reward += 100
        
        if done:
            next_observation = None
        else:
            next_observation = torch.tensor(observation_, dtype=torch.float32, device=device).unsqueeze(0)
        
        memory.push(observation, action, next_observation, reward)
        
        observation = next_observation
            
        optimize_model()
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
    pyboy.stop()
    
