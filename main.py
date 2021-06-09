import retro
import gym
import random
import numpy as np
import gym.spaces

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

def make_env(game, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = retro.make(game)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


class Discretizer(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        buttons = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
        actions = [['B'], ['A'], ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], [None]]
        self._actions=[]
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='rl_model')
    

if __name__ == '__main__':

    game="BubbleBobble-Nes"
    n_cpus = 6

    env = Discretizer(retro.make(game, scenario="training"))
    env = SubprocVecEnv([make_env(game, i) for i in range(n_cpus)])

    #env = DummyVecEnv([lambda:env])

    model = A2C("CnnpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000, callback=checkpoint_callback)

    model.save("A2C-BubbleBobble")

    del model # remove to demonstrate saving and loading

    model = A2C.load("A2C-BubbleBobble")

    model.set_env(env)

    obs = env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
   
