""" Authors:Damian Kreft, Sebastian Kreft
    Required environment: Python3, gymnasium, syable_baseline3"""

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv 
from stable_baselines3 import PPO
import os

def game():
    """Program is a bot that plays game Breakout, created with use of reinforcement learning.
    """
    env = gym.make('Breakout-v4', render_mode='human')
    env = EpisodicLifeEnv(env)

    model = PPO(policy          = 'CnnPolicy', 
                env             = env,     
                verbose         = 1, 
                seed            = 123,
                tensorboard_log = os.path.expanduser('~/models/breakout-v4/tb_log/'))

    model.learn(total_timesteps = 1e6,
                tb_log_name     = 'rl_game_breakout')

game()