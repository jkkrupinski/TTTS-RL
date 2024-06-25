import gymnasium as gym

import environment as environment
from torch import nn
from stable_baselines3 import DQN
import random


env = gym.make("camera-v2",  render_mode="human")

# loading models
model = DQN.load("logs/rl_model_200000_steps.zip")
model.load_replay_buffer('logs/rl_model_replay_buffer_200000_steps.pkl') # 135000
model.set_env(env)


obs, info = env.reset()


while True:

    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        seed = random.randint(0, 100)
        obs, info = env.reset(seed=seed)
