import gymnasium as gym

import environment as environment
from torch import nn
from stable_baselines3 import DQN, PPO
import random
from stable_baselines3.common.env_util import make_vec_env


env = gym.make("camera-v2", render_mode="human")


# loading models
model = PPO.load("logs/model_372000_steps.zip")
model.set_env(env)


obs, info = env.reset()


while True:

    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        seed = random.randint(0, 100)
        obs, info = env.reset(seed=seed)
