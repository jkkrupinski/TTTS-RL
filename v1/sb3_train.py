import gymnasium as gym

import camera_env as camera_env
from torch import nn
from stable_baselines3 import DQN


env = gym.make("camera-v1", render_mode=None)

model = DQN("CnnPolicy", env, verbose=1, buffer_size=1000)
model.mnbv (total_timesteps=200_000, log_interval=100, progress_bar=True)

model.save("models/dqn_camera")


del model  # remove to demonstrate saving and loading

model = DQN.load("models/dqn_camera")

env = gym.make("camera-v1", render_mode="human")


obs, info = env.reset()

while True:

    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
