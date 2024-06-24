import gymnasium as gym

import environment as environment
from torch import nn
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./logs/",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

env = gym.make("camera-v2", render_mode=None)

num_cpu = 7
env = make_vec_env("camera-v2", n_envs=num_cpu, vec_env_cls=DummyVecEnv)

model = DQN("MlpPolicy", env, verbose=1, buffer_size=1000)

# # loading models
# model = DQN.load("logs/rl_model_60000_steps.zip")
# model.load_replay_buffer('logs/rl_model_replay_buffer_60000_steps.pkl')
# model.set_env(env)


model.learn(
    total_timesteps=200_000,
    log_interval=100,
    progress_bar=True,
    callback=checkpoint_callback,
)

model.save("models/dqn_camera2_5")


del model  # remove to demonstrate saving and loading

model = DQN.load("models/dqn_camera2_5")

env = gym.make("camera-v2", render_mode="human")


obs, info = env.reset()

while True:

    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
