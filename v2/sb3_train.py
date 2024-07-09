import gymnasium as gym
import environment as environment
from torch import nn

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=2000,
    save_path="./logs/",
    name_prefix="model",
    save_vecnormalize=True,
)

env = gym.make("camera-v2", render_mode=None)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

# # loading models
# model = DQN.load("logs/rl_model2_200000_steps.zip")
# model.load_replay_buffer('logs/rl_model2_replay_buffer_200000_steps.pkl') # 135000
# model.set_env(env)


model.learn(
    total_timesteps=2_000_000,
    log_interval=1,
    progress_bar=True,
    callback=checkpoint_callback,
)

model.save("models/model2_fin")
