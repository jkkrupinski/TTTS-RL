import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import camera as cam
import numpy as np

register(
    id="camera-v1",
    entry_point="camera_env:CameraEnv",
)


class CameraEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):

        self.final_reward = 1000
        self.time_factor = 100
        self.step_limit = 100

        self.step_counter = 0
        self.render_mode = render_mode

        seed = 44
        self.camera = cam.Camera()

        self.action_space = spaces.Discrete(len(cam.CameraAction))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, 256, 256),
            dtype=np.uint8,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.camera.reset(seed=seed)
        self.step_counter = 0

        observations = self.camera.image.swapaxes(0, 2)

        info = {}

        if self.render_mode == "human":
            self.render()

        return observations, info

    def step(self, action):

        prev_num_pixels = self.camera.seen_white_pixels

        target_reached = self.camera.perform_action(cam.CameraAction(action))
        self.step_counter += 1

        new_seen_pixels = self.camera.seen_white_pixels - prev_num_pixels
        reward = new_seen_pixels - self.time_factor

        truncated = False
        if self.step_counter > self.step_limit:
            truncated = True

        terminated = False
        if target_reached:
            reward += self.final_reward
            terminated = True

        # observations = self.camera.image.flatten()
        observations = self.camera.image.swapaxes(0, 2)

        info = {}

        if self.render_mode == "human":
            print(
                cam.CameraAction(action),
                reward,
            )
            print()
            self.render()

        return observations, reward, terminated, truncated, info

    def render(self):
        self.camera.render()


if __name__ == "__main__":
    env = gym.make("camera-v1", render_mode="human")

    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    observations = env.reset()[0]

    for i in range(10):
        rand_action = env.action_space.sample()
        observations, reward, terminated, _, _ = env.step(rand_action)
