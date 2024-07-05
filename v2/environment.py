import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from PIL import Image

import torch

import agent as cam
import random
import numpy as np

register(
    id="camera-v2",
    entry_point="environment:Environment",
)

WHITE = 255


class Placenta:
    def __init__(self, file_path):

        self.image = self.load_image(file_path)

        self.width = self.image.shape[0]
        self.height = self.image.shape[1]

        # self.filled_areas = 30

    def load_image(self, file_path):
        image = Image.open(file_path).convert("L")
        image = np.array(image)
        image = np.swapaxes(image, 1, 0)

        return image


class Environment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):

        self.final_reward = 40
        self.discovery_reward = 5
        self.termination_penalty = 19
        self.time_penalty = 1

        self.step_limit = 60

        self.step_counter = 0
        self.render_mode = render_mode

        placenta_image_path = "placenta.png"
        placenta = Placenta(placenta_image_path)

        self.placenta_areas = 30 

        viewport_width = 256
        viewport_height = 256
        step_size = 256

        seed = random.randint(0, 100)

        self.camera = cam.Agent(
            placenta, viewport_width, viewport_height, step_size, seed
        )

        self.action_space = spaces.Discrete(len(cam.Actions))

        self.observation_space = spaces.Box(
            low=0,
            high=32,
            shape=((165,)),
            dtype=np.uint8,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.camera.reset(seed)
        self.step_counter = 0

        observations = self.camera.get_observation()
        info = {}

        if self.render_mode == "human":
            print("Seed: ", seed)
            self.render()

        return observations, info

    def step(self, action):

        action_succes, discovered_new_area = self.camera.perform_action(
            cam.Actions(action)
        )
        self.step_counter += 1

        reward = 0

        if discovered_new_area:
            reward += self.discovery_reward

        reward -= self.time_penalty

        truncated = False
        if self.step_counter > self.step_limit:
            truncated = True

        terminated = False
        if not action_succes:
            reward -= self.termination_penalty
            terminated = True

        elif self.camera.seen_areas == self.placenta_areas:
            reward += self.final_reward
            terminated = True

        observations = self.camera.get_observation()
        info = {}

        if self.render_mode == "human":
            print(
                cam.Actions(action),
                reward,
            )
            print()
            self.render()

        return observations, reward, terminated, truncated, info

    def render(self):
        self.camera.render()
        print(self.camera.map)


if __name__ == "__main__":
    env = gym.make("camera-v2", render_mode="human")

    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    observations = env.reset()[0]

    for i in range(10):
        rand_action = env.action_space.sample()
        observations, reward, terminated, _, _ = env.step(rand_action)
