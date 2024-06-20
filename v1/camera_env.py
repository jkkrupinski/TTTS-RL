import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from PIL import Image

import camera as cam
import numpy as np

register(
    id="camera-v1",
    entry_point="camera_env:CameraEnv",
)

WHITE = 255


class Placenta:
    def __init__(self, file_path):

        self.image = self.load_image(file_path)

        self.width = self.image.shape[0]
        self.height = self.image.shape[1]

        self.white_pixels = self.count_white_pixels()  # faster init without

    def load_image(self, file_path):
        image = Image.open(file_path).convert("L")
        image = np.array(image)
        image = np.swapaxes(image, 1, 0)

        return image

    def count_white_pixels(self):
        counter = 0
        for row in range(self.height):
            for column in range(self.width):
                if self.image[column, row] == WHITE:
                    counter += 1
        return counter


class CameraEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):

        self.final_reward = 1000
        self.termination_penalty = 300
        self.time_penalty = 100

        self.step_limit = 40

        self.step_counter = 0
        self.render_mode = render_mode

        placenta_image_path = "placenta.png"
        env = Placenta(placenta_image_path)
        self.white_pixels = env.white_pixels

        viewport_width = 256
        viewport_height = 256
        step_size = 256
        self.camera = cam.Camera(env, viewport_width, viewport_height, step_size)

        self.action_space = spaces.Discrete(len(cam.CameraAction))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, viewport_width, viewport_height),
            dtype=np.uint8,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.camera.reset(seed=seed)
        self.step_counter = 0

        observations = self.camera.observation.swapaxes(0, 2)

        info = {}

        if self.render_mode == "human":
            self.render()

        return observations, info

    def step(self, action):

        prev_num_pixels = self.camera.seen_white_pixels

        action_succes = self.camera.perform_action(cam.CameraAction(action))
        self.step_counter += 1

        new_seen_pixels = self.camera.seen_white_pixels - prev_num_pixels
        reward = new_seen_pixels - self.time_penalty

        truncated = False
        if self.step_counter > self.step_limit:
            truncated = True

        terminated = False
        if not action_succes:
            reward -= self.termination_penalty
            terminated = True

        elif self.camera.seen_white_pixels == self.white_pixels:
            reward += self.final_reward
            terminated = True

        observations = self.camera.observation.swapaxes(0, 2)

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

    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    observations = env.reset()[0]

    for i in range(40):
        rand_action = env.action_space.sample()
        observations, reward, terminated, _, _ = env.step(rand_action)
