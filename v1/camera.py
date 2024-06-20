from enum import Enum
import random
import numpy as np
from matplotlib import pyplot as plt

WHITE = 255


class CameraAction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Camera:

    def __init__(self, env, viewport_width, viewport_height, step_size, seed=None):

        self.env = env

        self.width = viewport_width
        self.height = viewport_height

        self.x_bound = self.env.width - self.width
        self.y_bound = self.env.height - self.height

        self.step = step_size

        self.start_x = 768
        self.start_y = 516

        self.reset(seed)

        plt.ion()
        self.fig = plt.figure()
        self.axes_img = plt.imshow(np.swapaxes(self.env_map, 1, 0))

    def reset(self, seed=None):

        if seed == None:
            self.position = [self.start_x, self.start_y]  # (top, left) corner

        else:
            random.seed(seed)
            self.position = [
                random.randint(0, self.x_bound),
                random.randint(0, self.y_bound),
            ]

        self.init_map()

    def init_map(self):

        self.seen_white_pixels = 0
        self.env_map = np.zeros((self.env.width, self.env.height), dtype=np.int32)

        self.get_observation()
        self.fill_map(self.observation, self.position[0], self.position[1])

    def update_map(self, action: CameraAction):

        partial_observation, x_begin, y_begin = self.get_partial_observation(action)
        self.fill_map(partial_observation, x_begin, y_begin)
        self.get_observation()

    def fill_map(self, observation, x_begin, y_begin):

        for x_cam in range(observation.shape[0]):
            for y_cam in range(observation.shape[1]):
                x_map, y_map = self.cam2map(x_cam, y_cam, x_begin, y_begin)

                if observation[x_cam, y_cam] == WHITE:
                    if self.env_map[x_map, y_map] != WHITE:
                        self.env_map[x_map, y_map] = WHITE
                        self.seen_white_pixels += 1

    def get_observation(self):

        x_begin = self.position[0]
        x_end = self.position[0] + self.width

        y_begin = self.position[1]
        y_end = self.position[1] + self.height

        self.observation = self.env.image[
            x_begin:x_end, y_begin:y_end, np.newaxis
        ].astype(np.uint8)

    def get_partial_observation(self, action):

        if action == CameraAction.LEFT:
            x_begin = self.position[0]
            x_end = self.position[0] + self.step

            y_begin = self.position[1]
            y_end = self.position[1] + self.height

        elif action == CameraAction.RIGHT:
            x_begin = self.position[0] + self.width - self.step
            x_end = self.position[0] + self.width

            y_begin = self.position[1]
            y_end = self.position[1] + self.height

        elif action == CameraAction.UP:
            x_begin = self.position[0]
            x_end = self.position[0] + self.width

            y_begin = self.position[1]
            y_end = self.position[1] + self.step

        elif action == CameraAction.DOWN:
            x_begin = self.position[0]
            x_end = self.position[0] + self.width

            y_begin = self.position[1] + self.height - self.step
            y_end = self.position[1] + self.height

        partial_observation = self.env.image[x_begin:x_end, y_begin:y_end].astype(
            np.uint8
        )

        return partial_observation, x_begin, y_begin

    def cam2map(self, x_cam, y_cam, x_begin, y_begin):
        x_map = x_cam + x_begin
        y_map = y_cam + y_begin

        return x_map, y_map

    def is_x_inside(self, x) -> bool:
        if x >= 0 and x <= self.x_bound:
            return True
        return False

    def is_y_inside(self, y) -> bool:
        if y >= 0 and y <= self.y_bound:
            return True
        return False

    def perform_action(self, action: CameraAction) -> bool:
        action_succes = False

        if action == CameraAction.LEFT:
            if self.is_x_inside(self.position[0] - self.step):
                self.position[0] -= self.step
                action_succes = True

        elif action == CameraAction.RIGHT:
            if self.is_x_inside(self.position[0] + self.step):
                self.position[0] += self.step
                action_succes = True

        elif action == CameraAction.UP:
            if self.is_y_inside(self.position[1] - self.step):
                self.position[1] -= self.step
                action_succes = True

        elif action == CameraAction.DOWN:
            if self.is_y_inside(self.position[1] + self.step):
                self.position[1] += self.step
                action_succes = True

        if action_succes:
            self.update_map(action)

        # Return true if Camera reaches all pixels
        return action_succes

    def render(self, mark_position=True):

        swapped_map = np.swapaxes(self.env_map, 1, 0)

        mark_size = 40
        mark_color = 128

        if mark_position:
            marked_map = swapped_map.__deepcopy__(None)
            marked_map[
                self.position[1] : self.position[1] + mark_size,
                self.position[0] : self.position[0] + mark_size,
            ] = (
                np.ones((mark_size, mark_size)) * mark_color
            )

            self.axes_img.set_data(marked_map)

        else:
            self.axes_img.set_data(swapped_map)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    camera = Camera()
    camera.render(True)

    for i in range(25):
        rand_action = random.choice(list(CameraAction))
        print(rand_action)

        camera.perform_action(rand_action)
        camera.render(True)
