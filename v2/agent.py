from enum import Enum
import numpy as np
import random
import numpy as np
from matplotlib import pyplot as plt

WHITE = 255


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Agent:
    def __init__(
        self, placenta, viewport_width, viewport_height, step_size, seed
    ) -> None:

        self.placenta = placenta

        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        self.step = step_size

        self._init_map()
        self._init_map_image()
        self._init_start_position(seed)

        self.seen_areas = 0

        self.update_map()

        plt.ion()
        self.fig = plt.figure()
        self.axes_img = plt.imshow(np.swapaxes(self.map_image, 1, 0))

    def _init_map(self):
        self.map = np.zeros((15, 11), dtype=np.uint8)

    def _init_map_image(self):
        self.map_image = np.zeros(
            (
                15 * self.viewport_width,
                11 * self.viewport_height,
            ),
            dtype=np.uint8,
        )

    def _init_start_position(self, seed):
        random.seed(seed)

        self.map_position = [7 * 256, 5 * 256]
        # self.placenta_start_position = [3 * 256, 2 * 256] # debug
        self.placenta_start_position = [
            random.randint(0, 6 - 1) * 256,  # placenta 7x5
            random.randint(0, 4 - 1)
            * 256,  # 6,4 spawn not to start on the edge of placenta
        ]
        self.placenta_position = self.placenta_start_position

    def get_observation(self):
        x_begin = self.placenta_position[0]
        x_end = self.placenta_position[0] + self.viewport_width

        y_begin = self.placenta_position[1]
        y_end = self.placenta_position[1] + self.viewport_height

        observation = self.placenta.image[
            x_begin:x_end, y_begin:y_end, np.newaxis
        ].astype(np.uint8)

        return observation

    def map2placenta(self):
        x = self.map_position[0] - self.placenta_start_position[0]
        y = self.map_position[1] - self.placenta_start_position[1]

        return x, y

    def cam2map(self, x_cam, y_cam):
        x_placenta = x_cam + self.placenta_position[0]
        y_placenta = y_cam + self.placenta_position[1]

        t_x, t_y = self.map2placenta()

        x_map = x_placenta + t_x
        y_map = y_placenta + t_y

        return x_map, y_map

    def fill_map_image(self):
        observation = self.get_observation()
        render = True  # for debug faster learning
        filled = False

        for x_cam in range(observation.shape[0]):
            for y_cam in range(observation.shape[1]):
                x_map, y_map = self.cam2map(x_cam, y_cam)

                if observation[x_cam, y_cam] == WHITE:
                    filled = True

                    if render:
                        self.map_image[x_map, y_map] = WHITE
                    else:
                        return filled
        return filled

    def update_map(self):

        # check if been there already
        if (
            self.map[int(self.map_position[0] / 256), int(self.map_position[1] / 256)]
            == 1
        ):
            return False

        filled = self.fill_map_image()

        if filled:
            self.map[
                int(self.map_position[0] / 256), int(self.map_position[1] / 256)
            ] = 1
            self.seen_areas += 1
            return True
        else:
            self.map[
                int(self.map_position[0] / 256), int(self.map_position[1] / 256)
            ] = 2
            return False

    def reset(self, seed):
        self._init_map()
        self._init_map_image()
        self._init_start_position(seed)

        self.seen_areas = 0

        self.update_map()

    def is_x_in_map(self, x) -> bool:
        if x >= 0 and x <= self.map_image.shape[0] - self.viewport_width:
            return True
        return False

    def is_y_in_map(self, y) -> bool:
        if y >= 0 and y <= self.map_image.shape[1] - self.viewport_height:
            return True
        return False

    def is_within_placenta(self) -> bool:
        is_within_x = (
            self.placenta_position[0] >= 0
            and self.placenta_position[0] <= (7 - 1) * 256
        )
        is_within_y = (
            self.placenta_position[1] >= 0
            and self.placenta_position[1] <= (5 - 1) * 256
        )
        return is_within_x and is_within_y

    def perform_action(self, action: Actions) -> bool:
        action_succes = False
        discovered_new_area = False

        if action == Actions.LEFT:
            if self.is_x_in_map(self.map_position[0] - self.step):
                self.map_position[0] -= self.step
                self.placenta_position[0] -= self.step
                action_succes = True

        elif action == Actions.RIGHT:
            if self.is_x_in_map(self.map_position[0] + self.step):
                self.map_position[0] += self.step
                self.placenta_position[0] += self.step
                action_succes = True

        elif action == Actions.UP:
            if self.is_y_in_map(self.map_position[1] - self.step):
                self.map_position[1] -= self.step
                self.placenta_position[1] -= self.step
                action_succes = True

        elif action == Actions.DOWN:
            if self.is_y_in_map(self.map_position[1] + self.step):
                self.map_position[1] += self.step
                self.placenta_position[1] += self.step
                action_succes = True

        if action_succes and self.is_within_placenta():
            discovered_new_area = self.update_map()

        map_x_index = int(self.map_position[0] / 256)
        map_y_index = int(self.map_position[1] / 256)

        if discovered_new_area:
            self.map[map_x_index, map_y_index] = 1
        elif self.map[map_x_index, map_y_index] == 0:
            self.map[map_x_index, map_y_index] = 2

        return action_succes, discovered_new_area

    def render(self, mark_position=True):
        swapped_map = np.swapaxes(self.map_image, 1, 0)

        mark_size = 40
        mark_color = 128

        if mark_position:
            marked_map = swapped_map.__deepcopy__(None)
            marked_map[
                self.map_position[1] : self.map_position[1] + mark_size,
                self.map_position[0] : self.map_position[0] + mark_size,
            ] = (
                np.ones((mark_size, mark_size)) * mark_color
            )
            self.axes_img.set_data(marked_map)

        else:
            self.axes_img.set_data(swapped_map)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
