from enum import Enum
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

WHITE = 255

class CameraAction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Env:
    def __init__(self, file_path):
   
        image = Image.open(file_path).convert('L')
        self.image = np.array(image)

        self.width = self.image.shape[0]
        self.height = self.image.shape[1]

    def count_white_pixels(self):
        couter = 0
        for row in range(self.height):
            for column in range(self.width):
                if self.image[column,row] == 255:
                    couter+=1
        print(couter)

class Camera:
    
    # Initialize the grid size. Pass in an integer seed to make randomness (Targets) repeatable.
    def __init__(self, seed=None):
    
        self.env = Env('seg_255rgb.png')

        self.width = 256
        self.height = 256

        self.x_bound = self.env.width - self.width
        self.y_bound = self.env.height - self.height

        self.step = 256
        self.all_white_pixels = 412304

        self.reset(seed)

    def reset(self, seed=None):
        # Initialize Camera's (top, left) corner starting position
        self.position = [512,512] # (Y,X)

        # Initialize map for rewards
        self.seen_white_pixels = 0
        self.env_map = np.zeros((self.env.width, self.env.height), dtype=np.int32)

        # Random Camera position
        if(seed != None):
            random.seed(seed)
            self.position = [
                random.randint(0, self.x_bound),
                random.randint(0, self.y_bound)
            ]

        self.init_map()

    def init_map(self):
        
        cam_image = self.env.image[self.position[1]:self.position[1]+self.height, self.position[0]:self.position[0]+self.width] 

        # Update map + get rewards
        for x_cam in range(self.width):
            for y_cam in range(self.height):
                x_map = x_cam + self.position[1] 
                y_map = y_cam + self.position[0] 

                if cam_image[x_cam, y_cam] == WHITE:
                    if self.env_map[x_map, y_map] != 1:
                        self.env_map[x_map, y_map] = WHITE
                        self.seen_white_pixels += 1

    def update_map_pixels(self, x_cam, y_cam):
            x_map = x_cam + self.position[0] 
            y_map = y_cam + self.position[1] 
            self.env_map[x_map, y_map] = WHITE
            self.seen_white_pixels += 1

    def update_map(self, action:CameraAction):

        if action == CameraAction.LEFT:         
            cam_image = self.env.image[self.position[0]:self.position[0]+self.step,                        self.position[1]:self.position[1]+self.height] 

        elif action == CameraAction.RIGHT:           
            cam_image = self.env.image[self.position[0]+self.width-self.step:self.position[0]+self.width,  self.position[1]:self.position[1]+self.height] 

        elif action == CameraAction.UP:           
            cam_image = self.env.image[self.position[0]:self.position[0]+self.width,  self.position[1]+self.height - self.step:self.position[1]+self.height] 

        elif action == CameraAction.DOWN:
            cam_image = self.env.image[self.position[0]:self.position[0]+self.width,  self.position[1]+self.step:self.position[1]+self.height] 

    
        for x_cam in range(cam_image.shape[0]):
            for y_cam in range(cam_image.shape[1]):
                if cam_image[x_cam, y_cam] == WHITE:
                    self.update_map_pixels(x_cam, y_cam)  

    def is_x_inside(self, x) -> bool:
        if x >= 0 and x <= self.x_bound:
            return True
        return False
    
    def is_y_inside(self, y) -> bool:
        if y >= 0 and y <= self.y_bound:
            return True
        return False 

    def perform_action(self, action:CameraAction) -> bool:
        action_succes = False

        if action == CameraAction.LEFT:
            if self.is_x_inside(self.position[0]-self.step):
                self.position[0]-=self.step
                action_succes = True

        elif action == CameraAction.RIGHT:
            if self.is_x_inside(self.position[0]+self.step):
                self.position[0]+=self.step
                action_succes = True

        elif action == CameraAction.UP:
            if self.is_y_inside(self.position[1]-self.step):
                self.position[1]-=self.step
                action_succes = True

        elif action == CameraAction.DOWN:
            if self.is_y_inside(self.position[1]+self.step):
                self.position[1]+=self.step  
                action_succes = True

        if action_succes:
            self.init_map()
            # self.update_map(action) # not ready

        # Return true if Camera reaches all pixels
        return self.seen_white_pixels == self.all_white_pixels

    def render(self):
        plt.imshow(self.env_map)
        plt.show()


# For unit testing
if __name__=="__main__":
    camera = Camera()
    camera.render()

    for i in range(25):
        rand_action = random.choice(list(CameraAction))
        print(rand_action)

        camera.perform_action(rand_action)
        camera.render()