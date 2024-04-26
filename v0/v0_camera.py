from enum import Enum
import random
import numpy as np

class CameraAction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class ImageGrid(Enum):
    _UNDISCOVERED = 0
    DISCOVERED = 1

    def __str__(self) -> str:
        return self.name[:1]
    
class Camera:
    
    # Initialize the grid size. Pass in an integer seed to make randomness (Targets) repeatable.
    def __init__(self, grid_rows=8, grid_cols=8,camera_width = 3,camera_height=3, seed=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        self.camera_width = camera_width
        self.camera_height = camera_height

        self.num_all_pixels = grid_rows * grid_cols
      
        self.reset(seed)

    def reset(self, seed=None):
        # Initialize Camera's starting position
        self.camera_pos = [0,0]

        self.num_seen_pixels = 0
        self.seen_pixels = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)

        self.update_seen_pixels()

        # # Random Camera position
        # random.seed(seed)
        # self.camera_pos = [
        #     random.randint(1, self.grid_rows-1),
        #     random.randint(1, self.grid_cols-1)
        # ]

    def update_seen_pixels(self):
        
        for row in range(self.camera_height):
            for column in range(self.camera_width):
                x_pos = row + self.camera_pos[0] - (self.camera_height-1)/2
                y_pos = column + self.camera_pos[1] - (self.camera_width-1)/2

                x_pos = int(x_pos)
                y_pos = int(y_pos)

                if self.inside_vertical(x_pos) and self.inside_horizontal(y_pos) and self.seen_pixels[x_pos, y_pos] != 1:
                    self.seen_pixels[x_pos, y_pos] = 1
                    self.num_seen_pixels += 1

    def inside_vertical(self, position) -> bool:
        if position >= 0 and position <= self.grid_rows-1:
            return True
        else:
            return False
    
    def inside_horizontal(self, position) -> bool:
        if position >= 0 and position <= self.grid_cols-1:
            return True
        else:
            return False 

    def perform_action(self, camera_action:CameraAction) -> bool:
        # Move Camera to the next cell
        if camera_action == CameraAction.LEFT:
            if self.inside_vertical(self.camera_pos[1]-1):
                self.camera_pos[1]-=1

        elif camera_action == CameraAction.RIGHT:
            if self.inside_vertical(self.camera_pos[1]+1):
                self.camera_pos[1]+=1

        elif camera_action == CameraAction.UP:
            if self.inside_horizontal(self.camera_pos[0]-1):
                self.camera_pos[0]-=1

        elif camera_action == CameraAction.DOWN:
            if self.inside_horizontal(self.camera_pos[0]+1):
                self.camera_pos[0]+=1

        self.update_seen_pixels()

        # Return true if Camera reaches all pixels
        return self.num_seen_pixels == self.num_all_pixels

    def render(self):
        # Print current state on console
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):

                if(self.seen_pixels[r,c] == 1):
                    print(ImageGrid.DISCOVERED, end=' ')
                elif(self.seen_pixels[r,c] == 0):
                    print(ImageGrid._UNDISCOVERED, end=' ')               
               

            print() # new line
        print() # new line

# For unit testing
if __name__=="__main__":
    camera = Camera(8,8,3,3)
    camera.render()

    for i in range(25):
        rand_action = random.choice(list(CameraAction))
        print(rand_action)

        camera.perform_action(rand_action)
        camera.render()