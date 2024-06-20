import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import v0.camera as cam
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='camera-v0',                                # call it whatever you want
    entry_point='v0_camera_env:CameraEnv', # module_name:class_name
)


# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/

class CameraEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, grid_rows=5, grid_cols=5, camera_width=3, camera_height=3, render_mode=None):

        self.final_reward = 100
        self.time_factor = 1
        self.step_limit = 200

        self.step_counter = 0
        self.grid_rows=grid_rows
        self.grid_cols=grid_cols
        self.render_mode = render_mode

        # Initialize the Camera problem
        self.camera = cam.Camera(grid_rows=grid_rows, grid_cols=grid_cols, camera_width=camera_width, camera_height=camera_height)

        # Gym requires defining the action space. The action space is camera's set of possible actions.
        # Training code can call action_space.sample() to randomly select an action. 
        self.action_space = spaces.Discrete(len(cam.CameraAction))

        # Gym requires defining the observation space. The observation space consists of the camera's and target's set of possible positions.
        # The observation space is used to validate the observation returned by reset() and step().
        # Use a 1D vector: [camera_row_pos, camera_col_pos]
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.grid_rows-1, self.grid_cols-1]),
            shape=(2,),
            dtype=np.int64
        )

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.

        # Reset the Camera. Optionally, pass in seed control randomness and reproduce scenarios.
        self.camera.reset(seed=seed)
        self.step_counter = 0

        # Construct the observation state:
        # [camera_row_pos, camera_col_pos]
        obs = np.array(self.camera.camera_pos)
        
        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            self.render()

        # Return observation and info
        return obs, info

    # Gym required function (and parameters) to perform an action
    def step(self, action):

        prev_num_pixels = self.camera.num_seen_pixels

        # Perform action
        target_reached = self.camera.perform_action(cam.CameraAction(action))
        self.step_counter +=1

        # Determine reward and termination
        new_seen_pixels = self.camera.num_seen_pixels - prev_num_pixels
        reward =  new_seen_pixels - self.time_factor
        # reward=0
     
        truncated =False
        if self.step_counter > self.step_limit:
            truncated = True
        

        terminated=False
        if target_reached:
            reward+=self.final_reward
            terminated=True

        # Construct the observation state: 
        # [camera_row_pos, camera_col_pos]
        obs = np.array(self.camera.camera_pos)

        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            print(cam.CameraAction(action))
            self.render()

        # Return observation, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info

    # Gym required function to render environment
    def render(self):
        self.camera.render()

# For unit testing
if __name__=="__main__":
    env = gym.make('camera-v0', render_mode='human')

    # Use this to check our custom environment
    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    for i in range(10):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)