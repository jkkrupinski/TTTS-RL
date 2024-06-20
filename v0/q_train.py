import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os
import v0.camera_env as camera_env  # Even though we don't use this class here, we should include it here so that it registers the Camera environment.
import time


# Train or test using Q-Learning
def run_q(episodes, is_training=True, render=False):

    env = gym.make("camera-v0", render_mode="human" if render else None)

    if is_training:
        # If training, initialize the Q Table, a 3D vector: [robot_row_pos, robot_row_col, actions]
        q = np.zeros(
            (env.unwrapped.grid_rows, env.unwrapped.grid_cols, env.action_space.n)
        )
    else:
        # If testing, load Q Table from file.
        f = open("v0_camera_solution.pkl", "rb")
        q = pickle.load(f)
        f.close()

    # Hyperparameters
    learning_rate_a = 0.9  # alpha or learning rate
    discount_factor_g = 0.9  # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1  # 1 = 100% random actions

    # Array to keep track of the number of steps per episode for the robot to find the target.
    # We know that the robot will inevitably find the target, so the reward is always obtained,
    # so we want to know if the robot is reaching the target efficiently.
    steps_per_episode = np.zeros(episodes)
    rewards_per_episode = np.zeros(episodes)

    step_count = 0
    for i in range(episodes):
        if render:
            print(f"Episode {i}")

        # Reset environment at the beginning of episode
        state = env.reset()[0]
        terminated = False
        truncated = False

        # Robot keeps going until it finds the target
        while not terminated and not truncated:
            # Select action based on epsilon-greedy
            if is_training and random.random() < epsilon:
                # select random action
                action = env.action_space.sample()
            else:
                # Convert state of [1,2,3,4] to (1,2,3,4), use this to index into the 4th dimension of the 5D array.
                q_state_idx = tuple(state)

                # select best action
                action = np.argmax(q[q_state_idx])

            # Perform action
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Convert state of [1,2] and action of [1] into (1,2,1), use this to index into the 3th dimension of the 3D array.
            q_state_action_idx = tuple(state) + (action,)

            # Convert new_state of [1,2] into (1,2), use this to index into the 4th dimension of the 3D array.
            q_new_state_idx = tuple(new_state)

            if is_training:
                # Update Q-Table
                q[q_state_action_idx] = q[q_state_action_idx] + learning_rate_a * (
                    reward
                    + discount_factor_g * np.max(q[q_new_state_idx])
                    - q[q_state_action_idx]
                )

            # Update current state
            state = new_state

            # Record steps
            step_count += 1
            if terminated:
                steps_per_episode[i] = step_count
                step_count = 0

            rewards_per_episode[i] = reward

        # Decrease epsilon
        epsilon = max(epsilon - 1 / episodes, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001

    env.close()

    # Graph steps
    sum_steps = np.zeros(episodes)
    for t in range(episodes):
        sum_steps[t] = np.mean(
            steps_per_episode[max(0, t - 100) : (t + 1)]
        )  # Average steps per 100 episodes
    plt.plot(sum_steps)

    # # Graph rewards
    # sum_rewards = np.zeros(episodes)
    # for t in range(episodes):
    #     sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    # plt.plot(sum_rewards)

    plt.savefig("v0_camera_solution.png")

    if is_training:
        # Save Q Table
        f = open("v0_camera_solution.pkl", "wb")
        pickle.dump(q, f)
        f.close()


if __name__ == "__main__":

    NUM_EPISODES = 10000

    # Train/test using Q-Learning
    # run_q(NUM_EPISODES, is_training=True, render=False)
    run_q(1, is_training=False, render=True)
