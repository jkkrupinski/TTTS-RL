import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import v1_camera_env as v1_camera_env


# https://poloclub.github.io/cnn-explainer/
class DQN(nn.Module):
    def __init__(self, input_shape, out_actions):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=10,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # flatten inputs into a single vector
            # After flattening the matrix into a vector, pass it to the output layer. To determine the input shape, use the print() statement in forward()
            nn.Linear(in_features=64 * 64 * 10, out_features=out_actions),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        # print(x.shape)  # Use this to determine input shape of the output layer.
        x = self.layer_stack(x)
        return x


# # Use this to check if DQN is valid
# temp_dqn = DQN(1, 4)  # (1 channels, 4 actions)
# temp_tensor = torch.randn(1, 1, 256, 256)  # (batch, channel, row, column)
# temp_dqn(temp_tensor)


class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class CameraDQL:

    def __init__(self, render=False):

        self.init_hyperparams()
        self.init_env(render)

        # Neural Network
        self.loss_fn = nn.MSELoss()
        self.optimizer = None  # NN Optimizer. Initialize later.

        self.memory = ReplayMemory(self.replay_memory_size)

    def init_hyperparams(self):
        self.learning_rate = 0.001  # learning rate
        self.gamma = 0.9  # discount rate
        self.network_sync_rate = 10  # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = 1000  # size of replay memory
        self.mini_batch_size = (
            32  # size of the training data set sampled from the replay memory
        )

    def init_env(self, render):

        self.env = gym.make("camera-v1", render_mode="human" if render else None)
        self.num_actions = self.env.action_space.n
        self.actions = [
            "U",
            "D",
            "L",
            "R",
        ]

    def state2tensor(self, state):
        return torch.unsqueeze(torch.from_numpy(state.astype(np.float32)), 0)

    def build_model(self):
        return DQN(input_shape=1, out_actions=self.num_actions)

    def save(self, name):
        torch.save(self.policy_dqn.state_dict(), name)

    def load(self, name):
        dqn = self.build_model()
        dqn.load_state_dict(torch.load(name))
        return dqn

    def memorize(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def plot(self, name, episodes, rewards_per_episode, epsilon_history):
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x - 100) : (x + 1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)

        plt.savefig(name)

    def train(self, episodes):
        epsilon = 1  # 1 = 100% random actions

        self.policy_dqn = self.build_model()
        self.target_dqn = self.build_model()

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(
            self.policy_dqn.parameters(), lr=self.learning_rate
        )

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0

        for i in range(episodes):

            print(f"Episode [{i}]")

            state = self.env.reset()[0]  # Initialize to state 0
            terminated = False  # True when agent falls in hole or reached goal
            truncated = False  # True when agent takes more than 200 actions

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while not terminated and not truncated:

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = self.env.action_space.sample()
                else:
                    # select best action
                    with torch.no_grad():
                        state_tensor = self.state2tensor(state)
                        action = self.policy_dqn(state_tensor).argmax().item()

                # Execute action
                new_state, reward, terminated, truncated, _ = self.env.step(action)

                # Save experience into memory
                self.memorize(state, action, new_state, reward, terminated)

                state = new_state
                step_count += 1

            # Keep track of the rewards collected per episode.
            if reward > 0:
                rewards_per_episode[i] = reward

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if (
                len(self.memory) > self.mini_batch_size
                and np.sum(rewards_per_episode) > 0
            ):
                mini_batch = self.memory.sample(self.mini_batch_size)
                self.optimize(mini_batch)

                # Decay epsilon
                epsilon = max(epsilon - 1 / episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    step_count = 0

        self.env.close()

        self.save("camera_dql_cnn.pt")

        self.plot("camera_dql_cnn.png", episodes, rewards_per_episode, epsilon_history)

    # Optimize policy network
    def optimize(self, mini_batch):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                with torch.no_grad():
                    new_state_tensor = self.state2tensor(new_state)
                    target = torch.FloatTensor(
                        reward + self.gamma * self.target_dqn(new_state_tensor).max()
                    )

            # Get the current set of Q values
            state_tensor = self.state2tensor(state)
            current_q = self.policy_dqn(state_tensor)
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = self.target_dqn(state_tensor)

            # Adjust the specific action to the target that was just calculated.
            # Target_q[batch][action], hardcode batch to 0 because there is only 1 batch.
            target_q[0][action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes):

        self.policy_dqn = self.load("camera_dql_cnn.pt")
        self.policy_dqn.eval()

        for _ in range(episodes):
            state = self.env.reset()[0]
            terminated = False
            truncated = False

            while not terminated and not truncated:
                # Select best action
                with torch.no_grad():
                    state_tensor = self.state2tensor(state)
                    action = self.policy_dqn(state_tensor).argmax().item()

                # Execute action
                state, _, terminated, truncated, _ = self.env.step(action)

        self.env.close()


if __name__ == "__main__":

    camera_dql = CameraDQL()
    camera_dql.train(10)

    camera_dql = CameraDQL(True)
    camera_dql.test(10)
