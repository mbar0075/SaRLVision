from models import *
from utils import *

import os
import time
import renderlab as rl
import gymnasium as gym
import cv2
import numpy as np
import torch
import torch.nn as nn
import itertools
import random
import warnings
warnings.filterwarnings("ignore")


class DQNAgent():
    """
        The DQN agent that interacts with the environment

        Args:
            env: The environment to interact with
            replay_buffer: The replay buffer to store and sample transitions from
            target_update_freq: The frequency with which the target network is updated
            criterion: The loss function used to train the policy network
            name: The name of the agent (default: DQN)
            network: The network used to estimate the action-value function (default: DQN)

        Attributes:
            env: The environment to interact with
            replay_buffer: The replay buffer to store and sample transitions from
            nsteps: The number of steps to run the agent for
            target_update_freq: The frequency with which the target network is updated
            ninputs: The number of inputs
            noutputs: The number of outputs
            policy_net: The policy network
            target_net: The target network
            optimizer: The optimizer used to update the policy network
            criterion: The loss function used to train the policy network
            epsilon: The probability of selecting a random action
            steps_done: The number of steps the agent has run for
            episodes: The number of episodes the agent has run for
            episode_avg_rewards: The average reward for each episode
            episode_lengths: The lengths of each episode
            best_episode: The best episode
            solved: Whether the environment is solved
            display_every_n_episodes: The number of episodes after which the results are displayed
            time: The time taken to run the agent
    """
    def __init__(self, env, replay_buffer, target_update_freq, criterion=nn.SmoothL1Loss(), name="DQN", network=DQN):
        self.env = env
        self.replay_buffer = replay_buffer
        self.target_update_freq = target_update_freq
        self.ninputs = env.observation_space.shape[0]
        self.noutputs = env.action_space.n
        self.policy_net = network(self.ninputs, self.noutputs).to(device)
        self.target_net = network(self.ninputs, self.noutputs).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=ALPHA)
        self.criterion = criterion
        self.epsilon = EPS_START
        self.steps_done = 0
        self.episodes = 0
        self.episode_info = {"name":name, "episode_avg_rewards": [], "episode_lengths": [], "iou": [], "recall": [],  "best_episode": {"episode": 0, "avg_reward": np.NINF}, "solved": False, "eps_duration": 0}
        self.display_every_n_episodes = 1

    def select_action(self, state):
        """ Selects an action using an epsilon greedy policy """
        # Selecting a random action with probability epsilon
        if random.random() <= self.epsilon: # Exploration
            action = self.env.action_space.sample()
        else: # Exploitation
            # Selecting the action with the highest Q-value otherwise
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                qvalues = self.policy_net(state)
                action = qvalues.argmax().item()
        return action
        
    def update(self):
        """ Updates the policy network using a batch of transitions """
        # Sampling a batch of transitions from the replay buffer
        states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch()

        # Converting the tensors to cuda tensors
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        next_states = next_states.to(device)
        
        # print("states.shape: ", states.shape)
        # print("actions.shape: ", actions.shape)
        # Calculating the Q-values for the current states
        qvalues = self.policy_net(states.squeeze(1)).gather(1, actions)

        # Calculating the Q-values for the next states
        with torch.no_grad():
            # Calculating the Q-values for the next states using the target network (Q(s',a'))
            target_qvalues = self.target_net(next_states.squeeze(1))

            # Calculating the maximum Q-values for the next states (max(Q(s',a'))
            max_target_qvalues = torch.max(target_qvalues, axis=1).values.unsqueeze(1)

            # Calculating the next Q-values using the Bellman equation (Q(s,a) = r + γ * max(Q(s',a')))
            next_qvalues = rewards + GAMMA * (1 - dones.type(torch.float32)) * max_target_qvalues

        # Calculating the loss
        loss = self.criterion(qvalues, next_qvalues)

        # Optimizing the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clipping the gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Updating the target network
        if self.episodes % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            

    def update_epsilon(self):
        """ Updates epsilon """
        self.epsilon = max(EPS_END, EPS_DECAY * self.epsilon)

    def train(self):
        """ Trains the agent for nsteps steps """
        # Resetting the environment
        obs, _ = self.env.reset()

        # Retrieving the starting time
        start_time = time.time()

        # Setting the episode_reward to 0
        episode_reward = 0

        # Running the agent for nsteps steps
        for step in itertools.count():
            # Selecting an action
            action = self.select_action(obs)

            # Taking a step in the environment
            new_obs, reward, terminated, truncated, info = self.env.step(action)

            # Adding the reward to the cumulative reward
            episode_reward += reward

            # Setting done to terminated or truncated
            done = terminated or truncated

            # Creating a transition
            transition = Transition(obs, action, reward, done, new_obs)

            # Appending the transition to the replay buffer
            self.replay_buffer.append(transition)

            # Resetting the observation
            obs = new_obs

            # Ending the episode and displaying the results if the episode is done
            if done:
                # Appending the rewards to the replay buffer
                self.replay_buffer.rewards.append(episode_reward)

                # Updating epsilon
                self.update_epsilon()

                # Resetting the environment
                obs, _ = self.env.reset()

                # Incrementing the number of episodes
                self.episodes += 1

                # Appending the average episode reward
                self.episode_info["episode_avg_rewards"].append(np.mean(self.replay_buffer.rewards))

                # Appending the episode length
                self.episode_info["episode_lengths"].append(self.steps_done)

                # Updating the best episode
                if self.episode_info["episode_avg_rewards"][-1] > self.episode_info["best_episode"]["avg_reward"]:
                    self.episode_info["best_episode"]["episode"] = self.episodes
                    self.episode_info["best_episode"]["avg_reward"] = self.episode_info["episode_avg_rewards"][-1]

                # Retrieving iou
                iou = info["iou"]

                # Retrieving recall
                recall = info["recall"]

                # Appending the iou
                self.episode_info["iou"].append(iou)

                # Appending the recall
                self.episode_info["recall"].append(recall)

                # Checking if the environment is solved
                # if np.mean(self.episode_info["episode_avg_rewards"][-MAX_REPLAY_SIZE:]) >= SUCCESS_CRITERIA:
                #     self.episode_info["solved"] = True
                # If the last 50 episodes had an average IoU of 0.75 or more, the environment is considered solved
                if np.mean(self.episode_info["iou"][-50:]) >= SUCCESS_CRITERIA:
                    self.episode_info["solved"] = True

                
                # Displaying the results
                if self.episodes % self.display_every_n_episodes == 0:
                    print("\033[35mEpisode:\033[0m {} \033[35mEpsilon:\033[0m {:.2f} \033[35mAverage Reward:\033[0m {} \033[35mEpisode Length:\033[0m {} \033[35mIoU:\033[0m {:.2f} \033[35mRecall:\033[0m {:.2f}".format(
                                self.episodes,
                                self.epsilon,
                                self.episode_info["episode_avg_rewards"][-1],
                                self.episode_info["episode_lengths"][-1],
                                iou,
                                recall)
                        )
                    print("-" * 100)

                # Resetting the cumulative reward
                episode_reward = 0

                self.steps_done = 0

                # Checking if the environment is solved
                if self.episode_info["solved"]:
                    print("\033[32mSolved in {} episodes!\033[0m".format(self.episodes))
                    print("-" * 100)
                    break

            # Updating the policy network
            self.update()

            # Updating the number of steps
            self.steps_done += 1

        # Retrieving the ending time
        end_time = time.time()

        # Calculating the time taken
        self.episode_info["eps_duration"] = end_time - start_time

    def run(self):
        """ Runs the agent """
        # Initializing the replay buffer
        self.replay_buffer.initialize()

        # Training the agent
        self.train()


    def test(self):
        """ Tests the trained agent and creates an MP4 video """
        # OpenCV video settings
        video_filename = 'output_video.mp4'
        width = self.env.width
        height = self.env.height
        video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))# 5 fps, lower is better

        # Resetting the environment
        obs, _ = self.env.reset()

        # Collecting frames for video creation
        frames = []

        # Playing the environment
        while True:
            # Selecting the action with the highest Q-value
            action = int(torch.argmax(self.policy_net(torch.from_numpy(obs).float().unsqueeze(0).to(device))).item())

            # Taking a step in the environment
            obs, _, terminated, truncated, _ = self.env.step(action)

            # Render the frame
            frame = self.env.render()
            frames.append(frame)

            # Setting done to terminated or truncated
            if terminated or truncated:
                break

        # Save frames to video
        for frame in frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Release video writer
        video_writer.release()
        print(f"Video saved to {video_filename}")


    def save(self, path="models/dqn"):
        """ Function to save the model 
            
            Args:
                path (str): The path to save the model to
        """
        # Creating the directory if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Saving the model
        torch.save(self.policy_net.state_dict(), path + "/policy_net.pth")
        torch.save(self.target_net.state_dict(), path + "/target_net.pth")

        # Saving optimizer state
        torch.save(self.optimizer.state_dict(), path + "/optimizer.pth")

        # Saving the episode info
        np.save(path + "/episode_info.npy", self.episode_info)

    def load(self, path="models/dqn"):
        """ Function to load the model 
            
            Args:
                path (str): The path to load the model from
        """
        # Loading the model
        self.policy_net.load_state_dict(torch.load(path + "/policy_net.pth"))
        self.target_net.load_state_dict(torch.load(path + "/target_net.pth"))

        self.policy_net.to(device)
        self.target_net.to(device)

        # Loading optimizer state
        self.optimizer.load_state_dict(torch.load(path + "/optimizer.pth"))

        # Loading the episode info
        self.episode_info = np.load(path + "/episode_info.npy", allow_pickle=True).item()

        self.epsilon = EPS_END

    def get_episode_info(self):
        """ Returns the episode info """
        return self.episode_info

class DoubleDQNAgent(DQNAgent):
    """ The Double DQN agent that interacts with the environment and inherits from the DQN agent """
    def __init__(self, env, replay_buffer, target_update_freq, criterion=nn.SmoothL1Loss(), name="DoubleDQN"):
        super().__init__(env, replay_buffer, target_update_freq, criterion, name, DQN)

    def update(self):
        """ Updates the policy network using a batch of transitions """
        # Sampling a batch of transitions from the replay buffer
        states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch()

        # Converting the tensors to cuda tensors
        states = states.to(device).squeeze(1)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        next_states = next_states.to(device).squeeze(1)
        
        # Calculating the Q-values for the current states
        qvalues = self.policy_net(states).gather(1, actions)

        # Calculating the Q-values for the next states
        with torch.no_grad():
            # Using the policy network to select the action with the highest Q-value for the next states (argmax(Q(s',a)))
            next_state_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            
            # Using the target network to calculate the Q-value of the selected action for the next states (Q'(s',argmax(Q(s',a))))
            next_qvalues = self.target_net(next_states).gather(1, next_state_actions)

            # Calculating the next Q-values using the Bellman equation (Q(s,a) = r + γ * Q'(s',argmax(Q(s',a))))
            target_qvalues = rewards + GAMMA * (1 - dones.type(torch.float32)) * next_qvalues

        # Calculating the loss
        loss = self.criterion(qvalues, target_qvalues)

        # Optimizing the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clipping the gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Updating the target network
        if self.episodes % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

class DuelingDQNAgent(DQNAgent):
    """ The Dueling DQN agent that interacts with the environment and inherits from the DQN agent """
    def __init__(self, env, replay_buffer, target_update_freq, criterion=nn.SmoothL1Loss(), name="DuelingDQN"):
        super().__init__(env, replay_buffer, target_update_freq, criterion, name, DuelingDQN)


class DoubleDuelingDQNAgent(DQNAgent):
    """ The Double Dueling DQN agent that interacts with the environment and inherits from the DQN agent """
    def __init__(self, env, replay_buffer, target_update_freq, criterion=nn.SmoothL1Loss(), name="DoubleDuelingDQN"):
        super().__init__(env, replay_buffer, target_update_freq, criterion, name, DuelingDQN)

    def update(self):
        """ Updates the policy network using a batch of transitions """
        # Sampling a batch of transitions from the replay buffer
        states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch()

        # Converting the tensors to cuda tensors
        states = states.to(device).squeeze(1)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        next_states = next_states.to(device).squeeze(1)
        
        # Calculating the Q-values for the current states
        qvalues = self.policy_net(states).gather(1, actions)

        # Calculating the Q-values for the next states
        with torch.no_grad():
            # Using the policy network to select the action with the highest Q-value for the next states (argmax(Q(s',a)))
            next_state_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            
            # Using the target network to calculate the Q-value of the selected action for the next states (Q'(s',argmax(Q(s',a))))
            next_qvalues = self.target_net(next_states).gather(1, next_state_actions)

            # Calculating the next Q-values using the Bellman equation (Q(s,a) = r + γ * Q'(s',argmax(Q(s',a))))
            target_qvalues = rewards + GAMMA * (1 - dones.type(torch.float32)) * next_qvalues

        # Calculating the loss
        loss = self.criterion(qvalues, target_qvalues)

        # Optimizing the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clipping the gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Updating the target network
        if self.episodes % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())