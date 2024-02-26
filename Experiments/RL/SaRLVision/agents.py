#-------------------------------------------------------------------------------
# Name:        agents.py
# Purpose:     Defining the DQN agents for SaRLVision.
#
# Author:      Matthias Bartolo <matthias.bartolo@ieee.org>
#
# Created:     February 24, 2024
# Copyright:   (c) Matthias Bartolo 2024-
# Licence:     All rights reserved
#-------------------------------------------------------------------------------
from SaRLVision.models import *
from SaRLVision.utils import *

import os
import time
import imageio
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
            target_update_freq: The frequency with which the target network is updated (default: TARGET_UPDATE_FREQ)
            criterion: The loss function used to train the policy network (default: nn.SmoothL1Loss())
            name: The name of the agent (default: DQN)
            network: The network used to estimate the action-value function (default: DQN)
            exploration_mode: The exploration mode used by the agent (default: GUIDED_EXPLORE)

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
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ, criterion=nn.SmoothL1Loss(), name="DQN", network=DQN, exploration_mode=EXPLORATION_MODE):
        self.env = env
        self.replay_buffer = replay_buffer
        self.target_update_freq = target_update_freq
        self.exploration_mode = exploration_mode
        self.ninputs = env.get_state().shape[1]
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
        self.episode_info = {"name":name, "episode_avg_rewards": [], "episode_lengths": [], "avg_iou": [], "iou": [], "final_iou": [], "recall": [], "avg_recall": [], "best_episode": {"episode": 0, "avg_reward": np.NINF}, "solved": False, "eps_duration": 0}
        self.display_every_n_episodes = 1

    def select_action(self, state):
        """ Selects an action using an epsilon greedy policy """
        # Selecting a random action with probability epsilon
        if random.random() <= self.epsilon: # Exploration
            if self.exploration_mode == GUIDED_EXPLORE: # Guided exploration
                # Expert agent action selection
                action = self.expert_agent_action_selection()
            else:# Random exploration
                # Normal Random action Selection
                action = self.env.action_space.sample()
        else: # Exploitation
            # Selecting the action with the highest Q-value otherwise
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                qvalues = self.policy_net(state)
                action = qvalues.argmax().item()
        return action
    
    def expert_agent_action_selection(self):
        """ Selects an action using an expert agent, by calculating the reward for each action and selecting a random action from the positive actions if the list is not empty, otherwise selecting a random action from the negative actions.

            Returns:
                action: The action selected by the expert agent
        """
        # Creating lists to hold the positive actions and negative actions
        positive_actions = []
        negative_actions = []
        
        # Retrieving the bounding box from the environment
        old_state = self.env.bbox

        # Retrieving the target bounding boxes from the environment
        target_bboxes = self.env.current_gt_bboxes

        # Looping through the actions
        for action in range(self.noutputs):
            # Retrieving the new state
            new_state = self.env.transform_action(action)

            # Calculating the reward
            if action < 8:
                reward = self.env.calculate_reward([new_state], [old_state], target_bboxes)
            else:
                reward = self.env.calculate_trigger_reward([new_state], target_bboxes)

            # Appending the action to the positive or negative actions list based on the reward
            if reward > 0:
                positive_actions.append(action)
            else:
                negative_actions.append(action)

        # Returning a random choice from the positive actions if the list is not empty
        if len(positive_actions) > 0:
            return random.choice(positive_actions)
        else:
            return random.choice(negative_actions)
        
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

            # Adding the IoU and recall to the episode info
            self.episode_info["iou"].append(info["iou"])
            self.episode_info["recall"].append(info["recall"])

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
                # Appending the final IoU to the episode info
                self.episode_info["final_iou"].append(info["iou"])
                
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

                # Calculating the average IoU and recall
                avg_iou = np.mean(self.episode_info["iou"][-self.env.step_count:])
                avg_recall = np.mean(self.episode_info["recall"][-self.env.step_count:])

                # Appending the average IoU and recall
                self.episode_info["avg_iou"].append(avg_iou)
                self.episode_info["avg_recall"].append(avg_recall)

                if USE_EPISODE_CRITERIA:
                    # If the environment number of episodes is greater than SUCCESS_CRITERIA, the environment is considered solved
                    if self.episodes >= SUCCESS_CRITERIA_EPS:
                        self.episode_info["solved"] = True
                else:
                    # If the environment number of epochs is greater than SUCCESS_CRITERIA, the environment is considered solved
                    if self.env.epochs >= SUCCESS_CRITERIA_EPOCHS:
                        self.episode_info["solved"] = True
                
                # Displaying the results
                if self.episodes % self.display_every_n_episodes == 0:
                    print("\033[35mEpisode:\033[0m {} \033[35mEpsilon:\033[0m {:.2f} \033[35mAverage Reward:\033[0m {} \033[35mEpisode Length:\033[0m {} \033[35mAverage IoU:\033[0m {:.2f} \033[35mAverage Recall:\033[0m {:.2f} \033[35mEpochs:\033[0m {} \033[35mFinal IoU:\033[0m {:.2f}".format(
                                self.episodes,
                                self.epsilon,
                                self.episode_info["episode_avg_rewards"][-1],
                                self.episode_info["episode_lengths"][-1],
                                avg_iou,
                                avg_recall,
                                self.env.epochs,
                                self.episode_info["final_iou"][-1])
                        )
                    print("-" * 100)

                # Resetting the cumulative reward
                episode_reward = 0

                self.steps_done = 0

                # Checking if the environment is solved
                if self.episode_info["solved"]:
                    print("\033[32mCompleted {} episodes!\033[0m".format(self.episodes))
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

    def evaluate(self, path="evaluation_results"):
        """ Evaluates the agent """
        # Resetting the environment
        obs, _ = self.env.reset()

        # Running the agent for an epoch
        while True:
            # Selecting an action
            action = self.select_action(obs)

            # Taking a step in the environment
            new_obs, _, terminated, truncated, _ = self.env.step(action)

            # Setting done to terminated or truncated
            done = terminated or truncated

            # Resetting the observation
            obs = new_obs

            # Ending the episode and displaying the results if the episode is done
            if done:
                # Resetting the environment
                obs, _ = self.env.reset()

                # Incrementing the number of episodes
                self.episodes += 1

            # Exiting if the number of epochs is greater than or equal to 1
            if self.env.epochs >= 1:
                break

        # Saving the evaluation results
        self.env.save_evaluation_results()

    def test(self, file_path='dqn_render',video_filename='output_video.mp4'):
        """ Tests the trained agent and creates an MP4 video """
        # Removing the file if it exists
        if os.path.exists(os.path.join(file_path, video_filename)):
            os.remove(os.path.join(file_path, video_filename))

        # OpenCV video settings
        width = self.env.width
        height = self.env.height

        # Creating path if it does not exist
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # Appending the file name to the path
        video_filename = os.path.join(file_path, video_filename)
        video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'avc1'), 1, (width, height))# 5 fps, lower is better

        # Resetting the environment
        obs, _ = self.env.reset()

        # Collecting frames for video creation
        frames = []
        frames.append(self.env.render())# Initial frame

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

        # Adding final frame
        frames.append(self.env.render())# Final frame with label

        # Saving frames to video
        for frame in frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Releasing video writer
        video_writer.release()
        print('\033[92mVideo saved to:\033[0m {}'.format(video_filename))

    def save_gif(self, file_path='dqn_render', gif_filename='output.gif'):
        """Tests the trained agent and creates a GIF animation."""
        # Creating path if it does not exist
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # Appending the file name to the path
        gif_filename = os.path.join(file_path, gif_filename)

        # Resetting the environment
        obs, _ = self.env.reset()

        # Collecting frames for GIF creation
        frames = []
        frames.append(self.env.render())  # Initial frame

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

        # Adding final frame
        frames.append(self.env.render())  # Final frame with label

        for i in range(len(frames)):
            original_height, original_width = frames[i].shape[:2]
            aspect_ratio = original_width / original_height

            # Assuming FRAME_SIZE is a tuple (new_width, new_height)
            new_width, new_height = FRAME_SIZE

            # If the original aspect ratio is greater than the new aspect ratio
            # it means the original width is greater than the original height.
            # So, we should set the new width based on the new height.
            if aspect_ratio > new_width / new_height:
                new_width = int(new_height * aspect_ratio)
            else:
                new_height = int(new_width / aspect_ratio)

            frames[i] = cv2.resize(frames[i], (new_width, new_height))

        # Saving frames as GIF
        imageio.mimsave(gif_filename, frames, duration=10)  # duration in seconds per frame

        print('\033[92mGIF saved to:\033[0m {}'.format(gif_filename))

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
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ, criterion=nn.SmoothL1Loss(), name="DoubleDQN"):
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
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ, criterion=nn.SmoothL1Loss(), name="DuelingDQN"):
        super().__init__(env, replay_buffer, target_update_freq, criterion, name, DuelingDQN)

class NoisyDQNAgent(DQNAgent):
    """ The Noisy DQN agent that interacts with the environment and inherits from the DQN agent """
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ, criterion=nn.SmoothL1Loss(), name="NoisyDQN"):
        super().__init__(env, replay_buffer, target_update_freq, criterion, name, NoisyDQN)

class DoubleDuelingDQNAgent(DQNAgent):
    """ The Double Dueling DQN agent that interacts with the environment and inherits from the DQN agent """
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ, criterion=nn.SmoothL1Loss(), name="DoubleDuelingDQN"):
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

class DoubleNoisyDQNAgent(DQNAgent):
    """ The Double Noisy DQN agent that interacts with the environment and inherits from the DQN agent """
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ, criterion=nn.SmoothL1Loss(), name="DoubleNoisyDQN"):
        super().__init__(env, replay_buffer, target_update_freq, criterion, name, NoisyDQN)

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