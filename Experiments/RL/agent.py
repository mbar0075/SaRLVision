from models import *
from utils import *

import os
import math
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim


class DQNAgent(object):
    """
        DQN Agent class.
    """
    def __init__(self, input_size, output_size, replay_buffer_size=10000, batch_size=32, gamma=0.99,
                 epsilon_start=1.0, epsilon_final=0.0001, epsilon_decay=0.999, target_update=10, learning_rate=1e-5, save_path="DQN_models/"):
        """
            Constructor of the DQNAgent class.
            
            Args:
                input_size: The input size of the model.
                output_size: The output size of the model.
                replay_buffer_size: The size of the replay buffer.
                batch_size: The batch size.
                gamma: The discount factor.
                epsilon_start: The initial value of epsilon.
                epsilon_final: The final value of epsilon.
                epsilon_decay: The decay of epsilon.
                target_update: The number of steps to update the target network.
                learning_rate: The learning rate.
                save_path: The path to save the network.
                
        """
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.episode_count = 0
        self.step_count = 0
        self.lr = learning_rate
        self.save_path = SAVE_MODEL_PATH + save_path

        # Creating the neural networks
        # The policy network is used to select the actions
        self.policy_net = DQN(input_size, output_size).to(self.device)

        # The target network is used to compute the target Q-values
        self.target_net = DQN(input_size, output_size).to(self.device)

        # Initializing the target network with the same weights as the policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Setting the target network in evaluation mode
        self.target_net.eval()

        # Setting the optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)


    def reset(self):
        """
            Resetting the agent.
        """
        self.episode_count += 1
        self.step_count = 0

    def select_action(self, state):
        """
        Selecting an action based on the current state.

        Args:
            state: The current state.

        Returns:
            The selected action (as a single integer).
        """
        # Converting state to a Torch tensor
        state = torch.tensor(state, device=self.device, dtype=FloatDType).unsqueeze(0)

        # Selecting an action based on the epsilon-greedy policy
        if random.random() > self.epsilon:
            with torch.no_grad():
                # Selecting the action with the highest Q-value
                q_values = self.policy_net(state)
                action = q_values.argmax(dim=1).item()  # Get the index of the maximum Q-value as the action
        else:
            # Selecting a random action
            action = random.randrange(self.output_size)

        # Returning the selected action
        return action
    
    def update_epsilon(self):
        """
            Updating epsilon.

            Formula:
                ε = ε_final + (ε_start - ε_final) * exp(-1. * step / ε_decay)
        """
        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                       math.exp(-1. * self.step_count / self.epsilon_decay)
        
    def update(self):
        """
            Updating the agent.
        """
        # Checking if the replay buffer contains enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        # Increasing the step count
        self.step_count += 1

        # Updating epsilon
        self.update_epsilon()

        # Sampling a batch of transitions from the replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)

        # Transposing the batch
        batch = Transition(*zip(*transitions))

        # Converting each element of the batch to a Torch tensor
        print(batch.state)
        state_batch = torch.tensor(batch.state, device=self.device, dtype=FloatDType)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=LongDType).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=FloatDType)
        next_state_batch = torch.tensor(batch.next_state, device=self.device, dtype=FloatDType)

        # Calculating the Q-values for the current states
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Calculating the Q-values for the next states
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()

        # Calculating the expected Q-values
        expected_q_values = reward_batch + (self.gamma * next_q_values)

        # Calculating the loss
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        # Backpropagating the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Updating the target network, copying all weights and biases in DQN
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_network(self):
        """
            Saving the network.
        """
        # Creating the folder if it does not exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Saving the network
        torch.save(self.policy_net, self.save_path + "policy_net.pt")
        torch.save(self.target_net, self.save_path + "target_net.pt")

        # Printing the path where the network is saved
        print("Network saved at: " + self.save_path)

    def load_network(self):
        """
            Loading the network.
        """
        # Checking if the network exists
        if torch.cuda.is_available():
            self.policy_net = torch.load(self.save_path + "policy_net.pt").to(self.device)
            self.target_net = torch.load(self.save_path + "target_net.pt").to(self.device)
        else:
            self.policy_net = torch.load(self.save_path + "policy_net.pt", map_location=torch.device('cpu'))
            self.target_net = torch.load(self.save_path + "target_net.pt", map_location=torch.device('cpu'))

        # Setting the target network in evaluation mode
        self.target_net.eval()

        # Printing the path where the network is loaded from
        print("Network loaded from: " + self.save_path)
        return self.policy_net, self.target_net

