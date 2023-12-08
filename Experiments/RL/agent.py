from models import *
from utils import *

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

class Agent():
    def __init__(self, alpha=0.2, nu=3.0, threshold=0.5, num_episodes=15, load=False, feature_extractor=VGG16FeatureExtractor(), target_size=(224, 224), use_cuda=True):
        """
            Constructor of the Agent class.
        """
        self.BATCH_SIZE = 100
        self.GAMMA = 0.900
        self.EPS = 1
        self.TARGET_UPDATE = 1
        self.save_path = SAVE_MODEL_PATH
        screen_height, screen_width = target_size
        self.n_actions = 9

        self.feature_extractor = feature_extractor
        if not load:
            self.policy_net = DQN(screen_height, screen_width, self.n_actions)
        else:
            self.policy_net = self.load_network()
            
        self.target_net = DQN(screen_height, screen_width, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.feature_extractor.eval()
        if use_cuda:
          self.feature_extractor = self.feature_extractor.cuda()
          self.target_net = self.target_net.cuda()
          self.policy_net = self.policy_net.cuda()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-6)
        self.memory = ReplayBuffer(10000)
        self.steps_done = 0
        self.episode_durations = []
        
        self.alpha = alpha # Scaling factor of the reward [0, 1]
        self.nu = nu # Reward of Trigger
        self.threshold = threshold
        self.actions_history = []
        self.num_episodes = num_episodes
        self.actions_history += [[100]*9]*20

    def save_network(self):
        """
            Saving the network.
        """
        torch.save(self.policy_net, self.save_path + "policy_net.pt")
        torch.save(self.target_net, self.save_path + "target_net.pt")
        print("Network saved.")

    def load_network(self):
        """
            Loading the network.
        """
        if torch.cuda.is_available():
            return torch.load(self.save_path + "policy_net.pt")
        else:
            return torch.load(self.save_path + "policy_net.pt", map_location=torch.device('cpu'))
