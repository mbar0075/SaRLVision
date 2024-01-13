import torch
import random
import numpy as np
from collections import namedtuple, deque

# Setting the device to cpu as it was faster than gpu for this task
device = torch.device("cpu")#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Defining use_cuda as True if cuda is available, False otherwise.
use_cuda = torch.cuda.is_available()

# Printing the device
if use_cuda:
    print("CUDA is available! Using GPU for computations.")

# Defining the types of tensors that we will be using throughout the project.
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

FloatDType = torch.float32
LongDType = torch.long
BoolDType = torch.bool

SAVE_MODEL_PATH = "models/"


# The learning rate α ∈ (0, 1] controls how much we update our current value estimates towards newly received returns.
ALPHA = 1e-4
# Gamma refers to the discount factor γ ∈ [0, 1]. It quantifies how much importance is given to future rewards.
GAMMA = 0.5 #0.99
# The batch size is the number of training examples used in one iteration (that is, one gradient update) of training.
BATCH_SIZE = 128#256
# The buffer size is the number of transitions stored in the replay buffer, which the agent samples from to learn.
BUFFER_SIZE = 500#10000
# The minimum replay size is the minimum number of transitions that need to be stored in the replay buffer before the agent starts learning.
MIN_REPLAY_SIZE = 250#5000
# The maximum replay size is the maximum number of transitions that can be stored in the replay buffer.
MAX_REPLAY_SIZE = 50
# Epsilon start, epsilon end and epsilon decay are the parameters for the epsilon greedy exploration strategy.
EPS_START = 0.9
EPS_END = 0.0005
EPS_DECAY = 0.9#9
# The target update frequency is the frequency with which the target network is updated.
TARGET_UPDATE_FREQ = 5
# The success criteria is the average reward over the last 50 episodes that the agent must achieve to be considered successful.
SUCCESS_CRITERIA = 0.92

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'next_state'))

class Replay_Buffer():
    """
        The replay buffer stores the transitions that the agent observes, allowing us to reuse this data later.

        Args:
            env: The environment to interact with
            fullsize: The maximum size of the replay buffer
            minsize: The minimum size of the replay buffer before the agent starts learning
            batchsize: The batch size used for training
    """
    def __init__(self, env, fullsize, minsize, batchsize):
        self.env = env
        self.memory = deque(maxlen=fullsize) # Using a deque instead of a list for faster appends and pops
        self.rewards = deque(maxlen=MAX_REPLAY_SIZE)
        self.batchsize = batchsize
        self.minsize = minsize

    def append(self, transition):
        """ Appends a transition to the replay buffer """
        self.memory.append(transition)

    def sample_batch(self):
        """ Samples a batch of transitions from the replay buffer """
        batch = random.sample(self.memory, self.batchsize)
        batch = Transition(*zip(*batch))
        states = torch.from_numpy(np.array(batch.state, dtype=np.float32))
        actions = torch.from_numpy(np.array(batch.action, dtype=np.int64)).unsqueeze(1)
        rewards = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).unsqueeze(1)
        dones = torch.from_numpy(np.array(batch.done, dtype=np.bool8)).unsqueeze(1).to(torch.bool)
        next_states = torch.from_numpy(np.array(batch.next_state, dtype=np.float32))
        return states, actions, rewards, dones, next_states

    def initialize(self):
        """ Initializes the replay buffer by sampling transitions from the environment """
        # Resetting the environment
        obs, info = self.env.reset()

        # Sampling transitions until the replay buffer is full
        for _ in range(self.minsize):
            # Sampling a random action
            action = self.env.action_space.sample()

            # Taking a step in the environment
            new_obs, reward, terminated, truncated, info = self.env.step(action)

            # Setting done to terminated or truncated
            done = terminated or truncated

            # Creating a transition
            transition = Transition(obs, action, reward, done, new_obs)

            # Appending the transition to the replay buffer
            self.append(transition)

            # Resetting the observation
            obs = new_obs

            # Resetting the environment if the episode is done
            if done:
                self.env.reset()
        return self


def iou(bbox1, target_bbox):
        """
            Calculating the IoU between two bounding boxes.

            Formula:
                IoU(b, g) = area(b ∩ g) / area(b U g)

            Args:
                bbox1: The first bounding box.
                target_bbox: The second bounding box.

            Returns:
                The IoU between the two bounding boxes.

        """
        # Unpacking the bounding boxes
        x1, y1, w1, h1 = bbox1
        x_gt, y_gt, w_gt, h_gt = target_bbox

        # Calculating the coordinates of the intersection area of the two bounding boxes
        # xi1, yi1 is the top left coordinate of the intersection area
        # xi2, yi2 is the bottom right coordinate of the intersection area
        xi1 = max(x1, x_gt)
        yi1 = max(y1, y_gt)
        xi2 = min(x1 + w1, x_gt + w_gt) # x1 + w1 is the rightmost coordinate of bbox1, x_gt + w_gt is the rightmost coordinate of target_bbox
        yi2 = min(y1 + h1, y_gt + h_gt) # y1 + h1 is the bottommost coordinate of bbox1, y_gt + h_gt is the bottommost coordinate of target_bbox

        # Calculating the intersection area
        # Multiplying the width by the height gives us the area, but we have to make sure that the result is not negative
        inter_width = max(xi2 - xi1, 0)
        inter_height = max(yi2 - yi1, 0)
        inter_area = inter_width * inter_height

        # Calculating the bounding boxes areas
        # Multiplying the width by the height gives us the area
        box1_area = w1 * h1
        box2_area = w_gt * h_gt

        # Calculating the union area
        # Union area = area of the two bounding boxes - intersection area
        union_area = box1_area + box2_area - inter_area

        # Handling the case where union_area might be zero to avoid division by zero
        if union_area == 0:
            return 0.0

        # Calculating the IoU
        iou = inter_area / union_area

        # Returning the IoU
        return iou

def recall(bbox, target_bbox):
    """
        Calculating the recall between two bounding boxes.

        Formula:
            Recall(b, g) = area(b ∩ g) / area(g)

        Args:
            bbox: The first bounding box.
            target_bbox: The second bounding box.

        Returns:
            The recall between the two bounding boxes.
    """
    # Unpacking the bounding boxes
    x1, y1, w1, h1 = bbox
    x_gt, y_gt, w_gt, h_gt = target_bbox

    # Calculating the coordinates of the intersection area of the two bounding boxes
    # xi1, yi1 is the top left coordinate of the intersection area
    # xi2, yi2 is the bottom right coordinate of the intersection area
    xi1 = max(x1, x_gt)
    yi1 = max(y1, y_gt)
    xi2 = min(x1 + w1, x_gt + w_gt) # x1 + w1 is the rightmost coordinate of bbox1, x_gt + w_gt is the rightmost coordinate of target_bbox
    yi2 = min(y1 + h1, y_gt + h_gt) # y1 + h1 is the bottommost coordinate of bbox1, y_gt + h_gt is the bottommost coordinate of target_bbox

    # Calculating the intersection area
    # Multiplying the width by the height gives us the area, but we have to make sure that the result is not negative
    inter_width = max(xi2 - xi1, 0)
    inter_height = max(yi2 - yi1, 0)
    inter_area = inter_width * inter_height

    # Calculating the bounding boxes areas
    # Multiplying the width by the height gives us the area
    box2_area = w_gt * h_gt

    # Calculating the recall
    recall = inter_area / box2_area

    # Returning the recall
    return recall