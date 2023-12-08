import torch
import random
from collections import namedtuple

# Setting the device to cuda if cuda is available.
use_cuda = True
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

SAVE_MODEL_PATH = "Experiments/RL/models/"

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):
    """
        Replay Buffer that stores the transitions that the agent observes.
    
    """
    def __init__(self, capacity):
        """
            Constructor of the ReplayBuffer class.
        
            Args:
                capacity: The capacity of the ReplayBuffer.
                memory: The memory of the ReplayBuffer.
                position: The position of the ReplayBuffer.

        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Save a transition."""
        # If the memory is not full, we append the transition.
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        # Saving the transition at the current position.
        self.memory[self.position] = Transition(*args)

        # Incrementing the position.
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sampling a batch of transitions."""
        # Returning a batch of random transitions.
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returning the length of the memory."""
        return len(self.memory)
    



def iou(bbox1, target_bbox):
        """
            Calculating the IoU between two bounding boxes.

            Formula:
                IoU(b, g) = area(b ∩ g) / area(b U g

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