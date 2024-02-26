#-------------------------------------------------------------------------------
# Name:        utils.py
# Purpose:     Defining the utility functions and classes for the SaRLVision.
#
# Author:      Matthias Bartolo <matthias.bartolo@ieee.org>
#
# Created:     February 24, 2024
# Copyright:   (c) Matthias Bartolo 2024-
# Licence:     All rights reserved
#-------------------------------------------------------------------------------
import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple, deque

# Setting the device to cpu as it was faster than gpu for this task
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#torch.device("cpu")

# Defining use_cuda as True if cuda is available, False otherwise.
use_cuda = torch.cuda.is_available()

# Printing the device
if use_cuda:
    print('\033[92m' + "CUDA is available! Using GPU for computations." + '\033[0m')

# Defining the types of tensors that we will be using throughout the project.
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# Defining the types of data that we will be using throughout the project.
FloatDType = torch.float32
LongDType = torch.long
BoolDType = torch.bool

# The path to the directory containing the saved models
SAVE_MODEL_PATH = "models/"
# Frame size for gif
FRAME_SIZE = (480 * 2, 360 * 2)
# The learning rate α ∈ (0, 1] controls how much we update our current value estimates towards newly received returns.
ALPHA = 1e-4
# Gamma refers to the discount factor γ ∈ [0, 1]. It quantifies how much importance is given to future rewards.
GAMMA = 0.99
# The batch size is the number of training examples used in one iteration (that is, one gradient update) of training.
BATCH_SIZE = 128
# The buffer size is the number of transitions stored in the replay buffer, which the agent samples from to learn.
BUFFER_SIZE = 10000
# The minimum replay size is the minimum number of transitions that need to be stored in the replay buffer before the agent starts learning.
MIN_REPLAY_SIZE = 250
# The maximum replay size is the maximum number of transitions that can be stored in the replay buffer.
MAX_REPLAY_SIZE = 50
# Epsilon start, epsilon end and epsilon decay are the parameters for the epsilon greedy exploration strategy.
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.999
# The target update frequency is the frequency with which the target network is updated.
TARGET_UPDATE_FREQ = 5
# The success criteria is the number of episodes the agent needs to solve the environment in order to consider the environment solved.
SUCCESS_CRITERIA_EPS = 50#100
# Success criteria for the the number of epochs the agent needs to solve the environment in order to consider the environment solved.
SUCCESS_CRITERIA_EPOCHS = 15
# Boolean Flag to determine which success criteria to use
USE_EPISODE_CRITERIA = False#True
# Environment Modes
TRAIN_MODE = 0
TEST_MODE = 1
# Exploration Modes
RANDOM_EXPLORE = 0
GUIDED_EXPLORE = 1
# Setting Exploration Mode
EXPLORATION_MODE = GUIDED_EXPLORE

# Defining the transition tuple
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
    def __init__(self, env, fullsize=BUFFER_SIZE, minsize=MIN_REPLAY_SIZE, batchsize=BATCH_SIZE):
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
        obs, _ = self.env.reset()

        # Sampling transitions until the replay buffer is full
        for _ in range(self.minsize):
            # Sampling a random action
            action = self.env.action_space.sample()

            # Taking a step in the environment
            new_obs, reward, terminated, truncated, _ = self.env.step(action)

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

    # Handling the case where there box2_area is 0
    if box2_area == 0:
        return 0.0
    
    # Calculating the recall
    recall = inter_area / box2_area

    # Returning the recall
    return recall

def calculate_best_iou(bounding_boxes, gt_boxes):
    """
        Calculating the best IoU between the bounding boxes and the ground truth boxes.

        Args:
            bounding_boxes: The predicted bounding boxes.
            gt_boxes: The ground truth bounding boxes.

        Returns:
            The best IoU between the bounding boxes and the ground truth boxes.
    """
    # Initializing the best IoU
    best_iou = -np.inf

    # Iterating through the bounding boxes
    for bbox in bounding_boxes:
        # Iterating through the ground truth boxes
        for gt_box in gt_boxes:
            # Calculating the IoU
            current_iou = iou(bbox, gt_box)

            # Updating the best IoU
            best_iou = max(best_iou, current_iou)

    # Returning the best IoU
    return best_iou

def calculate_best_recall(bounding_boxes, gt_boxes):
    """
        Calculating the best recall between the bounding boxes and the ground truth boxes.

        Args:
            bounding_boxes: The predicted bounding boxes.
            gt_boxes: The ground truth bounding boxes.

        Returns:
            The best recall between the bounding boxes and the ground truth boxes.
    """
    # Initializing the best recall
    best_recall = -np.inf

    # Iterating through the bounding boxes
    for bbox in bounding_boxes:
        # Iterating through the ground truth boxes
        for gt_box in gt_boxes:
            # Calculating the recall
            recall_score = recall(bbox, gt_box)

            # Updating the best recall
            best_recall = max(best_recall, recall_score)

    # Returning the best recall
    return best_recall

def calculate_precision_recall(bounding_boxes, gt_boxes, ovthresh):
    """
        Calculating the precision and recall using the Intersection over Union (IoU) and according to the threshold between the ground truths and the predictions.

        Args:
            bounding_boxes: The predicted bounding boxes.
            gt_boxes: The ground truth bounding boxes.
            ovthresh: The IoU threshold.

        Returns:
            precision (tp / (tp + fp))
            recall (tp / (tp + fn))
            f1 score (2 * (precision * recall) / (precision + recall))
            average IoU (sum of IoUs / number of bounding boxes)
            average precision (sum of precisions / number of bounding boxes)
    """
    # Retrieving the number of bounding boxes
    num_bounding_boxes = len(bounding_boxes)
    num_gt_boxes = num_bounding_boxes

    # Ensuring that the number of bounding boxes is the same as the number of ground truth boxes
    assert num_bounding_boxes == num_gt_boxes, "Evaluation Error: The number of bounding boxes must be the same as the number of ground truth boxes."

    # Initializing the true positives, false positives and false negatives
    tp = np.zeros(num_bounding_boxes)
    fp = np.zeros(num_bounding_boxes)
    fn = np.zeros(num_bounding_boxes)

    # Initializing the IoU, precision and recall
    ious = np.zeros(num_bounding_boxes)
    precision = np.zeros(num_bounding_boxes)
    recall = np.zeros(num_bounding_boxes)

    # Retrieving img_ids
    img_ids = list(bounding_boxes.keys())

    # Iterating through the bounding boxes
    for index in range(num_bounding_boxes):
        # Retrieving the bounding boxes
        prediction = bounding_boxes[img_ids[index]]
        target = gt_boxes[img_ids[index]]

        # Calculating the IoU
        ious[index] = calculate_best_iou(prediction, target)

        # Calculating the precision and recall
        if ious[index] > ovthresh:
            tp[index] = 1.0
        else:
            fp[index] = 1.0
            fn[index] = 1.0

    # Calculating the precision and recall
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    fn = np.cumsum(fn)

    # Calculating the precision and recall for each bounding box (finfo is used to avoid division by zero)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    recall = tp / np.maximum(tp + fn, np.finfo(np.float64).eps)

    # Calculating the f1 score
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Calculating the average IoU
    avg_iou = np.mean(ious)

    # Calculating the average precision
    avg_precision = np.mean(precision)

    # Returning the precision, recall, f1 score, average IoU and average precision
    return precision, recall, f1_score, avg_iou, avg_precision





#------------------------------------------------------------------------------------------------
def voc_ap(rec, prec, voc2007=False):
    """
    Calculating the Average Precision (AP) and Recall.

        Args:
            rec: Array of recall values.
            prec: Array of precision values.
            voc2007: Boolean flag indicating whether to use the method recommended by the PASCAL VOC 2007 paper (11-point method).

        Returns:
            The average precision (AP).

        More information:
        - If voc2007 is True, then the method recommended by the PASCAL VOC 2007 paper (11-point method) is used.
        - If voc2007 is False, then the method recommended by the PASCAL VOC 2010 paper is used.
    """
    # Ensuring inputs are numpy arrays
    rec = np.array(rec)
    prec = np.array(prec)

    if voc2007:
        # PASCAL VOC 2007 11-point method
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            # Finding the maximum precision at recall levels greater than or equal to the threshold
            p = np.max(prec[rec >= t]) if np.sum(rec >= t) > 0 else 0
            # Calculating AP using the 11-point method
            ap += p / 11.0
    else:
        # PASCAL VOC 2010 method
        # Concatenating arrays with 0 and 1 to ensure that precision is interpolated correctly
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # Interpolating precision at different recall levels by taking the maximum precision at a recall level greater than the current recall level
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Calculating the average precision by integrating the precision-recall curve
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def calculate_class_detection_metrics(current_class, bounding_boxes, gt_boxes, ovthresh):
    """
        Calculating the VOC detection metric.

        Args:
            current_class: The current class/label.
            bounding_boxes: The predicted bounding boxes.
            gt_boxes: The ground truth bounding boxes.
            ovthresh: The IoU threshold.

        Returns:
            The average precision.
    """
    # Calculating the precision and recall
    prec, rec, f1_score, avg_iou, avg_precision = calculate_precision_recall(bounding_boxes, gt_boxes, ovthresh)

    # Calculating the average precision
    ap = voc_ap(rec, prec)

    # Returning all the metrics in a dictionary
    return {"class": current_class, "average_iou": avg_iou, "average_precision": avg_precision, "average_precision_voc": ap, "iou_threshold": ovthresh, "num_images": len(bounding_boxes)}, {"precision": prec, "recall": rec, "f1_score": f1_score,}

# def plot_precision_recall_curve_for_all_classes(pre_rec_f1, title="Precision-Recall Curve", save_path=None, figsize=(20, 10)):
#     """
#         Plotting the precision-recall curve for all the classes.

#         Args:
#             pre_rec_f1: The dataframe containing the precision, recall, and F1-score for each class.
#             title: The title of the plot.
#             save_path: The path to save the plot.

#         Returns:
#             None
#     """
#     # Initializing the figure
#     plt.figure(figsize=figsize)

#     # Using the seaborn style
#     plt.style.use("seaborn")

#     # Iterating through the unique classes in pre_rec_f1
#     for current_class in pre_rec_f1['class'].unique():
#         # Selecting the rows for the current class
#         df = pre_rec_f1[pre_rec_f1['class'] == current_class]

#         # Retrieving the precision and recall
#         precision = df["precision"].values
#         recall = df["recall"].values

#         # Plotting the precision-recall curve
#         plt.plot(recall, precision, label=f"{current_class}")

#     # Setting the title and labels
#     plt.title(title)
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")

#     # Setting the legend
#     plt.legend()

#     plt.tight_layout()

#     # Saving the plot if a save path is provided
#     if save_path is not None:
#         plt.savefig(save_path)

#     # Showing the plot
#     plt.show()

def calculate_detection_metrics(results_path, save_path=None, threshold_list=[0.5]): # list(np.arange(0.5, 0.95, 0.05))):
    """
    Calculating the detection metrics for all the classes.

        Args:
            results_path: Path to the directory containing detection results.
            threshold_list: List of IoU thresholds to evaluate detection metrics.

        Returns:
            dfs: An array of pandas dataframes containing the detection metrics for each class at given IoU thresholds.
            mAps: The mean average precision for each class at given IoU thresholds.
            pre_rec_f1: Precision, recall, and F1-score for each class at given IoU thresholds.
    """
    # Declaring the results dictionary
    results = {}

    # Loading all files in the results path directory
    for file in os.listdir(results_path):
        # Loading the results
        current_results = np.load(os.path.join(results_path, file), allow_pickle=True).item()

        # Retrieving the class
        current_class = current_results["class"]

        # Storing the results in the results dictionary
        results[current_class] = current_results

    # Retrieving the classes
    classes = list(results.keys())

    # Initializing list of dataframes to store the detection metrics for each class and mean average precision
    dfs = []
    mAps = {}
    pre_rec_f1 = {}

    # Iterating through the threshold values
    for ovthresh in threshold_list:
        # Storing the average precision for each class at given IoU thresholds
        aps = {}

        pre_rec_f1[ovthresh] = {}

        # Creating a dictionary to store the detection metrics
        detection_metrics_dict = {}

        # Iterating through the classes
        for current_class in classes:
            # Retrieving the bounding boxes and ground truth boxes
            bounding_boxes = results[current_class]["bounding_boxes"]
            gt_boxes = results[current_class]["gt_boxes"]

            # Calculating the detection metrics
            detection_metrics, pre_rec_f1[ovthresh][current_class] = calculate_class_detection_metrics(current_class, bounding_boxes, gt_boxes, ovthresh)

            # Adding the detection metrics to the detection metrics dictionary
            detection_metrics_dict[current_class] = detection_metrics

            # Storing the average precision for each class at given IoU thresholds
            aps[current_class] = detection_metrics["average_precision_voc"]

        # Calculating the mean average precision for each class at given IoU thresholds
        mAps[ovthresh] = np.mean(list(aps.values()))

        # Transforming the detection metrics dictionary into a dataframe
        df = pd.DataFrame(detection_metrics_dict).T

        # Removing class column
        df = df.drop(columns=["class"])

        print(df)
        dfs.append(df)

    # If save_path is provided, save the results
    if save_path is not None:
        # If directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Saving the results in a csv file
        for i, df in enumerate(dfs):
            df.to_csv(os.path.join(save_path, f"detection_metrics_{threshold_list[i]}.csv"))

        # Saving the mean average precision in a csv file
        with open(os.path.join(save_path, "mean_average_precision.csv"), "w") as f:
            for key in mAps.keys():
                f.write("%s,%s\n"%(key,mAps[key]))

    # Returning the list of dataframes and the mean average precision
    return dfs, mAps

