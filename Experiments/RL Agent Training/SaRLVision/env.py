#-------------------------------------------------------------------------------
# Name:        env.py
# Purpose:     Defining the environment for SaRLVision.
#
# Author:      Matthias Bartolo <matthias.bartolo@ieee.org>
#
# Created:     February 24, 2024
# Copyright:   (c) Matthias Bartolo 2024-
# Licence:     All rights reserved
#-------------------------------------------------------------------------------
import cv2
import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
import torch
import torch.nn as nn
import torchvision 
from torchvision import datasets
import matplotlib.pyplot as plt
from SaRLVision.utils import *
from SaRLVision.models import *
import time
import math
import colorsys
import pygame
import os
import sys
import json

# Importing SaRa (Saliency Ranking (Seychell et al. IEEE IC3D))
import SaRLVision.SaRa.saraRC1 as sara
# Saliency Map Generator (Itti Model)
generator = 'itti' 
# Optimal Grid Size for the Saliency Map
GRID_SIZE =  9
# Boolean flag to use the SARA model for initial bounding box prediction
USE_SARA = False#True

# Importing Mask To Annotation (Creating Annotations from Masks (IEEE ISM 2023))
import SaRLVision.MaskToAnnotation.coco as coco
import SaRLVision.MaskToAnnotation.yolo as yolo
import SaRLVision.MaskToAnnotation.vgg as vgg
import SaRLVision.MaskToAnnotation.annotation_helper as ah

# Defining constants
# The number of actions is 9 because we have 8 actions + 1 trigger action (move right, move left, move up, move down, make bigger, make smaller, make fatter, make taller, trigger).
NUMBER_OF_ACTIONS = 9
# The size of the action history which controls the number of actions that the environment will remember.
ACTION_HISTORY_SIZE = 10
# The action history is a binary matrix of size NUMBER_OF_ACTIONS x ACTION_HISTORY_SIZE, where each row corresponds to an action and each column corresponds to a step.
ACTION_HISTORY = [[0]*NUMBER_OF_ACTIONS]*ACTION_HISTORY_SIZE
# The trigger reward is 3.0 as mentioned in multiple papers.
NU = 3.0
# The threshold denotes the IoU threshold for the trigger action positive or negative reward (0.6 for training, 0.5 for testing).
THRESHOLD = 0.6
# The alpha is the scaling factor for bounding box movements (large alpha will make the bounding box move faster).
ALPHA = 0.2
# The maximum number of steps is 200 as mentioned in the paper.
MAX_STEPS = 200
# The number of steps before the trigger is 40 as mentioned in the paper.
TRIGGER_STEPS = 40
# The environment render mode is used to render the environment in different modes (None, human, trigger_image, bbox, rgb_array).
RENDER_MODE = None
# The feature extractor is the CNN used to extract the features of the image in the environment.
FEATURE_EXTRACTOR = VGG16FeatureExtractor()
# The target size is the size of the image that will be used as input to the feature extractor.
TARGET_SIZE = VGG16_TARGET_SIZE
# The classifier is the CNN used to classify the image ROI in the environment.
CLASSIFIER = ResNet50V2()
# The classifier target size is the size of the image that will be used as input to the classifier.
CLASSIFIER_TARGET_SIZE = RESNET50_TARGET_SIZE
# The reward function is the function used to calculate the reward of the environment.
REWARD_FUNC = calculate_best_iou
# The environment mode is used to specify whether the environment is in training or testing mode (0 for training, 1 for testing).
ENV_MODE = TRAIN_MODE
# The use dataset is used to specify whether the environment will use the dataset or not (None for not using the dataset, or the path of the dataset ('PascalVOC2007_2012Dataset')).
USE_DATASET = 'PascalVOC2007_2012Dataset'
# The dataset year is used to specify the year of the dataset (2007, 2012, or 2007+2012).
DATASET_YEAR = '2007'
# The dataset image set is used to specify the image set of the dataset (train, val, test).
DATASET_IMAGE_SET = 'train'
# The object configuration is used to specify whether the environment will use single object or multiple objects (0 for single object, 1 for multiple objects).
SINGLE_OBJ = 0
MULTI_OBJ = 1
# The object configuration is used to specify whether the environment will use single object or multiple objects (0 for single object, 1 for multiple objects).
OBJ_COFIGURATION = SINGLE_OBJ
# The allow classification is used to specify whether the environment will allow classification or not (True for allowing classification, False for not allowing classification).
ALLOW_CLASSIFICATION = False


"""
    Detection Environment
    
    The environment is used to define the environment for the SaRLVision agent. Environment inherits from the gymnaisum environment.
"""
class DetectionEnv(Env):
    # Metadata for the environment
    metadata = {"render_modes": ["human", "rgb_array", "bbox", "trigger_image"], "render_fps": 3}
    
    def __init__(self, env_config={}):
        """
            Constructor of the DetectionEnv class.

            Args:
                - env_config: Dictionary that contains the configuration of the environment.
                
            Parameters (env_config):
                - 'dataset': The path of the dataset ('PascalVOC2007_2012Dataset').
                - 'dataset_year': The year of the dataset (2007, 2012, or 2007+2012).
                - 'dataset_image_set': The image set of the dataset (train, val, test).
                - 'obj_configuration': Whether the environment will use single object or multiple objects (0 for single object, 1 for multiple objects).
                - 'current_class': The current class to be used in the environment.
                - 'image': The image to be used in the environment.
                - 'original_image': The original image to be used in the environment.
                - 'target_gt_boxes': The target bounding boxes to be used in the environment.
                - 'target_size': The size of the image that will be used as input to the feature extractor.
                - 'use_sara': Whether the environment will use the SARA model for initial bounding box prediction (True for using the SARA model, False for not using the SARA model).
                - 'feature_extractor': The CNN used to extract the features of the image in the environment.
                - 'max_steps': The maximum number of steps in the environment.
                - 'trigger_steps': The number of steps before the trigger in the environment.
                - 'alpha': The scaling factor for bounding box movements in the environment.
                - 'nu': The trigger reward in the environment.
                - 'threshold': The IoU threshold for the trigger action positive or negative reward in the environment.
                - 'classifier': The CNN used to classify the image ROI in the environment.
                - 'classifier_target_size': The size of the image that will be used as input to the classifier.
                - 'allow_classification': Whether the environment will allow classification or not (True for allowing classification, False for not allowing classification).
                - 'render_mode': The render mode of the environment (None, human, trigger_image, bbox, rgb_array).
                
            Returns:
                - None
        """
        super(DetectionEnv, self).__init__()
        # Initialising the current ground truth bounding boxes
        self.current_gt_bboxes = []
        self.current_gt_index =0

        # dataset variables
        if 'dataset' in env_config:
            self.use_dataset = env_config['dataset']
            del env_config['dataset']
        else:
            self.use_dataset = USE_DATASET

        if 'dataset_year' in env_config:
            self.dataset_year = env_config['dataset_year']
            del env_config['dataset_year']
        else:
            self.dataset_year = DATASET_YEAR

        if 'dataset_image_set' in env_config:
            self.dataset_image_set = env_config['dataset_image_set']
            del env_config['dataset_image_set']
        else:
            self.dataset_image_set = DATASET_IMAGE_SET

        # Variable to hold the number of epochs
        self.epochs = 0

        # Variables to hold the classes and the current class
        self.classes = []
        self.current_class = None
        self.class_image_index = 0
        self.total_images = 0

        # For environment mode
        self.env_mode = ENV_MODE

        # Setting the object configuration
        if 'obj_configuration' in env_config:
            self.obj_configuration = env_config['obj_configuration']
            del env_config['obj_configuration']
        else:
            self.obj_configuration = OBJ_COFIGURATION

        # Loading the dataset if self.use_dataset is not None
        if self.use_dataset is not None:
            # Loading the training dataset if the dataset year is 2007+2012, else loading user specified dataset
            if self.dataset_year != '2007+2012':
                self.dataset = self.load_pascal_voc_dataset(path=self.use_dataset, year=self.dataset_year, image_set=self.dataset_image_set)
            else:
                self.dataset = self.load_training_dataset(path=self.use_dataset, image_set=self.dataset_image_set)

            # Extracting the current class
            if 'current_class' in env_config:
                self.current_class = env_config['current_class']
                if self.current_class not in self.classes:
                    raise ValueError('Current class not in the dataset, possible classes are: ' + str(self.classes))
                del env_config['current_class']
            else:
                # Extracting the first class
                self.current_class = self.classes[0]

            print('\033[37m' + "Current Class: " + self.current_class + '\033[0m')

            # For Evaluation, if the environment mode is testing, then create the evaluation results dictionary
            if self.env_mode == TEST_MODE: # Testing mode
                self.evaluation_results = {'class': self.current_class, 'gt_boxes': {}, 'bounding_boxes': {}, 'total_images': len(self.dataset[self.current_class]), 'labels': {}, 'confidences': {}}

            # Extracting the first image
            self.extract()
            # Decrementing the class image index by 1, since the first image is already extracted
            self.class_image_index -= 1
        else:
            # Initialising image, the original image which will be used as a visualisation, the target bounding box, the height and the width of the image.
            if 'image' not in env_config:
                raise ValueError('Image is required for Environment Creation')
            self.image = env_config['image']
            del env_config['image']

            if 'original_image' not in env_config:
                raise ValueError('Original image is required for Environment Creation')
            self.original_image = env_config['original_image']
            del env_config['original_image']

            # Extracting the current ground truth bounding boxes (multiple objects)
            if 'target_gt_boxes' not in env_config:
                raise ValueError('Target bounding boxes is required for Environment Creation')
            self.current_gt_bboxes = env_config['target_gt_boxes']
            self.target_bbox = self.current_gt_bboxes[self.current_gt_index]
            del env_config['target_gt_boxes']

        # Extracting the height and the width of the image.
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        # Extracting the target size of the image.
        if 'target_size' in env_config:
            self.target_size = env_config['target_size']
            del env_config['target_size']
        else:
            self.target_size = TARGET_SIZE

        # Extracting the use_sara variable
        if 'use_sara' in env_config:
            self.use_sara = env_config['use_sara']
            del env_config['use_sara']
        else:
            self.use_sara = USE_SARA

        # Initialising the actions history and the number of episodes.
        self.actions_history = []
        self.num_episodes = 0
        self.actions_history += ACTION_HISTORY

        # Initialising the bounding box of the image.
        self.bbox = [0, 0, self.width, self.height]

        # If use_sara is True, then invoke the generate_initial_bbox function
        if self.use_sara:
            self.bbox = self.generate_initial_bbox()

        # Initialising the feature extractor and the transform method.
        if 'feature_extractor' in env_config:
            self.feature_extractor = env_config['feature_extractor']
            del env_config['feature_extractor']
        else:
            self.feature_extractor = FEATURE_EXTRACTOR

        # Setting the feature extractor to the device (for GPU or CPU usage)
        self.feature_extractor.to(device)

        # Setting the transform method to the device (for GPU or CPU usage)
        self.transform = transform_input(self.image, self.target_size)

        # Initialising the action space and the observation space.
        # Action space is 9 because we have 8 actions + 1 trigger action (move right, move left, move up, move down, make bigger, make smaller, make fatter, make taller, trigger).
        self.action_space = gym.spaces.Discrete(NUMBER_OF_ACTIONS)

        # Initialising the observation space.
        # Calculating the size of the state vector.
        state = self.get_state()
        # The observation space will be the features of the image concatenated with the history of the actions (size of the feature vector + size of the history vector).
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=10.0, # Since the values of the features are between 0 and 10.
            shape=(state.shape[1],),
            dtype=np.float32
        )

        # Setting terminated and truncated to False.
        self.terminated = False
        self.truncated = False
        
        # Initialising the maximum number of steps, the current step, the scaling factor of the reward, the reward of the trigger, the cumulative reward, the threshold, the actions history and the number of episodes.
        if 'max_steps' in env_config:
            self.max_steps = env_config['max_steps']
            del env_config['max_steps']
        else:
            self.max_steps = MAX_STEPS

        # Initialising the number of triggers.
        if 'trigger_steps' in env_config:
            self.trigger_steps = env_config['trigger_steps']
            del env_config['trigger_steps']
        else:
            self.trigger_steps = TRIGGER_STEPS

        # If the object configuration is single object, then the maximum number of steps will be the trigger steps
        if self.obj_configuration == SINGLE_OBJ:
            self.max_steps = self.trigger_steps

        # Setting the number of triggers to 0
        self.no_of_triggers = 0

        # Scaling factor for bounding box movements
        if 'alpha' in env_config:
            self.alpha = env_config['alpha']
            del env_config['alpha']
        else:
            self.alpha = ALPHA

        # Reward of Trigger
        if 'nu' in env_config:
            self.nu = env_config['nu']
            del env_config['nu']
        else:
            self.nu = NU

        # Initialising the step counnt, the cumulative reward, and truncated to False.
        self.step_count = 0
        self.cumulative_reward = 0
        self.truncated = False

        # Initialising the threshold.
        if 'threshold' in env_config:
            self.threshold = env_config['threshold']
            del env_config['threshold']
        else:
            self.threshold = THRESHOLD

        # Classification part
        # Defining the classification dictionary (label, confidence, bounding box, color).
        self.classification_dictionary = {'label': [], 'confidence': [], 'bbox': [], 'color': []}

        # Initialising the classifier.
        if 'classifier' in env_config:
            self.classifier = env_config['classifier']
            del env_config['classifier']
        else:
            self.classifier = CLASSIFIER

        # Initialising flag to allow classification
        if 'allow_classification' in env_config:
            self.allow_classification = env_config['allow_classification']
            del env_config['allow_classification']
        else:
            self.allow_classification = ALLOW_CLASSIFICATION
        
        # Initialising the classifier target size.
        if 'classifier_target_size' in env_config:
            self.classifier_target_size = env_config['classifier_target_size']
            del env_config['classifier_target_size']
        else:
            self.classifier_target_size = CLASSIFIER_TARGET_SIZE

        # Rendering part (Retrieving a random color for the bounding box).
        self.color = self.generate_random_color()

        # Initialising the render mode.
        if 'render_mode' in env_config:
            self.render_mode = env_config['render_mode']
            del env_config['render_mode']
        else:
            self.render_mode = RENDER_MODE

        # Ensuring that the render mode is in the metadata render modes.
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        # Initialising the window size, the window and the clock for rendering.
        self.window_size = (self.width, self.height)
        self.window = None
        self.clock = None
        self.is_render = False

        # If the render mode is not None, then we set the is_render to True and initialise the pygame window.
        if self.render_mode is not None:
            self.is_render = True
            pygame.init()
            self.render_mode = self.render_mode
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Detection Environment")
            icon = pygame.image.load("SaRLVision/Icon/SaRLVisionIcon.png")
            pygame.display.set_icon(icon)
            self.clock = pygame.time.Clock()

        # Setting the current action to None.
        self.current_action = None

        # Segmentation part
        # Defining the segmentation dictionary (bounding boxes, masks, names, labels, colors).
        self.segmentation_dictionary = {'bboxes': [], 'masks': [], 'names': [], 'labels': [], 'colors': []}
        pass

    def train(self):
        """
            Function that sets the environment mode to training.
        """
        # Setting the environment mode to training.
        self.env_mode = TRAIN_MODE
        pass

    def test(self):
        """
            Function that sets the environment mode to testing.
        """
        # Setting the environment mode to testing.
        self.env_mode = TEST_MODE

        # For Classification, if the environment mode is testing, then allow classification
        self.allow_classification = True

        # Checking if the classification dictionary is empty, then we get the labels
        if self.classification_dictionary['label'] == [] and self.classification_dictionary['bbox'] != []:
            self.get_labels()

        # For Evaluation, if the environment mode is testing, then create the evaluation results dictionary
        if self.use_dataset is not None:
            self.evaluation_results = {'class': self.current_class, 'gt_boxes': {}, 'bounding_boxes': {}, 'total_images': len(self.dataset[self.current_class]), 'labels': {}, 'confidences': {}}
        pass

    def eval(self):
        """
            Function that sets the environment mode to testing.
        """
        # Setting the environment mode to testing.
        self.env_mode = TEST_MODE

        # For Classification, if the environment mode is testing, then allow classification
        self.allow_classification = True

        # Checking if the classification dictionary is empty, then we get the labels
        if self.classification_dictionary['label'] == [] and self.classification_dictionary['bbox'] != []:
            self.get_labels()

        # For Evaluation, if the environment mode is testing, then create the evaluation results dictionary
        if self.use_dataset is not None:
            self.evaluation_results = {'class': self.current_class, 'gt_boxes': {}, 'bounding_boxes': {}, 'total_images': len(self.dataset[self.current_class]), 'labels': {}, 'confidences': {}}
        pass

    def calculate_reward(self, current_state, previous_state, target_bbox, reward_function=REWARD_FUNC):
        """
            Calculating the reward.

            Input:
                - Current state
                - Previous state
                - Target bounding box
                - Reward function

            Output:
                - Reward
        """
        # Calculating the IoU between the current state and the target bounding box.
        iou_current = reward_function(current_state, target_bbox)
 
        # Calculating the IoU between the previous state and the target bounding box.
        iou_previous = reward_function(previous_state, target_bbox)

        # Calculating the reward.
        reward = iou_current - iou_previous

        # Enabling binary reward in the range of {-1, 1}
        if reward > 0:
            reward = 1
        elif reward <= 0: # Including the equal to zero case as to mitigate redundant actions such continuing to maximise the bounding box
            reward = -1

        # Returning the reward.
        return reward
    
    def calculate_trigger_reward(self, current_state, target_bbox, reward_function=REWARD_FUNC):
        """
            Calculating the reward.

            Input:
                - Current state
                - Target bounding box
                - Reward function

            Output:
                - Reward
        """
        # Calculating the IoU between the current state and the target bounding box.
        iou_current = reward_function(current_state, target_bbox)

        # Calculating the reward.
        reward = iou_current

        # If the reward is larger or equal to the threshold, then return the reward * 2 * nu (inspired by the paper).
        # The times reward is extra, to give the agent the incentive to find better bounding boxes
        if reward >= self.threshold:
            return self.nu * 2 * reward
        
        # Else return -1 * nu.
        # Multiplying the negative reward by the IoU is not necessary, as doing so would be downscaling the negative reward, which makes the agent trigger on IoUs lower than the threshold.
        return -1*self.nu
    
    def get_features(self, image, dtype=FloatTensor):
        """
            Getting the features of the image.

            Input:
                - Image
                - Data type

            Output:
                - Features of the image
        """
        # Transforming the image.
        image = transform_input(image, target_size=self.target_size)

        # Retrieving the features of the image (unsqueeze is added since it is expecting a batch, and squeeze is added to remove the batch dimension).
        features = self.feature_extractor(image.unsqueeze(0).to(device)).squeeze(0)

        # Returning the features.
        return features.data
    
    def get_state(self, dtype=FloatDType):
        """
            Getting the state of the environment.

            Args:
                - dtype: Data type

            Output:
                - State of the environment
        """
        # Extracting current bounding box
        bbox = self.bbox

        # Extracting current image
        image = self.image.copy()

        # Cropping the image based on the bounding box
        image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # Transforming the image (So what is being fed to the feature extractor is the region of intesrest denoted by the current bounding box)
        image = transform_input(image, target_size=self.target_size)

        # Retrieving the features of the image.
        features = self.get_features(image)

        # Transposing the features and detaching it from the GPU (view is added to flatten the tensor and detach is added to remove the gradient from the tensor)
        features = features.view(1, -1).detach().cpu()

        # Flattenning the action history and converting it to a tensor of type float (view is added to flatten the tensor and detach is added to remove the gradient from the tensor)
        action_history = torch.tensor(self.actions_history, dtype=dtype).flatten().view(1, -1)

        # Concatenating the features and the action history (1  is beiing used to specify the dimension along which the tensors are concatenated).
        state = torch.cat((action_history, features), 1)

        # Returning the state.
        return state.numpy()
    
    def update_history(self, action):
        """
            Function that updates the history of the actions by adding the last one.
            It is creating a history vector of size 9, where each element is 0 except the one corresponding to the action performed.
            It is then shifting the history vector by one and adding the new action vector to the history vector.

            Input:
                - Last action performed

            Output:
                - History of the actions
        """
        # Creating the action vector.
        action_vector = [0] * NUMBER_OF_ACTIONS
        action_vector[action] = 1

        # Retrieving the size of the history list.
        size_history_list = len(self.actions_history)

        # If the size of the history list is smaller than the number of actions, we add the action vector to the history vector.
        if size_history_list < NUMBER_OF_ACTIONS:
            self.actions_history.append(action_vector)
        else:
            # Else we shift the history list by one and we add the action vector to the history vector.
            for i in range((NUMBER_OF_ACTIONS-1),0,-1):
                self.actions_history[i] = self.actions_history[i-1].copy()
            self.actions_history[0] = action_vector

        # Returning the history of the actions.
        return self.actions_history

    def transform_action(self, action):
        """
            Function that applies the action to the image.

            Actions:
                - 0: Move right
                - 1: Move left
                - 2: Move up
                - 3: Move down
                - 4: Make bigger
                - 5: Make smaller
                - 6: Make fatter
                - 7: Make taller

            Input:
                - Action to apply

            Output:
                - Bounding box of the image
        
        """
        # Retrieving the bounding box of the image.
        bbox = self.bbox

        # Retrieving the coordinates of the bounding box.
        xmin, xmax, ymin, ymax = bbox[0], bbox[2], bbox[1], bbox[3]

        # Calculating the alpha_h and alpha_w mentioned in the paper (scaling factor for bounding box movements).
        alpha_h = int(self.alpha * (  ymax - ymin ))
        alpha_w = int(self.alpha * (  xmax - xmin ))

        # If the action is 0, we move the bounding box to the right.
        if action == 0:
            xmin += alpha_w
            xmax += alpha_w
        # If the action is 1, we move the bounding box to the left. 
        elif action == 1:
            xmin -= alpha_w
            xmax -= alpha_w
        # If the action is 2, we move the bounding box up.
        elif action == 2:
            ymin -= alpha_h
            ymax -= alpha_h
        # If the action is 3, we move the bounding box down.
        elif action == 3:
            ymin += alpha_h
            ymax += alpha_h
        # If the action is 4, we make the bounding box bigger.
        elif action == 4:
            ymin -= alpha_h
            ymax += alpha_h
            xmin -= alpha_w
            xmax += alpha_w
        # If the action is 5, we make the bounding box smaller.
        elif action == 5:
            ymin += alpha_h
            ymax -= alpha_h
            xmin += alpha_w
            xmax -= alpha_w
        # If the action is 6, we make the bounding box fatter.
        elif action == 6:
            ymin += alpha_h
            ymax -= alpha_h
        # If the action is 7, we make the bounding box taller.
        elif action == 7:
            xmin += alpha_w
            xmax -= alpha_w

        # Returning the bounding box, whilst ensuring that the bounding box is within the image dimensions.
        return [self.rewrap(xmin, self.width), self.rewrap(ymin, self.height), self.rewrap(xmax, self.width), self.rewrap(ymax, self.height)]
    
    def get_actions(self):
        """
            Function that prints the name of the actions.
        """
        print('\033[1m' + "Actions:" + '\033[0m')
        print('\033[31m' + "0: Move right → " + '\033[0m')
        print('\033[32m' + "1: Move left ←" + '\033[0m')
        print('\033[33m' + "2: Move up ↑" + '\033[0m')
        print('\033[34m' + "3: Move down ↓" + '\033[0m')
        print('\033[35m' + "4: Make bigger +" + '\033[0m')
        print('\033[36m' + "5: Make smaller -" + '\033[0m')
        print('\033[37m' + "6: Make fatter W" + '\033[0m')
        print('\033[38m' + "7: Make taller H" + '\033[0m')
        print('\033[1m' + "8: Trigger T" + '\033[0m')
        pass

    def decode_action(self, action):
        """
            Function that decodes the action.

            Input:
                - Action to decode

            Output:
                - Unique print of the action
        """
        if action == 0:
            print('\033[31m' + "Action: Move right →" + '\033[0m')
        elif action == 1:
            print('\033[32m' + "Action: Move left ←" + '\033[0m')
        elif action == 2:
            print('\033[33m' + "Action: Move up ↑" + '\033[0m')
        elif action == 3:
            print('\033[34m' + "Action: Move down ↓" + '\033[0m')
        elif action == 4:
            print('\033[35m' + "Action: Make bigger +" + '\033[0m')
        elif action == 5:
            print('\033[36m' + "Action: Make smaller -" + '\033[0m')
        elif action == 6:
            print('\033[37m' + "Action: Make fatter W" + '\033[0m')
        elif action == 7:
            print('\033[38m' + "Action: Make taller H" + '\033[0m')
        elif action == 8:
            print('\033[1m' + "8: Trigger T" + '\033[0m')
        pass

    def rewrap(self, coordinate, size):
        """
            Function that rewrap the coordinate if it is out of the image.

            Input:
                - Coordinate to rewrap
                - Size of the image

            Output:
                - Rewrapped coordinate
        """
        return min(max(0, coordinate), size)
    
    def get_info(self):
        """
            Function that returns the information of the environment.

            Output:
                - Information dictionary of the environment
        """
        return {
            'target_bbox': self.target_bbox,
            'height': self.height,
            'width': self.width,
            'target_size': self.target_size,
            'max_steps': self.max_steps,
            'step_count': self.step_count,
            'alpha': self.alpha,
            'nu': self.nu,
            'cumulative_reward': self.cumulative_reward,
            'threshold': self.threshold,
            'actions_history': self.actions_history,
            'num_episodes': self.num_episodes,
            'bbox': self.bbox,
            'feature_extractor': self.feature_extractor,
            'transform': self.transform,
            'iou': calculate_best_iou([self.bbox], self.current_gt_bboxes),
            'recall': calculate_best_recall([self.bbox], self.current_gt_bboxes),
            'gt_bboxes': self.current_gt_bboxes,
            'threshold': self.threshold,
            'classification_dictionary': self.classification_dictionary,
            'env_mode': self.env_mode,
            'use_dataset': self.use_dataset,
            'dataset_year': self.dataset_year,
            'dataset_image_set': self.dataset_image_set,
            'epochs': self.epochs,
            'classes': self.classes,
            'current_class': self.current_class,
        }
    
    def generate_random_color(self, threshold=0.3):
        """
            Function that generates a random color.

            Input:
                - Threshold

            Output:
                - Random color
        """
        # Generating a random color.
        red, green, blue = random.uniform(-threshold * 255, threshold * 255), random.uniform(-threshold * 255, threshold * 255), random.uniform(-threshold * 255, threshold * 255)

        # Setting the colors to integers.
        red, green, blue = int(red), int(green), int(blue)

        # Making the color a bright color.
        hue = random.uniform(0, 360)

        # Converting the HSV color to RGB
        hsv_color = (hue / 360, 1, 1)
        random_rgb = tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*hsv_color))

        # Returning the new color
        return random_rgb
    
    def reset(self, env_config={}, seed=None, options=None):
        """
            Function that resets the environment.

            Args:
                - env_config: Dictionary that contains the configuration of the environment.
                - seed: Seed for the environment.
                - options: Options for the environment.
                
            Parameters (env_config):
                - 'image': The image to be used in the environment.
                - 'original_image': The original image to be used in the environment.
                - 'target_bbox': The target bounding box to be used in the environment.
                - 'target_gt_boxes': The target bounding boxes to be used in the environment.
                - 'alpha': The scaling factor for bounding box movements in the environment.
                - 'nu': The trigger reward in the environment.
                - 'threshold': The IoU threshold for the trigger action positive or negative reward in the environment.
                - 'classifier': The CNN used to classify the image ROI in the environment.
                - 'classifier_target_size': The size of the image that will be used as input to the classifier.
                
            Output:
                - State and information of the environment
        """
        # Resetting the environment
        super().reset(seed=seed)

        # Extracting the next image from the dataset if the dataset is not None
        if self.use_dataset is not None:
            self.extract()
        else:
            # Initialising image, the original image which will be used as a visualisation, the target bounding box, the height and the width of the image.
            if 'image' in env_config:
                self.image = env_config['image']
                self.height = self.image.shape[0]
                self.width = self.image.shape[1]
                del env_config['image']
            else:
                # If the image is not in the environment configuration, then take the original image as the image was changed during the process.
                self.image = self.original_image.copy()

            if 'original_image' in env_config:
                self.original_image = env_config['original_image']
                del env_config['original_image']

            if 'target_bbox' in env_config:
                self.target_bbox = env_config['target_bbox']
                del env_config['target_bbox']

            # Extracting current ground truth bounding boxes (multiple objects).
            if 'target_gt_boxes' in env_config:
                self.current_gt_bboxes = env_config['target_gt_boxes']
                self.target_bbox = self.current_gt_bboxes[self.current_gt_index]
                del env_config['target_gt_boxes']

        # Resetting the current ground truth index to 0
        self.current_gt_index = 0

        # Resetting the number of triggers to 0
        self.no_of_triggers = 0

        # Initialising the aspect ratio of the bounding box
        if 'alpha' in env_config:
            self.alpha = env_config['alpha']
            del env_config['alpha']
        else:
            self.alpha = ALPHA

        # Reward of Trigger
        if 'nu' in env_config:
            self.nu = env_config['nu']
            del env_config['nu']
        else:
            self.nu = NU

        # Resetting the step count, the cumulative reward, and truncated and terminated to False.
        self.step_count = 0
        self.cumulative_reward = 0
        self.truncated = False
        self.terminated = False

        # Initialising the threshold.
        if 'threshold' in env_config:
            self.threshold = env_config['threshold']
            del env_config['threshold']
        else:
            self.threshold = THRESHOLD

        # Initialising the actions history and the number of episodes.
        self.actions_history = []
        self.actions_history += ACTION_HISTORY

        # Initialising the bounding box of the image.
        self.bbox = [0, 0, self.width, self.height]

        # If use_sara is True, then invoke the generate_initial_bbox function
        if self.use_sara:
            self.bbox = self.generate_initial_bbox()

        # Classification part (Resetting the classification dictionary).
        self.classification_dictionary = {'label': [], 'confidence': [], 'bbox': [], 'color': []}

        # For Classification
        # Initialising the classifier.
        if 'classifier' in env_config:
            self.classifier = env_config['classifier']
            del env_config['classifier']
        else:
            self.classifier = CLASSIFIER

        # Initialising the classifier target size.
        if 'classifier_target_size' in env_config:
            self.classifier_target_size = env_config['classifier_target_size']
            del env_config['classifier_target_size']
        else:
            self.classifier_target_size = CLASSIFIER_TARGET_SIZE

        # Rendering part (resetting the window size, the window and the clock for rendering).
        if self.is_render:
            self.window_size = (self.width, self.height) #WINDOW_SIZE
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()

        # Resetting the current action to None.
        self.current_action = None

        # Rendering part (Retrieving a random color for the bounding box).
        self.color = self.generate_random_color()

        # Segmentation part (Resetting the segmentation dictionary).
        self.segmentation_dictionary = {'bboxes': [], 'masks': [], 'names': [], 'labels': [], 'colors': []}

        # Returning the observation space and the information of the environment.
        return self.get_state(), self.get_info()
    
    def get_labels(self):
        """
            Function that returns the labels of the images.

            Output:
                - Labels of the images
        """
        # Initialising an empty list to store the labels.
        images = []

        # Iterating through the bounding boxes
        for bbox in self.classification_dictionary['bbox']:
            # Retrieving the bounding box coordinates.
            x1, y1, x2, y2 = bbox

            # Cropping the image.
            image = self.original_image[y1:y2, x1:x2]

            # Resizing the image to the target size using OpenCV
            image = cv2.resize(image, self.classifier_target_size)

            # Preparing the image for the Classifier
            image = preprocess_input(image)

            # Expanding the dimensions to match the model's expectations
            image = np.expand_dims(image, axis=0)

            # Adding the image to the list of images
            images.append(image)

        # Ensuring images is not empty
        if images != []:
            # Converting the list of images to a numpy array
            images = np.concatenate(images, axis=0)

            # Predicting the classes
            preds = self.classifier.predict(images, verbose=0)

            # Decoding the predictions
            labels = decode_predictions(preds, top=1)

            # Iterating through the predictions and the bounding boxes
            for label, bbox in zip(labels, self.classification_dictionary['bbox']):
                # Retrieving the Label and the confidence of the image.
                label = label[0]

                # Storing the label, the confidence and the bounding box of the image in the classification dictionary.
                self.classification_dictionary['label'].append(label[1])
                self.classification_dictionary['confidence'].append(label[2])
                self.classification_dictionary['color'].append(self.generate_random_color())
        pass
    
    def predict(self, do_display=True, do_save=False, save_path=None):
        """
            Function that predicts the label of the image.

            Args:
                - do_display: Whether to display the image or not
                - do_save: Whether to save the image or not
                - save_path: Path to save the image

            Output:
                - Image
        """
        # Displaying the image.
        image = self.display(mode='detection', do_display=do_display, text_display=True)
        
        # Saving the image.
        if do_save:
            # If the save path is not provided, then create the path.
            if save_path is None:
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, 'image' + str(self.step_count) + str(self.current_gt_index) + '.png')
            # Saving the image
            cv2.imwrite(save_path, image)

        # Returning the image.
        return image

    def restart_and_change_state(self):
        """
            Function that restarts the environment and changes the state.
        """
        if self.env_mode == TRAIN_MODE: # Training mode
            # Incrementing the current ground truth index
            self.current_gt_index += 1

            # Checking if the current ground truth index is equal to the length of the current ground truth bounding boxes
            if self.current_gt_index == len(self.current_gt_bboxes): 
                # Resetting the current ground truth index to 0
                self.current_gt_index = 0

            # Retrieving the current ground truth bounding box
            self.target_bbox = self.current_gt_bboxes[self.current_gt_index]

        # Drawing an IoR (Inhibition of Return) cross on the image based on the current bounding box
        self.image = self.draw_ior_cross(self.image.copy(), self.bbox)

        # Adding the current bounding box to the classification dictionary
        self.classification_dictionary['bbox'].append(self.bbox)

        # Resetting bounding box to start from either of the corners and instead of having the whole image size, it will have a 75% of the image size
        if self.env_mode == TEST_MODE: # Testing mode
            if self.no_of_triggers % 4 == 0:
                self.bbox = [0, 0, int(self.width*0.75), int(self.height*0.75)]
            elif self.no_of_triggers % 4 == 1:
                self.bbox = [int(self.width*0.25), 0, self.width, int(self.height*0.75)]
            elif self.no_of_triggers % 4 == 2:
                self.bbox = [0, int(self.height*0.25), int(self.width*0.75), self.height]
            elif self.no_of_triggers % 4 == 3:
                self.bbox = [int(self.width*0.25), int(self.height*0.25), self.width, self.height]
        else:
            """
            Since the environment is in training mode, and a number of ground truth bounding boxes are provided, we set the initial bounding box 
            closest to the current ground truth bounding box. This is done such that the model can learn to predict the bounding box closest to the
            ground truth bounding box.
            """
            bbox1 = [0, 0, int(self.width*0.75), int(self.height*0.75)]
            bbox2 = [int(self.width*0.25), 0, self.width, int(self.height*0.75)]
            bbox3 = [0, int(self.height*0.25), int(self.width*0.75), self.height]
            bbox4 = [int(self.width*0.25), int(self.height*0.25), self.width, self.height]
            start_boxes = [bbox1, bbox2, bbox3, bbox4]

            # Retrieving the best IoU from the four bounding boxes and setting the bounding box to the current initial bounding box
            ious = [iou(bbox1, self.target_bbox), iou(bbox2, self.target_bbox), iou(bbox3, self.target_bbox), iou(bbox4, self.target_bbox)]

            # Finding the argmax of the IoUs
            best_iou_index = np.argmax(ious)

            # Setting the bounding box to the best bounding box
            self.bbox = start_boxes[best_iou_index]

        # Incrementing the number of triggers
        self.no_of_triggers += 1
        pass
        
    def draw_ior_cross(self, image, bbox, color=(0, 0, 0)):
        """
            Function that draws an IoR (Inhibition of Return) cross on the image based on the current bounding box.

            Args:
                - Image
                - Bounding box
                - Color
                - Thickness

            Output:
                - Image with the IoR cross
        """
        # Retrieving the coordinates of the bounding box.
        x1, y1, x2, y2 = bbox

        # Calculating box width and height
        width = x2 - x1
        height = y2 - y1

        # Calculating the thickness of the IoR cross.
        thickness = max(5, int(min(width, height) / 20))

        # Calculating the center of the bounding box.
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Drawing the horizontal line of the IoR cross.
        cv2.line(image, (center[0], y1), (center[0], y2), color, thickness)

        # Drawing the vertical line of the IoR cross.
        cv2.line(image, (x1, center[1]), (x2, center[1]), color, thickness)

        # Returning the image with the IoR cross.
        return image
    
    def step(self, action):
        """
            Function that performs an action on the environment.

            Input:
                - Action to perform

            Output:
                - State of the environment
                - Reward of the action
                - Whether the episode is finished or not
                - Information of the environment
        """
        # Updating the history of the actions.
        self.update_history(action)

        # Declaring the reward.
        reward = 0

        # Updating the current action.
        self.current_action = action
        
        # Checking the action type and applying the action to the image (transform action).
        if action < 8 and (self.step_count % self.trigger_steps != 0 or self.step_count == 0):
        # Retrieving the previous state
            previous_state = [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]]

            # Applying the action to the image.
            self.bbox = self.transform_action(action)

            # Retrieving the current state.
            current_state = [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]]

            # Calculating the reward.
            reward = self.calculate_reward([current_state], [previous_state], self.current_gt_bboxes) #self.target_bbox)
        else:
            # Retrieving the current state.
            current_state = self.bbox

            # Calculating the reward.
            reward = self.calculate_trigger_reward([current_state], self.current_gt_bboxes) #self.target_bbox)

            self.restart_and_change_state()

            # In case of single object detection, we terminate the episode after the first trigger
            if self.obj_configuration == SINGLE_OBJ:
                self.terminated = True

        # Calculating the cumulative reward.
        self.cumulative_reward += reward

        # Incrementing the step count.
        self.step_count += 1

        # Checking if the episode is finished and truncated.
        if self.step_count >= self.max_steps:
            self.terminated = True
            self.truncated = False

            # If the action was not a trigger, then we set reset and change the state
            self.restart_and_change_state()

        # If the episode is finished, increment the number of episodes.
        if self.terminated or self.truncated:
            self.num_episodes += 1

            # For classification, retrieve the labels if the classification dictionary is empty
            if self.allow_classification and self.classification_dictionary['label']==[]:
                self.get_labels()
            self.filter_bboxes() # Saving to evaluation results

        # Rendering the environment, if it is set to be rendered.
        if self.is_render:
            self.render(self.render_mode)

        # Returning the state of the environment, the reward, whether the episode is finished or not, whether the episode is truncated or not and the information of the environment.
        return self.get_state(), reward, self.terminated, self.truncated, self.get_info()
    
    def decode_render_action(self, action):
        """
        Function that decodes the action.

        Input:
            - Action to decode

        Output:
            - Decoded action as a string
        """
        if action == 0:
            return "Move right"
        elif action == 1:
            return "Move left"
        elif action == 2:
            return "Move up"
        elif action == 3:
            return "Move down"
        elif action == 4:
            return "Make bigger"
        elif action == 5:
            return "Make smaller"
        elif action == 6:
            return "Make fatter"
        elif action == 7:
            return "Make taller"
        elif action == 8:
            return "Trigger"
        else:
            return "N/A"
        pass

    def _render_frame(self, mode='human', close=False, alpha=0.2, text_display=True):
        """
            Function that renders the environment.

            Args:
                - Mode: Mode of rendering (human, trigger_image, bbox, rgb_array)
                - Close: Whether to close the environment or not
                - Alpha: Alpha value for blending the image with the rectangle
                - Text_display: Whether to display the text or not

            Output:
                - Image
        """
        # Retrieving bounding box coordinates.
        x1, y1, x2, y2 = self.bbox  # Make sure self.bbox is defined in your environment

        # Creating a Pygame Surface to render on
        canvas = pygame.Surface((self.window_size[0], self.window_size[1]))

        # Checking the mode of rendering.
        if mode == 'human' or mode == 'trigger_image' or mode == 'bbox':
            # Setting the background color of the canvas depending on the mode
            img = self.original_image.copy()

            if mode == 'trigger_image':
                img = self.image.copy()

            # Creating target bounding box
            if mode == 'bbox':
                img = np.zeros_like(self.original_image)
                alpha = 0.5

            # Creating target bounding box
            if self.env_mode == TRAIN_MODE: # Training mode
                # Creating a different color for the target bounding box from the current bounding box
                target_color = (0, 255, 0) if self.color != (0, 255, 0) else (255, 0, 0)

                # Creating a copy of the image
                image_copy = img.copy()

                # Creating a filled rectangle for the target bounding box
                cv2.rectangle(image_copy, (self.target_bbox[0], self.target_bbox[1]), (self.target_bbox[2], self.target_bbox[3]), target_color, cv2.FILLED)

                # Blending the image with the rectangle using cv2.addWeighted
                img = cv2.addWeighted(img, 1 - alpha, image_copy, alpha, 0)

                # Adding a rectangle outline to the image
                cv2.rectangle(img, (self.target_bbox[0], self.target_bbox[1]), (self.target_bbox[2], self.target_bbox[3]), target_color, 3)

            # Creating a copy of the image
            image_copy = img.copy()

            if not (self.truncated or self.terminated):
                # Creating a filled rectangle for the current bounding box
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), self.color, cv2.FILLED)

                # Blending the image with the rectangle using cv2.addWeighted
                img = cv2.addWeighted(img, 1 - alpha, image_copy, alpha, 0)

                # Adding a rectangle outline to the image
                cv2.rectangle(img, (x1, y1), (x2, y2), self.color, 3)

            # If the mode is not trigger_image, then iterate through the classification dictionary and draw the bounding boxes
            if mode != 'trigger_image':
                # Iterating through the classification dictionary (for bounding boxes)
                for label_idx in range(len(self.classification_dictionary['bbox'])):
                    # Retrieving the label and the confidence and the bounding box
                    if self.allow_classification and self.classification_dictionary['label']:
                        label = self.classification_dictionary['label'][label_idx]
                        label_confidence = self.classification_dictionary['confidence'][label_idx]
                    predicted_bbox = self.classification_dictionary['bbox'][label_idx]
                    # Extracting coordinates
                    x1, y1, x2, y2 = predicted_bbox

                    # Drawing the bounding box on the image (Creating a filled rectangle for the bounding box)
                    cv2.rectangle(img, (predicted_bbox[0], predicted_bbox[1]), (predicted_bbox[2], predicted_bbox[3]), self.color, cv2.FILLED)

                    # Blending the image with the rectangle using cv2.addWeighted
                    img = cv2.addWeighted(img, 1 - alpha, image_copy, alpha, 0)

                    # Adding a rectangle outline to the image
                    cv2.rectangle(img, (predicted_bbox[0], predicted_bbox[1]), (predicted_bbox[2], predicted_bbox[3]), self.color, 3)

            # Adding the label to the image
            if text_display and self.classification_dictionary['label'] and (self.truncated or self.terminated) and self.allow_classification:
                # Setting the font and the font scale
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2

                # Iterating through the classification dictionary (for text)
                for label_idx in range(len(self.classification_dictionary['label'])):
                    # Retrieving the label and the confidence and the bounding box
                    label = self.classification_dictionary['label'][label_idx]
                    label_confidence = self.classification_dictionary['confidence'][label_idx]
                    predicted_bbox = self.classification_dictionary['bbox'][label_idx]
                    # Extracting coordinates
                    x1, y1, x2, y2 = predicted_bbox

                    # Creating the label text
                    text = str(label.capitalize()) + '  ' + str(round(label_confidence, 2))

                    # Drawing the label on the image, whilst ensuring that it doesn't go out of bounds
                    (label_width, label_height), baseline = cv2.getTextSize(text, font, font_scale, 2)

                    # Ensuring that the label doesn't go out of bounds
                    if y1 - label_height - baseline < 0:
                        y1 = label_height + baseline
                    if x1 + label_width > img.shape[1]:
                        x1 = img.shape[1] - label_width
                    if y1 + label_height + baseline > img.shape[0]:
                        y1 = img.shape[0] - label_height - baseline
                    if x1 < 0:
                        x1 = 0

                    # Creating a filled rectangle for the label background
                    cv2.rectangle(img, (x1, y1 - label_height - baseline), (x1 + label_width, y1), self.color, -1)

                    # Adding the label text to the image
                    cv2.putText(img, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

            # Converting the image to a Pygame surface
            image_surface = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))

            # Displaying Action on image surface at the top left corner
            font = pygame.font.SysFont('Lato', 50)#, bold=True)

            # Adding Action on the image surface at the top left corner
            text = font.render('Action: ' + str(self.decode_render_action(self.current_action)), True, (255, 255, 255))
            image_surface.blit(text, (0, 0))

            # Adding Step | Reward | IoU on the image surface at the bottom left corner
            if self.env_mode == TRAIN_MODE: # Training mode
                marker_ratio = 0.05  # 5% of the window size
                label_ratio_x = 0.25  # 25% from the right edge of the window
                label_ratio_y = 0.05  # 5% from the bottom edge of the window

                # Add Step | Reward | IoU  on the image surface at the bottom left corner
                font_ratio = 0.04  # 5% of the image height
                font_size = int(font_ratio * self.height)  # Calculate the font size
                font = pygame.font.SysFont('Lato', font_size)#, bold=True) (font_size was 20 before)

                text = font.render('Step: ' + str(self.step_count) + ' | Reward: ' + str(round(self.cumulative_reward, 3)) + ' | IoU: ' + str(round(calculate_best_iou([self.bbox], self.current_gt_bboxes), 3)) + ' | Recall: ' + str(round(calculate_best_recall([self.bbox], self.current_gt_bboxes), 3)), True, (255, 255, 255))
                image_surface.blit(text, (0, self.window_size[1] - font_size))

                # Create font with Lato, size 30
                font_ratio = 0.07  # 5% of the image height
                font_size = int(font_ratio * self.height)  # Calculate the font size
                font = pygame.font.SysFont('Lato', font_size)#, bold=True) (font_size was 30 before)

                # Adding bottom right legend for bounding box colors
                window_width, window_height = self.window_size  # Get the size of the window
                target_marker_size = int(marker_ratio * min(window_width, window_height))  # Calculate the marker size
                prediction_marker_size = target_marker_size  # Same size for prediction marker

                # Calculating the positions of the markers and labels
                label_text = font.render('Ground Truth', True, target_color)
                text_width, _ = label_text.get_size()
                target_marker_position = (self.window_size[0] - text_width - target_marker_size, self.window_size[1] - target_marker_size)
                target_label_position = (target_marker_position[0] + target_marker_size + 2, target_marker_position[1])

                label_text = font.render('Prediction', True, self.color)
                text_width, _ = label_text.get_size()
                prediction_marker_position = (self.window_size[0] - text_width - prediction_marker_size, self.window_size[1] - 2 * target_marker_size)
                prediction_label_position = (prediction_marker_position[0] + prediction_marker_size + 2, prediction_marker_position[1])

                # Drawing the markers and labels
                # Marker for Ground Truth (using a rectangular marker)
                pygame.draw.rect(image_surface, target_color, (*target_marker_position, target_marker_size, target_marker_size))
                label_text = font.render('Ground Truth', True, target_color)
                image_surface.blit(label_text, target_label_position)

                # Marker for Prediction (using a circular marker)
                pygame.draw.circle(image_surface, self.color, (prediction_marker_position[0] + prediction_marker_size//2, prediction_marker_position[1] + prediction_marker_size//2), prediction_marker_size//2)
                label_text = font.render('Prediction', True, self.color)
                image_surface.blit(label_text, prediction_label_position)

            # Drawing the original image on the canvas
            canvas.blit(image_surface, (0, 0))

            # Displaying the frame
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()

            # Processing Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

            self.clock.tick(10)  # Adjust the frame rate as needed

            # Returning the image surface as a NumPy array
            return np.swapaxes(pygame.surfarray.array3d(image_surface), 0, 1)
        
        elif mode == 'rgb_array':
            # Creating an RGB array for Gym rendering
            rgb_array = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)

            # Drawing bounding box on the RGB array
            cv2.rectangle(rgb_array, (x1, y1), (x2, y2), self.color, 3)

            # Adding Action on the RGB array at the top left corner
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rgb_array, 'Action: ' + str(self.decode_render_action(self.current_action)),
                        (10, 40), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Adding Step | Reward | IoU on the RGB array at the bottom left corner
            cv2.putText(rgb_array, 'Step: ' + str(self.step_count) + ' | Reward: ' + str(round(self.cumulative_reward, 3)) +
                        ' | IoU: ' + str(round(calculate_best_iou([self.bbox], self.current_gt_bboxes), 3)) + ' | Recall: ' + str(round(calculate_best_recall([self.bbox], self.current_gt_bboxes), 3)),
                        (10, self.window_size[1] - 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Returning the RGB array
            return rgb_array

        # Returning a numpy array of the image canvas
        return np.array(pygame.surfarray.array3d(canvas))

    def render(self, mode=RENDER_MODE, close=False):
        """
            Function that renders the environment.

            Args:
                - Mode: Mode of rendering (human, trigger_image, bbox, rgb_array)
                - Close: Whether to close the environment or not

            Output:
                - Image
        """
        mode = self.render_mode
        return self._render_frame(mode, close)
    
    def display(self, mode='image', do_display=False, text_display=True, alpha=0.3, color=(0, 255, 0)):
        """
            Function that renders the environment.

            Args:
                - Mode: Mode of rendering (image, trigger_image, bbox, detection, heatmap, None)
                - Do_display: Whether to display the image or not
                - Text_display: Whether to display the text or not
                - Alpha: Alpha value for blending the image with the rectangle
                - Color: Color of the bounding box

            Output:
                - Image of the environment
        """
        # Retrieving bounding box coordinates.
        x1, y1, x2, y2 = self.bbox

        # Checking the mode of rendering.
        if mode == 'image' or mode == 'trigger_image' or mode == 'detection':
            # Creating a copy of the original image.
            image_copy = self.original_image.copy()

            # Checking the mode of rendering.
            if mode == 'trigger_image':
                image_copy = self.image.copy()

            if mode == 'detection':
                alpha = 0.5

            # Creating a filled rectangle for the bounding box
            if not (self.truncated or self.terminated):
                # Creating a filled rectangle for the bounding box
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), self.color, cv2.FILLED)

                # Blending the image with the rectangle using cv2.addWeighted
                image = cv2.addWeighted(self.image.copy(), 1 - alpha, image_copy, alpha, 0)

                # Adding a rectangle outline to the image
                cv2.rectangle(image, (x1, y1), (x2, y2), self.color, 3)
            else:
                image = image_copy

            if mode != 'trigger_image':
                # Iterating through the classification dictionary (for bounding boxes)
                for label_idx in range(len(self.classification_dictionary['label'])):
                    # Retrieving the label and the confidence and the bounding box
                    label = self.classification_dictionary['label'][label_idx]
                    label_confidence = self.classification_dictionary['confidence'][label_idx]
                    predicted_bbox = self.classification_dictionary['bbox'][label_idx]
                    predicted_bbox_color = self.classification_dictionary['color'][label_idx]

                    # Extracting coordinates
                    x1, y1, x2, y2 = predicted_bbox
                    image_copy = image.copy()

                    # Drawing the bounding box on the image (Creating a filled rectangle for the bounding box)
                    cv2.rectangle(image, (predicted_bbox[0], predicted_bbox[1]), (predicted_bbox[2], predicted_bbox[3]), predicted_bbox_color, cv2.FILLED)

                    # Blending the image with the rectangle using cv2.addWeighted
                    image = cv2.addWeighted(image, 1 - alpha, image_copy, alpha, 0)

                    # Adding a rectangle outline to the image
                    cv2.rectangle(image, (predicted_bbox[0], predicted_bbox[1]), (predicted_bbox[2], predicted_bbox[3]), predicted_bbox_color, 3)
    
            # Adding the label to the image
            if text_display and self.classification_dictionary['label'] and mode == 'detection':
                # Setting the font and the font scale
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2

                # Iterating through the classification dictionary (for text)
                for label_idx in range(len(self.classification_dictionary['label'])):
                    # Retrieving the label and the confidence and the bounding box
                    label = self.classification_dictionary['label'][label_idx]
                    label_confidence = self.classification_dictionary['confidence'][label_idx]
                    predicted_bbox = self.classification_dictionary['bbox'][label_idx]
                    predicted_bbox_color = self.classification_dictionary['color'][label_idx]

                    # Extracting coordinates
                    x1, y1, x2, y2 = predicted_bbox

                    # Creating the label text
                    text = str(label.capitalize()) + '  ' + str(round(label_confidence, 2))

                    # Drawing the label on the image, whilst ensuring that it doesn't go out of bounds
                    (label_width, label_height), baseline = cv2.getTextSize(text, font, font_scale, 2)

                    # Ensuring that the label doesn't go out of bounds
                    if y1 - label_height - baseline < 0:
                        y1 = label_height + baseline
                    if x1 + label_width > image.shape[1]:
                        x1 = image.shape[1] - label_width
                    if y1 + label_height + baseline > image.shape[0]:
                        y1 = image.shape[0] - label_height - baseline
                    if x1 < 0:
                        x1 = 0

                    # Creating a filled rectangle for the label background
                    cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), predicted_bbox_color, -1)

                    # Adding the label text to the image
                    cv2.putText(image, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

            # Plotting the image.
            if do_display and mode != 'detection':
                self.plot_img(image, title='Step: ' + str(self.step_count) + ' | Reward: ' + str(round(self.cumulative_reward, 3)) + ' | IoU: ' + str(round(calculate_best_iou([self.bbox], self.current_gt_bboxes), 3)) + ' | Recall: ' + str(round(calculate_best_recall([self.bbox], self.current_gt_bboxes), 3)))
            elif do_display:
                self.plot_img(image, title='Object Detection',figure_size=(10, 7))
            # Returning the image.
            return image
        
        # If mode is bbox, then return the image with the bounding box
        elif mode == 'bbox':
            # Creating a black image from the original image.
            image = np.zeros_like(self.original_image)

            # Drawing the bounding box on the image.
            cv2.rectangle(image, (x1, y1), (x2, y2), color, cv2.FILLED)

            # Plotting the image.
            if do_display:
                self.plot_img(image, title='Step: ' + str(self.step_count) + ' | Reward: ' + str(round(self.cumulative_reward, 3)) + ' | IoU: ' + str(round(calculate_best_iou([self.bbox], self.current_gt_bboxes), 3)) + ' | Recall: ' + str(round(calculate_best_recall([self.bbox], self.current_gt_bboxes), 3)))

            # Returning the image.
            return image
        
        # If mode is heatmap, then return the heatmap of the bounding box
        elif mode == 'heatmap':
            # Creating a black image.
            image = np.zeros_like(self.original_image)

            # Drawing the bounding box on the image.
            cv2.rectangle(image, (x1, y1), (x2, y2), color, cv2.FILLED)

            # Creating the heatmap.
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)

            # Plotting the image.
            if do_display:
                self.plot_img(heatmap, title='Step: ' + str(self.step_count) + ' | Reward: ' + str(round(self.cumulative_reward, 3)) + ' | IoU: ' + str(round(calculate_best_iou([self.bbox], self.current_gt_bboxes), 3)) + ' | Recall: ' + str(round(calculate_best_recall([self.bbox], self.current_gt_bboxes), 3)))

            # Returning the image.
            return heatmap
        
    def segment(self, display_mode="mask", do_display=False, do_save=False, save_path=None, text_display=True, alpha=0.7, color=(0, 255, 0)):
        """
            Function that segments the object in the bounding box.

            Note: This function is used for segmentation tasks and is incomplete and not precise as it is a work in progress.

            Args:
                - Display_mode: Mode of display (mask, contour, None)
                - Do_display: Whether to display the image or not
                - Do_save: Whether to save the image or not
                - Save_path: Path to save the image
                - Text_display: Whether to display the text or not
                - Alpha: Alpha value for blending the image with the rectangle
                - Color: Color of the bounding box

            Output:
                - Segmentation dictionary

        """
        # Resetting the segmentation dictionary
        self.segmentation_dictionary = {'names': [], 'masks': [], 'bboxes': [], 'labels': [], 'colors': []}

        # Iterating through the classification dictionary
        for label_idx in range(len(self.classification_dictionary['label'])):
            # Extracting the information from the classification dictionary
            label = self.classification_dictionary['label'][label_idx]
            label_confidence = self.classification_dictionary['confidence'][label_idx]
            predicted_bbox = self.classification_dictionary['bbox'][label_idx]
            predicted_bbox_color = self.classification_dictionary['color'][label_idx]

            # Retrieving bounding box coordinates.
            x1, y1, x2, y2 = predicted_bbox

            offset = 0
            # Going a bit outside the bounding box
            x1, y1, x2, y2 = x1 - offset, y1 - offset, x2 + offset, y2 + offset

            # Creating a black 3-channel mask image.
            mask = np.zeros_like(self.original_image)

            # Changing to 1-channel mask image.
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            # Extracting the object from the mask.
            bbox_object = self.original_image[y1:y2, x1:x2]
            
            # From the bbox_object, we extract the mask via Canny Edge Detection.
            edges = cv2.Canny(bbox_object, 100, 200)
            mask_copy = mask.copy()

            # Mapping the edges to the mask.
            mask_copy[y1:y2, x1:x2] = edges

            # Retrieving the contours of the mask.
            contours = ah.single_object_polygon_approximation(mask_copy, epsilon=0.005, do_cvt=False)

            # Drawing the single polygon onto the mask and fill it
            cv2.fillPoly(mask, pts=contours, color=(255, 255, 255))
    
            # Defining a larger structuring element
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))

            # Applying the closing operation multiple times
            for _ in range(100):  # Change this number to apply the operation more or less times
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Filling the remaining holes in the mask with the closing operation but different kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
            for _ in range(4):
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Creating a filled polygon annotation for the object mask on a copy of the original image.
            image_copy = self.original_image.copy()       

            # SaRa algorithm
            sara.reset()
            # Calculating Itti Saliency Map
            saliency_map_itti = sara.return_saliency(image_copy.copy(), generator=generator)
            saliency_map_rgb_itti = cv2.cvtColor(saliency_map_itti, cv2.COLOR_BGR2RGB)
            saliency_map_gray_itti = cv2.cvtColor(saliency_map_rgb_itti, cv2.COLOR_RGB2GRAY)

            # Thresholding the saliency map using Otsu's method
            ret, thresh_itti = cv2.threshold(saliency_map_gray_itti, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Performing morphological operations on the thresholded image
            kernel = np.ones((5,5), np.uint8)
            dilated = cv2.dilate(thresh_itti, kernel, iterations=2)
            erosion = cv2.erode(dilated, kernel, iterations=2)

            # Applying the saliency map to the mask, to remove holes in the mask
            mask = cv2.bitwise_and(erosion, mask)#, mask=erosion)

            # Apply Gaussian blur to smooth the mask
            mask = cv2.GaussianBlur(mask, (3, 3), 0)

            # Appending name (label with percentage confidence), mask and bounding box to the segmentation dictionary
            self.segmentation_dictionary['names'].append(label+" "+str(round(label_confidence*100, 2))+ "%")
            self.segmentation_dictionary['masks'].append(mask)
            self.segmentation_dictionary['bboxes'].append(predicted_bbox)
            self.segmentation_dictionary['labels'].append(label)
            self.segmentation_dictionary['colors'].append(predicted_bbox_color)

        # Plotting the image.
        if do_display:
            if display_mode == "mask":
                # Retrieving the number of objects
                num_objects = len(self.segmentation_dictionary['masks'])

                max_cols = 4

                # Calculating the rows and columns via the number of objects and modulus on max_cols
                rows = int(num_objects / max_cols) + 1 if num_objects % max_cols != 0 else int(num_objects / max_cols)
                cols = max_cols if num_objects > max_cols else num_objects

                # Creating dictionary for the masks
                num_objects = len(self.segmentation_dictionary['names'])

                masks = {}
                for i in range(num_objects):
                    # Ensuring that the name is unique
                    key = "Mask " + str(i + 1) + " - " + self.segmentation_dictionary['names'][i]
                    masks[key] = self.segmentation_dictionary['masks'][i]

                # Plotting the masks
                self.plot_multiple_imgs(masks, rows, cols, suptitle='Instance Segmentation Masks')

            elif display_mode == "image":
                # Creating a filled polygon annotation for the object mask on a copy of the original image.
                image_copy = self.original_image.copy()

                # Iterating through the segmentation dictionary
                for i in range(len(self.segmentation_dictionary['masks'])):
                    # Extracting the mask and the bounding box
                    mask = self.segmentation_dictionary['masks'][i]
                    bbox = self.segmentation_dictionary['bboxes'][i]
                    name = self.segmentation_dictionary['names'][i]
                    current_color = self.segmentation_dictionary['colors'][i]

                    x1, y1, x2, y2 = bbox

                    # Calculating contours of the mask
                    contours_mask, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                    # Making a copy of the original image
                    original_image = image_copy.copy()

                    # Fill a polygon on the image copy
                    cv2.fillPoly(image_copy, pts=contours_mask, color=current_color)

                    # Blend the original image with the image copy
                    image_copy = cv2.addWeighted(original_image, 1 - alpha, image_copy, alpha, 0)

                    # Draw a rectangle on the blended image
                    cv2.rectangle(image_copy, (x1, y1), (x2, y2), current_color, 3)

                    if text_display:
                        # Setting the font and the font scale
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.2
                        
                        text = name

                        # Drawing the label on the image, whilst ensuring that it doesn't go out of bounds
                        (label_width, label_height), baseline = cv2.getTextSize(text, font, font_scale, 2)

                        # Ensuring that the label doesn't go out of bounds
                        if y1 - label_height - baseline < 0:
                            y1 = label_height + baseline
                        if x1 + label_width > image_copy.shape[1]:
                            x1 = image_copy.shape[1] - label_width
                        if y1 + label_height + baseline > image_copy.shape[0]:
                            y1 = image_copy.shape[0] - label_height - baseline
                        if x1 < 0:
                            x1 = 0

                        # Creating a filled rectangle for the label background
                        cv2.rectangle(image_copy, (x1, y1 - label_height - baseline), (x1 + label_width, y1), current_color, -1)

                        # Adding the label text to the image
                        cv2.putText(image_copy, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

                self.plot_img(image_copy, title='Instance Segmentation', figure_size=(10, 7))

                # Saving the image
                if do_save:
                    # If the save path is None, then create a new directory
                    if save_path is None:
                        os.makedirs(save_path, exist_ok=True)
                        save_path = os.path.join(save_path, "Segmentation" + str(self.step_count) + str(self.num_episodes) + ".png")

                    # Saving the image
                    cv2.imwrite(save_path, image_copy)

        # Returning the segmentation dictionary
        return self.segmentation_dictionary
    
    def annotate(self, image, id, title, project_name, save_dir, category="No category", annotation_format="coco", do_display=False, do_save=False, do_print=True, annotation_color=(255, 0, 255), epsilon=0.005, configuration=coco.POLY_APPROX, object_configuration=coco.SINGLE_OBJ, do_cvt=True):
        """
            Function which utilise the Mask to Annotation software to annotate an object mask in an image.

            Note: This function is a wrapper for the Mask to Annotation software (IEEE ISM 2023) (https://github.com/dylanseychell/mask-to-annotation)

            Args:
                - Image: Image to annotate
                - Id: Id of the image
                - Title: Title of the image
                - Project_name: Name of the project
                - Save_dir: Directory to save the annotation
                - Category: Category of the object
                - Annotation_format: Format of the annotation (coco, vgg, yolo)
                - Do_display: Whether to display the image or not
                - Do_save: Whether to save the image or not
                - Do_print: Whether to print the annotation or not
                - Annotation_color: Color of the annotation
                - Epsilon: Epsilon value for polygon approximation
                - Configuration: Configuration for the annotation
                - Object_configuration: Object configuration for the annotation
                - Do_cvt: Whether to convert the annotation or not
        """
        # Checking the annotation format.
        if annotation_format == "coco":
            coco.annotate((id, title, image, project_name, category, save_dir), do_display=do_display, do_save=do_save, do_print=do_print, annotation_color=annotation_color, epsilon=epsilon, configuration=configuration, object_configuration=object_configuration, do_cvt=do_cvt)
        elif annotation_format == "vgg":
            vgg.annotate((id, title, image, project_name, category, save_dir), do_display=do_display, do_save=do_save, do_print=do_print, annotation_color=annotation_color, epsilon=epsilon, configuration=configuration, object_configuration=object_configuration, do_cvt=do_cvt)
        elif annotation_format == "yolo":
            yolo.annotate((id, title, image, project_name, category, save_dir), do_display=do_display, do_save=do_save, do_print=do_print, annotation_color=annotation_color, object_configuration=object_configuration, do_cvt=do_cvt)
        else:
            raise Exception("Unknown Annotation Format.")
        
    def plot_img(self, image, title=None, figure_size=(10,7)):
        """
            Function that plots the image.

            Args:
                - Image: Image to plot
                - Title: Title of the image
                - Figure_size: Size of the figure
        """
        # Plotting the image.
        plt.figure(figsize=figure_size)
        plt.imshow(image, cmap='Blues_r')
        plt.axis('off')
        if title is not None:
            plt.title(title, fontsize=20)
        plt.show()
       
    def plot_multiple_imgs(self, images, rows=1, cols=1, figure_size=(20, 7)):
        """
            Function that plots multiple images.

            Args:
                - Images: Images to plot
                - Rows: Number of rows
                - Cols: Number of columns
                - Figure_size: Size of the figure

        """
        # Creating the subplot grid
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figure_size)

        # Flattening the axes in case rows or cols > 1
        axes = np.ravel(axes)

        # Iterating through images
        for i, (imageName, image) in enumerate(images.items()):
            # Plotting the image
            axes[i].imshow(image, interpolation='nearest', cmap='Blues_r')
            axes[i].set_title(imageName, fontsize=18)
            axes[i].axis('off')

        # Deleting the extra axes which are not used
        if len(images) < len(axes):
            for j in range(len(images), len(axes)):
                fig.delaxes(axes[j])

        # Adjusting layout
        fig.tight_layout()

        # Showing the plot
        plt.show()
        
    def close(self):
        """
            Function that closes the environment.
        """
        # Close the Pygame window
        if self.is_render:
            pygame.quit()

        # Close the environment
        Env.close(self)
        pass
    
    def load_pascal_voc_dataset(self, path='data', year='2007', download=True, image_set='train'):
        """
            Function that loads the Pascal VOC dataset.

            Args:
                - Path: Path to the dataset
                - Year: Year of the dataset (2007, 2012)
                - Download: Whether to download the dataset or not
                - Image_set: Image set of the dataset (train, val, test)

            Output:
                - Dataset
        """
        # Loading the Pascal VOC dataset
        dataset = datasets.VOCDetection(path, year, image_set, download)

        # Shuffling the dataset
        indices = torch.randperm(len(dataset))
        dataset = torch.utils.data.Subset(dataset, indices)

        # Pascal VOC classes
        self.classes = ['cat', 'bird', 'motorbike', 'diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant', 'person', 'chair', 'boat', 'bottle', 'bicycle', 'dog', 'aeroplane', 'cow', 'sheep', 'sofa']

        # Sorting the dataset by class
        dataset = self.sort_pascal_voc_by_class(dataset)
        
        # Calculating the total number of images in the dataset
        self.total_images = 0

        # Iterating through the classes to calculate the total number of images
        for c_class in self.classes:
            self.total_images += len(dataset[c_class])
        
        # Printing the total number of classes and images in the dataset
        print('\033[92m' + 'Dataset loaded successfully.' + '\033[0m')
        print('\033[93m' + 'Total number of classes in the dataset:', len(self.classes))
        print('\033[94m' + 'Total number of images in the dataset:', self.total_images)

        # Returning the dataset
        return dataset
    
    def load_training_dataset(self, path='data', download=True, image_set='train'):
        """
            Function that loads the Pascal VOC 2007 + 2012 dataset.

            Args:
                - Path: Path to the dataset
                - Download: Whether to download the dataset or not
                - Image_set: Image set of the dataset (train, val, test)

            Output:
                - Dataset
        """
        # Loading the Pascal VOC 2007 + 2012 dataset
        dataset_2007 = datasets.VOCDetection(path, year='2007', image_set=image_set, download=download)
        dataset_2012 = datasets.VOCDetection(path, year='2012', image_set=image_set, download=download)

        # Concatenating the datasets
        dataset = torch.utils.data.ConcatDataset([dataset_2007, dataset_2012])

        # Shuffling the dataset
        indices = torch.randperm(len(dataset))
        dataset = torch.utils.data.Subset(dataset, indices)

        # Pascal VOC classes
        self.classes = ['cat', 'bird', 'motorbike', 'diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant', 'person', 'chair', 'boat', 'bottle', 'bicycle', 'dog', 'aeroplane', 'cow', 'sheep', 'sofa']

        # Sorting the dataset by class
        dataset = self.sort_pascal_voc_by_class(dataset)
        
        # Calculating the total number of images in the dataset
        self.total_images = 0

        # Iterating through the classes to calculate the total number of images
        for c_class in self.classes:
            self.total_images += len(dataset[c_class])
        
        # Printing the total number of classes and images in the dataset
        print('\033[92m' + 'Dataset loaded successfully.' + '\033[0m')
        print('\033[93m' + 'Total number of classes in the dataset:', len(self.classes))
        print('\033[94m' + 'Total number of images in the dataset:', self.total_images)

        # Returning the dataset
        return dataset
    
    def sort_pascal_voc_by_class(self, dataset):    
        """
            Function that sorts the Pascal VOC dataset by class, by iterating through the dataset and adding the images to the corresponding class.

            Input:
                - Datasets
            
            Output:
                - Dictionary of datasets (keys: classes, values: all the data of this class)
        """
        # Creating a dictionary of the dataset
        dataset_per_class = {}

        # Iterating through the classes
        for c_class in self.classes:
            dataset_per_class[c_class] = {}

        # Iterating through all the entries in the dataset
        for entry in dataset:
            # Extracting the image and the target
            img, target = entry

            # Extracting the class and the filename
            classe = target['annotation']['object'][0]["name"]
            filename = target['annotation']['filename']

            # Creating a dictionary of the dataset
            org = {}

            # Iterating through the classes
            for c_class in self.classes:

                # Adding the image to the class
                org[c_class] = []
                org[c_class].append(img)

            # Iterating through the objects to retrieve the object bounding box and size for every class
            for c_object in range(len(target['annotation']['object'])):
                classe = target['annotation']['object'][c_object]["name"]
                org[classe].append([target['annotation']['object'][c_object]["bndbox"], target['annotation']['size']])
            
            # Iterating through the classes
            for c_class in self.classes:
                # If the class has more than one image in the dataset, then we add the image to the class
                if len( org[c_class] ) > 1:
                    try:
                        dataset_per_class[c_class][filename].append(org[c_class])
                    except KeyError:
                        dataset_per_class[c_class][filename] = []
                        dataset_per_class[c_class][filename].append(org[c_class])

        # Returning the dataset per class
        return dataset_per_class
    
    def extract(self):
        """
            Function that extracts the current image, original image and target bounding box from the dataset.
        """        
        # Extracting image per class
        extracted_imgs_per_class = self.dataset[self.current_class]

        # If the class image index is greater than the length of the current class dataset, then we reset the class image index
        if self.class_image_index >= len(self.dataset[self.current_class]):
            self.class_image_index = 0

            # Incrementing the epoch count and printing the epoch count
            self.epochs += 1
            print("*"*100)
            print('\033[92m' + 'Epoch ' + str(self.epochs) + ' done for class ' + self.current_class + '.' + '\033[0m')
            print("*"*100)

        # Finding the key which corresponds to the current class image index
        img_name = list(extracted_imgs_per_class.keys())[self.class_image_index]

        # Extracting the image information
        img_information = extracted_imgs_per_class[img_name][0]

        # Extracting the image
        self.image = img_information[0]

        # Converting image to cv2 format and copying the original image
        self.image = np.array(self.image)
        self.original_image = self.image.copy()

        # Extracting the height and width of the image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        # Extracting the ground truth bounding boxes in dictionary format
        gt_bboxes_dict = img_information[1:]

        # Flattening the list of dictionaries
        gt_bboxes_dict = [item for sublist in gt_bboxes_dict for item in sublist]

        # Removing odd indices from the list which correspond to the width and height of the image
        gt_bboxes_dict = [gt_bboxes_dict[i] for i in range(len(gt_bboxes_dict)) if i % 2 == 0]

        # For each entry in the list, which is a dictionary, we change it in the form of [x1, y1, x2, y2]
        for i in range(len(gt_bboxes_dict)):
            gt_bboxes_dict[i] = [int(gt_bboxes_dict[i]['xmin']), int(gt_bboxes_dict[i]['ymin']), int(gt_bboxes_dict[i]['xmax']), int(gt_bboxes_dict[i]['ymax'])]

        # Extracting all the ground truth bounding boxes and labels
        self.current_gt_bboxes = gt_bboxes_dict

        # Setting the target bounding box
        self.target_bbox = self.current_gt_bboxes[0]
        
        # Incrementing the class image index
        self.class_image_index += 1

        # For Evaluation, appending the ground truth bounding boxes to the evaluation results
        if self.env_mode == TEST_MODE: # Testing mode
            # Appending the ground truth bounding boxes to the evaluation results
            self.evaluation_results['gt_boxes'][img_name] = self.current_gt_bboxes
        pass

    def save_evaluation_results(self, path='evaluation_results'):
        """
            Function that saves the evaluation results to a file.

            Args:
                - Path: Path to save the evaluation results
        """
        # Creating the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Saving the evaluation results to a numpy file
        np.save(os.path.join(path, self.evaluation_results['class'] + '_evaluation_results.npy'), self.evaluation_results)
        pass

    def load_evaluation_results(self, path='evaluation_results'):
        """
            Function that loads the evaluation results from a file.

            Args:
                - Path: Path to load the evaluation results
        """
        # Loading the evaluation results from a numpy file
        self.evaluation_results = np.load(os.path.join(path, self.evaluation_results['class'] + '_evaluation_results.npy'), allow_pickle=True)
        pass
    
    def filter_bboxes(self):
        """
            Function that filters the bounding boxes and adds them to the evaluation results.
        """
        # For Evaluation (Testing), appending the bounding boxes to the evaluation results
        if self.env_mode == TEST_MODE and self.use_dataset is not None: # Testing mode
            # Extracting the image name
            img_name = list(self.dataset[self.current_class].keys())[self.class_image_index - 1]

            # Appending the bounding boxes to the evaluation results
            # In this case no regression or cascading is done, and all bounding boxes are accepted
            self.evaluation_results['bounding_boxes'][img_name] = self.classification_dictionary['bbox']
            self.evaluation_results['labels'][img_name] = self.classification_dictionary['label']
            self.evaluation_results['confidences'][img_name] = self.classification_dictionary['confidence']
        pass
        
    def generate_initial_bbox(self, threshold=0.3, iterations=1):
        """
            Function that generates an initial bounding box prediction based on Saliency Ranking.

            Args:
                - Threshold: Threshold for the Saliency Ranking algorithm
                - Iterations: Number of iterations for the Saliency Ranking algorithm

            Output:
                - Initial bounding box prediction
        """
        # Creating a copy of the original image
        image = self.image.copy()

        # SaRa algorithm
        sara_info = sara.return_sara(image, GRID_SIZE, generator, mode=2)

        # Calculating the most important ranks to retain for the initial bounding box prediction based on the threshold and the number of iterations
        _ , _, sara_bbox = sara.sara_resize(self.image.copy(), sara_info, GRID_SIZE, rate=threshold, iterations=iterations)

        # Returning the initial bounding box prediction
        return sara_bbox
    
    def plot_sara(self, threshold=0.3):
        """
            Function that plots the Saliency Ranking algorithm.

            Args:
                - Threshold: Threshold for the Saliency Ranking algorithm
        """
        # Creating a copy of the original image
        image = self.image.copy()

        # SaRa algorithm
        sara_info = sara.return_sara(image, GRID_SIZE, generator, mode=1)

        # Plotting a 3D graph of the Saliency Ranking algorithm
        sara.plot_3D(self.image.copy(), sara_info, GRID_SIZE, rate=threshold)
        
        return sara_info