import cv2
import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *
from models import *
import time
import math
import colorsys
import pygame

ACTION_HISTORY = [[100]*9]*20
NU = 3.0
THRESHOLD = 1.0#0.95
MAX_THRESHOLD = 1.0
GROWTH_RATE = 0.0009
ALPHA = 0.1#0.1 #0.15
MAX_STEPS = 100#200
RENDER_MODE = "rgb_array" #None
FEATURE_EXTRACTOR = VGG16FeatureExtractor()
TARGET_SIZE = VGG16_TARGET_SIZE
CLASSIFIER = ResNet50V2()
CLASSIFIER_TARGET_SIZE = RESNET50_TARGET_SIZE
WINDOW_SIZE = 500
SIZE = 224
REWARD_FUNC = iou
ACTION_MODE =1 #0  

class DetectionEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    reward_penalty_dict = {
        0: 1,
        1: 0,
        2: 3,
        3: 2,
        4: 5,
        5: 4,
    }
    
    def __init__(self, image, original_image, target_bbox, render_mode=RENDER_MODE, mode=ACTION_MODE, max_steps=MAX_STEPS, alpha=ALPHA, nu=NU, threshold=THRESHOLD, feature_extractor=FEATURE_EXTRACTOR, target_size=TARGET_SIZE, classifier=CLASSIFIER, classifier_target_size=CLASSIFIER_TARGET_SIZE):
        """
            Constructor of the DetectionEnv class.

            Input:
                - Image
                - Original image
                - Target bounding box
                - Maximum number of steps
                - Alpha
                - Nu (Reward of trigger)
                - Threshold
                - Feature extractor
                - Target size

            Output:
                - Environment
        """
        super(DetectionEnv, self).__init__()
        # Initializing image, the original image which will be used as a visualisation, the target bounding box, the height and the width of the image.
        self.image = image
        self.original_image = original_image
        self.target_bbox = target_bbox
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.target_size = target_size

        # Initializing the actions history and the number of episodes.
        self.actions_history = []
        self.num_episodes = 0
        self.actions_history += ACTION_HISTORY

        # Initializing the bounding box of the image.
        self.bbox = [0, 0, self.width, self.height]

        # Initializing the feature extractor and the transform method.
        self.feature_extractor = feature_extractor
        self.transform = transform_input(self.image, target_size)

        # Initializing the action space and the observation space.
        # Action space is 9 because we have 8 actions + 1 trigger action (move right, move left, move up, move down, make bigger, make smaller, make fatter, make taller, trigger).
        self.action_space = gym.spaces.Discrete(9)
        self.action_mode = mode

        # Initializing the observation space.
        # Calculating the size of the state vector.
        state = self.get_state()
        # The observation space will be the features of the image concatenated with the history of the actions (size of the feature vector + size of the history vector).
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=100.0, # Since the values of the features are between 0 and 100.
            shape=(state.shape[1],),
            dtype=np.float32
        )

        # Setting terminated and truncated to False.
        self.terminated = False
        self.truncated = False
        
        # Initializing the maximum number of steps, the current step, the scaling factor of the reward, the reward of the trigger, the cumulative reward, the threshold, the actions history and the number of episodes.
        self.max_steps = max_steps
        self.step_count = 0
        self.alpha = alpha
        self.nu = nu # Reward of Trigger
        self.cumulative_reward = 0
        self.truncated = False
        self.threshold = threshold
        self.max_threshold = MAX_THRESHOLD
        self.growth_rate = GROWTH_RATE

        # Classification part
        self.label = None
        self.label_confidence = None
        self.classifier = classifier
        self.classifier_target_size = classifier_target_size

        # Displaying part (Retrieving a random color for the bounding box).
        self.color = self.generate_random_color()

        # For rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if render_mode is None:
            self.is_render = False
        else:
            self.is_render = True
        self.render_mode = render_mode
        self.window_size = WINDOW_SIZE
        self.size = SIZE
        self.window = None
        self.clock = None

        # For opposite actions
        self.current_action = None

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

        # scaling_factor = 0.01
        # normalized_value = -(self.step_count / self.max_steps) * scaling_factor
        # reward += normalized_value

        # scaling_factor = 0.01
        # normalized_value = -(self.reward_penalty()) * scaling_factor
        # reward += normalized_value

        return reward 
        # If the reward is smaller than 0, we return -1 else we return 1.
        if reward <= 0:
            return -1
        
        # Returning 1.
        return 1
    
    def reward_penalty(self):
        """
            Calculating the reward penalty for those actions which do the opposite of the previous action.

            Output:
                - Reward penalty

        """
        penalty = 0
        # Creating the action vector.
        action_vector = [0] * 9
        # Setting the current action to 1, based on the reward penalty dictionary which maps the action to the opposite action.
        if self.current_action in self.reward_penalty_dict.keys():
            action_vector[self.reward_penalty_dict[self.current_action]] = 1

        # print("Action vector: ", action_vector, "Actions history: ", self.actions_history)
        
        # Iterating over the history of the actions backwards.
        for i in range(len(self.actions_history)-1, -1, -1):
            
            # Checking if the action vector is equal to the action vector in the history of the actions, if yes we add the index to the penalty.
            if  action_vector == self.actions_history[i]:
                penalty += i

        # Returning the reward penalty.
        return penalty
    
    def calculate_trigger_reward(self, current_state, target_bbox, reward_function=iou):
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

        # Updating the threshold.
        # self.update_threshold()

        # If the reward is larger than the threshold, we return trigger reward else we return -1*trigger reward.
        if reward >= self.threshold:
            return self.nu*abs(reward)
        
        # Returning -1*trigger reward.
        return -1*self.nu#/abs(reward)
    
    def update_threshold(self):
        """
            Updating the threshold.
        
            Formula:
                threshold = max_threshold - (max_threshold / (1.0 + exp(growth_rate * num_episodes)))
        """
        # Calculating the new threshold, by growing the threshold.
        self.threshold = self.max_threshold - (self.max_threshold / (1.0 + math.exp(self.growth_rate * self.num_episodes)))

        # Clipping the threshold.
        self.threshold = min(self.threshold, self.max_threshold)
    
    def get_features(self, image, dtype=FloatTensor):
        """
            Getting the features of the image.

            Input:
                - Image

            Output:
                - Features of the image
        """
        # Transforming the image.
        image = transform_input(image, target_size=self.target_size)

        # Retrieving the features of the image.
        features = self.feature_extractor(image.unsqueeze(0))

        # Returning the features.
        return features

    def get_state(self, dtype=FloatDType):
        """
            Getting the state of the environment.

            Output:
                - State of the environment
        """
        #----------------------------------------------
        # Drawing the bounding box on the image.
        # xmin, ymin, xmax, ymax = self.bbox
        # image = self.original_image.copy()

        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

        # image = transform_input(image, target_size=self.target_size)
        #----------------------------------------------
        # Transforming the image.
        image = transform_input(self.image, target_size=self.target_size)

        # Retrieving the features of the image.
        features = self.get_features(image)

        # Transposing the features.
        features = features.view(1, -1).detach()

        # Flattenning the action history.
        action_history = torch.tensor(self.actions_history, dtype=dtype).flatten().view(1, -1)

        # Appending bounding box coordinates to the beginning of the action history.
        action_history = torch.cat((torch.tensor(self.bbox, dtype=dtype).view(1, -1), action_history), 1)
        # action_history = torch.tensor(self.bbox, dtype=dtype).view(1, -1)

        # Concatenating the features and the action history.
        state = torch.cat((action_history, features), 1)

        # Returning the state.
        return state.detach().cpu().numpy()
    
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
        action_vector = [0] * 9
        action_vector[action] = 1

        # Retrieving the size of the history list.
        size_history_list = len(self.actions_history)

        # If the size of the history list is smaller than 9, we add the action vector to the history vector.
        if size_history_list < 9:
            self.actions_history.append(action_vector)
        else:
            # Else we shift the history list by one and we add the action vector to the history vector.
            for i in range(8,0,-1):
                self.actions_history[i] = self.actions_history[i-1].copy()
            self.actions_history[0] = action_vector

        # Returning the history of the actions.
        return self.actions_history
    
    def transform_action(self, action):
        """
            Function that applies the action to the image.
        
            Input:
                - Action to apply

            Output:
                - Bounding box of the image depending on the action mode
        """
        if self.action_mode == 0:
            return self.transform_action_0(action)
        elif self.action_mode == 1:
            return self.transform_action_1(action)

    def get_actions(self):
        """
            Function that prints the name of the actions depending on the action mode.
        """
        if self.action_mode == 0:
            self.get_actions_0()
        elif self.action_mode == 1:
            self.get_actions_1()

    def decode_action(self, action):
        """
            Function that decodes the action depending on the action mode.

            Input:
                - Action to decode

            Output:
                - Decoded action
        """
        if self.action_mode == 0:
            self.decode_action_0(action)
        elif self.action_mode == 1:
            self.decode_action_1(action)
        
    def transform_action_0(self, action):
        """
        Function that applies the action to the image.

        Actions:
            - 0: X1 Left
            - 1: X1 Right
            - 2: X2 Left
            - 3: X2 Right
            - 4: Y1 Up
            - 5: Y1 Down
            - 6: Y2 Up
            - 7: Y2 Down

        Input:
            - Action to apply

        Output:
            - Bounding box of the image
        """
        # Retrieving the bounding box of the image.
        bbox = self.bbox

        # Retrieving the coordinates of the bounding box.
        xmin, xmax, ymin, ymax = bbox[0], bbox[2], bbox[1], bbox[3]

        # Calculating the alpha_h and alpha_w mentioned in the paper, and adding to it a decreasing factor depending on the step count.
        # alpha_h = int(self.alpha * (ymax - ymin)) + int(self.step_count * self.alpha * (ymax - ymin) / self.max_steps)
        # alpha_w = int(self.alpha * (xmax - xmin)) + int(self.step_count * self.alpha * (xmax - xmin) / self.max_steps)

        alpha_h = int(self.alpha * (ymax - ymin))
        alpha_w = int(self.alpha * (xmax - xmin))

        # If the action is 0, move X1 to the left.
        if action == 0:
            xmin -= alpha_w
            xmax -= alpha_w
        # If the action is 1, move X1 to the right.
        elif action == 1:
            xmin += alpha_w
            xmax += alpha_w
        # If the action is 2, move X2 to the left.
        elif action == 2:
            xmin -= alpha_w
            xmax -= alpha_w
        # If the action is 3, move X2 to the right.
        elif action == 3:
            xmin += alpha_w
            xmax += alpha_w
        # If the action is 4, move Y1 up.
        elif action == 4:
            ymin -= alpha_h
            ymax -= alpha_h
        # If the action is 5, move Y1 down.
        elif action == 5:
            ymin += alpha_h
            ymax += alpha_h
        # If the action is 6, move Y2 up.
        elif action == 6:
            ymin -= alpha_h
            ymax -= alpha_h
        # If the action is 7, move Y2 down.
        elif action == 7:
            ymin += alpha_h
            ymax += alpha_h

        # Returning the bounding box, ensuring it remains within the image bounds.
        return [self.rewrap(xmin, self.width), self.rewrap(ymin, self.height), self.rewrap(xmax, self.width), self.rewrap(ymax, self.height)]

    def get_actions_0(self):
        """
        Function that prints the name of the actions.
        """
        print('\033[1m' + "Actions:" + '\033[0m')
        print('\033[31m' + "0: X1 Right → " + '\033[0m')
        print('\033[32m' + "1: X1 Left ←" + '\033[0m')
        print('\033[33m' + "2: X2 Right →" + '\033[0m')
        print('\033[34m' + "3: X2 Left ←" + '\033[0m')
        print('\033[35m' + "4: Y1 Up ↑" + '\033[0m')
        print('\033[36m' + "5: Y1 Down ↓" + '\033[0m')
        print('\033[37m' + "6: Y2 Up ↑" + '\033[0m')
        print('\033[38m' + "7: Y2 Down ↓" + '\033[0m')
        print('\033[1m' + "8: Trigger T" + '\033[0m')
        pass

    def decode_action_0(self, action):
        """
        Function that decodes the action.

        Input:
            - Action to decode

        Output:
            - Decoded action
        """
        # If the action is 0, we print the name of the action.
        if action == 0:
            print('\033[31m' + "Action: X1 Right →" + '\033[0m')
        # If the action is 1, we print the name of the action.
        elif action == 1:
            print('\033[32m' + "Action: X1 Left ←" + '\033[0m')
        # If the action is 2, we print the name of the action.
        elif action == 2:
            print('\033[33m' + "Action: X2 Right →" + '\033[0m')
        # If the action is 3, we print the name of the action.
        elif action == 3:
            print('\033[34m' + "Action: X2 Left ←" + '\033[0m')
        # If the action is 4, we print the name of the action.
        elif action == 4:
            print('\033[35m' + "Action: Y1 Up ↑" + '\033[0m')
        # If the action is 5, we print the name of the action.
        elif action == 5:
            print('\033[36m' + "Action: Y1 Down ↓" + '\033[0m')
        # If the action is 6, we print the name of the action.
        elif action == 6:
            print('\033[37m' + "Action: Y2 Up ↑" + '\033[0m')
        # If the action is 7, we print the name of the action.
        elif action == 7:
            print('\033[38m' + "Action: Y2 Down ↓" + '\033[0m')
        # If the action is 8, we print the name of the action.
        elif action == 8:
            print('\033[1m' + "Action: Trigger T" + '\033[0m')
        pass


    def transform_action_1(self, action):
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

        # Calculating the alpha_h and alpha_w mentioned in the paper.
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

        # Returning the bounding box, whilst ensuring that the bounding box is within the image.
        return [self.rewrap(xmin, self.width), self.rewrap(ymin, self.height), self.rewrap(xmax, self.width), self.rewrap(ymax, self.height)]
    
    def get_actions_1(self):
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

    def decode_action_1(self, action):
        """
            Function that decodes the action.

            Input:
                - Action to decode

            Output:
                - Decoded action
        """
        # If the action is 0, we print the name of the action.
        if action == 0:
            print('\033[31m' + "Action: Move right →" + '\033[0m')
        # If the action is 1, we print the name of the action.
        elif action == 1:
            print('\033[32m' + "Action: Move left ←" + '\033[0m')
        # If the action is 2, we print the name of the action.
        elif action == 2:
            print('\033[33m' + "Action: Move up ↑" + '\033[0m')
        # If the action is 3, we print the name of the action.
        elif action == 3:
            print('\033[34m' + "Action: Move down ↓" + '\033[0m')
        # If the action is 4, we print the name of the action.
        elif action == 4:
            print('\033[35m' + "Action: Make bigger +" + '\033[0m')
        # If the action is 5, we print the name of the action.
        elif action == 5:
            print('\033[36m' + "Action: Make smaller -" + '\033[0m')
        # If the action is 6, we print the name of the action.
        elif action == 6:
            print('\033[37m' + "Action: Make fatter W" + '\033[0m')
        # If the action is 7, we print the name of the action.
        elif action == 7:
            print('\033[38m' + "Action: Make taller H" + '\033[0m')
        # If the action is 8, we print the name of the action.
        elif action == 8:
            print('\033[1m' + "Action: Trigger T" + '\033[0m')
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
                - Information of the environment
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
            'iou': iou(self.bbox, self.target_bbox),
            'recall': recall(self.bbox, self.target_bbox),
            'threshold': self.threshold,
            'label': self.label,
            'label_confidence': self.label_confidence,
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
    
    def reset(self, seed=None, options=None, image=None, original_image=None, target_bbox=None, max_steps=MAX_STEPS, alpha=ALPHA, nu=NU, threshold=THRESHOLD, feature_extractor=FEATURE_EXTRACTOR, target_size=TARGET_SIZE, classifier=CLASSIFIER, classifier_target_size=CLASSIFIER_TARGET_SIZE):
        """
            Function that resets the environment.

            Input:
                - Seed
                - Image
                - Original image
                - Target bounding box
                - Maximum number of steps
                - Alpha
                - Nu
                - Threshold
                - Feature extractor
                - Target size

            Output:
                - State of the environment
        """
        super().reset(seed=seed)
        # Initializing image, the original image which will be used as a visualisation, the target bounding box, the height and the width of the image.
        if image is not None:
            self.image = image
            self.height = image.shape[0]
            self.width = image.shape[1]
        if original_image is not None:
            self.original_image = original_image
        if target_bbox is not None:
            self.target_bbox = target_bbox
        self.target_size = target_size
        
        # Initializing the maximum number of steps, the current step, the scaling factor of the reward, the reward of the trigger, the cumulative reward, the threshold, the actions history and the number of episodes.
        self.max_steps = max_steps
        self.step_count = 0
        self.alpha = alpha
        self.nu = nu # Reward of Trigger
        self.cumulative_reward = 0
        self.truncated = False
        self.terminated = False
        self.threshold = threshold

        # Initializing the actions history and the number of episodes.
        self.actions_history = []
        self.actions_history += ACTION_HISTORY

        # Initializing the bounding box of the image.
        self.bbox = [0, 0, self.width, self.height]

        # Initializing the feature extractor and the transform method.
        self.feature_extractor = feature_extractor
        self.transform = transform_input(self.image, target_size)

        # Classification part
        self.label = None
        self.label_confidence = None
        self.classifier = classifier
        self.classifier_target_size = classifier_target_size

        self.current_action = None

        # Displaying part (Retrieving a random color for the bounding box).
        self.color = self.generate_random_color()

        # Returning the observation space.
        return self.get_state(), self.get_info()
    
    def get_label(self):
        """
            Function that returns the label of the image.

            Output:
                - Label of the image
        """
        # Retrieving the bounding box coordinates.
        x1, y1, x2, y2 = self.bbox

        # Cropping the image.
        image = self.original_image[y1:y2, x1:x2]

        # Resize the image to the target size using OpenCV
        image = cv2.resize(image, self.classifier_target_size)

        # Prepare the image for the VGG16 model
        image = preprocess_input(image)

        # Expanding the dimensions to match the model's expectations
        image = np.expand_dims(image, axis=0)

        # Predicting the class
        preds = self.classifier.predict(image, verbose=0)

        # Retrieving the Label and the confidence of the image.
        label = decode_predictions(preds, top=1)[0][0]

        # Retrieving the label and the confidence.
        self.label = label[1]
        self.label_confidence = label[2]

        # Returning the label and the confidence.
        return self.label, self.label_confidence
    
    def predict(self):
        """
            Function that predicts the label of the image.
        """
        # Retrieving the label and the confidence of the image.
        self.get_label()

        # Displaying the image.
        image = self.display(mode='image', do_display=True, text_display=True)
        
        # Returning the image.
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
        if action < 8:
            # Retrieving the previous state
            previous_state = [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]]

            # Applying the action to the image.
            self.bbox = self.transform_action(action)

            # Retrieving the current state.
            current_state = [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]]

            # Calculating the reward.
            reward = self.calculate_reward(current_state, previous_state, self.target_bbox)
        else:
            # Retrieving the current state.
            current_state = self.bbox

            # Calculating the reward.
            reward = self.calculate_trigger_reward(current_state, self.target_bbox)

            # Setting the episode to be terminated.
            self.terminated = True

        # Calculating the cumulative reward.
        self.cumulative_reward += reward

        # Incrementing the step count.
        self.step_count += 1

        # Checking if the episode is finished and truncated.
        if self.step_count >= self.max_steps:
            self.terminated = True
            self.truncated = False

        # If the episode is finished, we increment the number of episodes.
        if self.terminated or self.truncated:
            self.num_episodes += 1

        # Returning the state of the environment, the reward, whether the episode is finished or not, whether the episode is truncated or not and the information of the environment.
        return self.get_state(), reward, self.terminated, self.truncated, self.get_info()
    
    def render(self, mode='rgb_array'):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        return self._render_frame(mode=mode)

    def _render_frame(self, mode='rgb_array', do_display=False, text_display=True, alpha=0.4, color=(0, 255, 0)):
        # Retrieving bounding box coordinates.
        x1, y1, x2, y2 = self.bbox

        # Create a Pygame Surface to render on
        canvas = pygame.Surface((self.window_size, self.window_size))

        # Checking the mode of rendering.
        if mode == 'rgb_array':
            # Fill the canvas with white color
            canvas.fill((255, 255, 255))

            # Create a filled rectangle for the bounding box
            pygame.draw.rect(canvas, self.color, (x1, y1, x2 - x1, y2 - y1), 0)
            pygame.draw.rect(canvas, self.color, (x1, y1, x2 - x1, y2 - y1), 3)

            # Adding the label to the image
            if text_display and self.label is not None:
                font = pygame.font.Font(None, 30)
                text = str(self.label.capitalize()) + '  ' + str(round(self.label_confidence, 2))
                text_surface = font.render(text, True, (255, 255, 255))
                canvas.blit(text_surface, (x1, y1))

        elif mode == 'bbox':
            # Fill the canvas with black color
            canvas.fill((0, 0, 0))

            # Create a rectangle for the bounding box
            pygame.draw.rect(canvas, color, (x1, y1, x2 - x1, y2 - y1), 0)

        elif mode == 'heatmap':
            # Fill the canvas with black color
            canvas.fill((0, 0, 0))

            # Create a rectangle for the bounding box
            pygame.draw.rect(canvas, color, (x1, y1, x2 - x1, y2 - y1), 0)
            # Apply color map to the rectangle
            heatmap = cv2.applyColorMap(canvas, cv2.COLORMAP_JET)
            canvas = pygame.surfarray.make_surface(heatmap.swapaxes(0, 1))

        if do_display:
            # Display the frame
            self.window.blit(canvas, (0, 0))
            pygame.display.update()

        return pygame.surfarray.array3d(canvas)
    
    def display(self, mode='image', do_display=False, text_display=True, alpha=0.4, color=(0, 255, 0)):
        """
            Function that renders the environment.

            Input:
                - Mode of rendering
                - Whether to display the image or not
                - Whether to display the text or not
                - Alpha (transparency of the bounding box)
                - Color of the bounding box

            Output:
                - Image of the environment
        """
        # Retrieving bounding box coordinates.
        x1, y1, x2, y2 = self.bbox

        # Checking the mode of rendering.
        if mode == 'image':
            # Creating a copy of the original image.
            image_copy = self.original_image.copy()

            # Creating a filled rectangle for the bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), self.color, cv2.FILLED)

            # Blending the image with the rectangle using cv2.addWeighted
            image = cv2.addWeighted(self.image, 1 - alpha, image_copy, alpha, 0)

            # Adding a rectangle outline to the image
            cv2.rectangle(image, (x1, y1), (x2, y2), self.color, 3)

            # Adding the label to the image
            if text_display and self.label is not None:
                # Setting the font and the font scale
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                
                text = str(self.label.capitalize()) + '  ' + str(round(self.label_confidence, 2))

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
                cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), self.color, -1)

                # Adding the label text to the image
                cv2.putText(image, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

            # Plotting the image.
            if do_display:
                self.plot_img(image, title='Step: ' + str(self.step_count) + ' | Reward: ' + str(round(self.cumulative_reward, 3)) + ' | IoU: ' + str(round(iou(self.bbox, self.target_bbox), 3)) + ' | Recall: ' + str(round(recall(self.bbox, self.target_bbox), 3)))

            # Returning the image.
            return image
        elif mode == 'bbox':
            # Creating a black image from the original image.
            image = np.zeros_like(self.original_image)

            # Drawing the bounding box on the image.
            cv2.rectangle(image, (x1, y1), (x2, y2), color, cv2.FILLED)

            # Plotting the image.
            if do_display:
                self.plot_img(image, title='Step: ' + str(self.step_count) + ' | Reward: ' + str(round(self.cumulative_reward, 3)) + ' | IoU: ' + str(round(iou(self.bbox, self.target_bbox), 3)) + ' | Recall: ' + str(round(recall(self.bbox, self.target_bbox), 3)))

            # Returning the image.
            return image
        elif mode == 'heatmap':
            # Creating a black image.
            image = np.zeros_like(self.original_image)

            # Drawing the bounding box on the image.
            cv2.rectangle(image, (x1, y1), (x2, y2), color, cv2.FILLED)

            # Creating the heatmap.
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)

            # Plotting the image.
            if do_display:
                self.plot_img(heatmap, title='Step: ' + str(self.step_count) + ' | Reward: ' + str(round(self.cumulative_reward, 3)) + ' | IoU: ' + str(round(iou(self.bbox, self.target_bbox), 3)) + ' | Recall: ' + str(round(recall(self.bbox, self.target_bbox), 3)))

            # Returning the image.
            return heatmap
        
    def segment(self, display_mode="mask", do_display=False, text_display=True, alpha=0.7, color=(0, 255, 0)):
        """
            Function that segments the object in the bounding box.

            Input:
                - Whether to display the image or not
                - Whether to display the text or not
                - Alpha (transparency of the bounding box)
                - Color of the bounding box

            Output:
                - Segment Mask
        """
        # Retrieving bounding box coordinates.
        x1, y1, x2, y2 = self.bbox

        # Creating a black 3-channel mask image.
        mask = np.zeros_like(self.original_image)

        # Changing to 1-channel mask image.
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # Extracting the object from the mask.
        bbox_object = self.original_image[y1:y2, x1:x2]

        # From the bbox_object, we extract the mask via Canny Edge Detection.
        edges = cv2.Canny(bbox_object, 100, 200)

        # We dilate the edges using a larger kernel and multiple iterations.
        kernel_dilation = np.ones((5,5),np.uint8)
        edges = cv2.dilate(edges, kernel_dilation, iterations = 3)

        # We erode the edges using a smaller kernel and fewer iterations.
        kernel_erosion = np.ones((3,3),np.uint8)
        edges = cv2.erode(edges, kernel_erosion, iterations = 1)

        # We fill the holes in the edges.
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, None)

        # Find contours in the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours onto the mask and fill them
        for contour in contours:
            cv2.drawContours(mask[y1:y2, x1:x2], [contour], -1, (255), thickness=cv2.FILLED)

        # Apply Gaussian blur to smooth the mask
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Creating a filled polygon annotation for the object mask on a copy of the original image.
        image_copy = self.original_image.copy()       

        # Plotting the image.
        if do_display:
            if display_mode == "mask":
                self.plot_img(mask, title='Segmentation Mask')

            elif display_mode == "image":
                # Creating a filled polygon annotation for the object mask on a copy of the original image.
                image_copy = self.original_image.copy()

                # Offset the contour points by the top-left coordinates of the bounding box
                contour_offset = contour + np.array([x1, y1])

                cv2.fillPoly(image_copy, pts=[contour_offset], color=self.color)

                # Blending the image with the rectangle using cv2.addWeighted
                image = cv2.addWeighted(self.image, 1 - alpha, image_copy, alpha, 0)

                # Adding a rectangle outline to the image
                cv2.rectangle(image, (x1, y1), (x2, y2), self.color, 3)

                if text_display and self.label is not None:
                    # Setting the font and the font scale
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2
                    
                    text = str(self.label.capitalize()) + '  ' + str(round(self.label_confidence, 2))

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
                    cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), self.color, -1)

                    # Adding the label text to the image
                    cv2.putText(image, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

                self.plot_img(image, title='Instance Segmentation')

        # Returning the image mask.
        return mask
        
    def plot_img(self, image, title=None):
        """
            Function that plots the image.

            Input:
                - Image to plot
        """
        # Plotting the image.
        plt.figure(figsize=(10, 7))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        if title is not None:
            plt.title(title, fontsize=14)
        plt.show()
        
    def close(self):
        """
            Function that closes the environment.
        """
        # Close the Pygame window
        if self.is_render:
            pygame.quit()

        Env.close(self)
        pass
