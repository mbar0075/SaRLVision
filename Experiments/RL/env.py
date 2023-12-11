import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *
from models import *
import time
import math
import colorsys

ACTION_HISTORY = [[100]*9]*10
NU = 10.0
THRESHOLD = 0.4
MAX_THRESHOLD = 1.0
GROWTH_RATE = 0.0009
ALPHA = 0.2
MAX_STEPS = 100
RENDER_MODE = None
FEATURE_EXTRACTOR = VGG16FeatureExtractor()
TARGET_SIZE = VGG16_TARGET_SIZE
CLASSIFIER = ResNet50V2()
CLASSIFIER_TARGET_SIZE = RESNET50_TARGET_SIZE


class DetectionEnv(gym.Env):
    def __init__(self, image, original_image, target_bbox, render_mode=RENDER_MODE, max_steps=MAX_STEPS, alpha=ALPHA, nu=NU, threshold=THRESHOLD, feature_extractor=FEATURE_EXTRACTOR, target_size=TARGET_SIZE, classifier=CLASSIFIER, classifier_target_size=CLASSIFIER_TARGET_SIZE):
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

    def calculate_reward(self, current_state, previous_state, target_bbox, reward_function=iou):
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

        # If the reward is smaller than 0, we return -1 else we return 1.
        if reward <= 0:
            return -1
        
        # Returning 1.
        return 1
    
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
        return -1*self.nu
    
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

        # Concatenating the features and the action history.
        state = torch.cat((features, action_history), 1)

        # Returning the state.
        return state
    
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
            self.actions_history[size_history_list][action] = 1
        else:
            # Else we shift the history list by one and we add the action vector to the history vector.
            for i in range(8,0,-1):
                self.actions_history[i][:] = self.actions_history[i-1][:]
            self.actions_history[0][:] = action_vector[:]

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

        # Displaying part (Retrieving a random color for the bounding box).
        self.color = self.generate_random_color()

        # Returning the observation space.
        return self.get_state(), {}
    
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
        self.display(mode='image', do_display=True, text_display=True)
        pass
    
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
    
    def render(self, mode='human'):
        pass
    
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
                self.plot_img(image, title='Step: ' + str(self.step_count) + ' | Reward: ' + str(self.cumulative_reward))

            # Returning the image.
            return image
        elif mode == 'bbox':
            # Creating a black image from the original image.
            image = np.zeros_like(self.original_image)

            # Drawing the bounding box on the image.
            cv2.rectangle(image, (x1, y1), (x2, y2), color, cv2.FILLED)

            # Plotting the image.
            if do_display:
                self.plot_img(image, title='Step: ' + str(self.step_count) + ' | Reward: ' + str(self.cumulative_reward))

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
                self.plot_img(heatmap, title='Step: ' + str(self.step_count) + ' | Reward: ' + str(self.cumulative_reward))

            # Returning the image.
            return heatmap
        
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
            plt.title(title)
        plt.show()
        
    def close(self):
        """
            Function that closes the environment.
        """
        gym.Env.close(self)
        pass
