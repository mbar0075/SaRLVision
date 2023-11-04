import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

class ObjectLocalizationEnv(gym.Env):
    def __init__(self, image, target_box, max_steps=100, alpha=0.2):
        super(ObjectLocalizationEnv, self).__init__()

        self.image = image  # The input image
        self.target_box = target_box  # Ground truth box for the target object
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.observation_space = spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8)  # RGB image
        self.action_space = spaces.Discrete(9)  # 8 transformations and 1 trigger action
        self.max_steps = max_steps
        self.step_count = 0
        self.alpha = alpha  # Transformation factor
        self.cumulative_reward = 0
        self.truncated = False
        self.history_vector = np.zeros(9, dtype=np.uint8)  # History of taken actions

        # Initialize the bounding box [x1, y1, x2, y2]
        self.bbox = [0, 0, self.width, self.height]

        # Load a pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier = nn.Sequential(*list(self.vgg16.classifier.children())[:-3])  # Remove the last layers

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def step(self, action):
        previous_iou = self.calculate_iou(self.bbox, self.target_box)

        if action < 8:  # Transformation actions
            # Apply the transformation to the bounding box
            self.transform_box(action)
        elif action == 8:  # Trigger action
            return self.image, 0, True, False, {}
        elif self.truncated:
            return self.image, -1.0, True, True, {}

        # Calculate IoU and compute the reward
        current_iou = self.calculate_iou(self.bbox, self.target_box)
        reward = self.calculate_reward(previous_iou, current_iou)

        # Update the observation, step count, and done flag
        observation = self.get_state()
        self.step_count += 1
        terminated = self.step_count >= self.max_steps

        return observation, reward, terminated, self.truncated, {}

    def transform_box(self, action):
        x1, y1, x2, y2 = self.bbox
        alpha_w = int(self.alpha * (x2 - x1))
        alpha_h = int(self.alpha * (y2 - y1))

        # Calculate the new coordinates without actually updating the bounding box
        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

        if action == 0:  # Move X1 left
            new_x1 = max(0, new_x1 - alpha_w)  # Ensure new_x1 is not outside the left boundary
        elif action == 1:  # Move X1 right
            new_x1 = min(self.width, new_x1 + alpha_w)  # Ensure new_x1 is not outside the right boundary
        elif action == 2:  # Move X2 left
            new_x2 = max(0, new_x2 - alpha_w)  # Ensure new_x2 is not outside the left boundary
        elif action == 3:  # Move X2 right
            new_x2 = min(self.width, new_x2 + alpha_w)  # Ensure new_x2 is not outside the right boundary
        elif action == 4:  # Move Y1 up
            new_y1 = max(0, new_y1 - alpha_h)  # Ensure new_y1 is not outside the top boundary
        elif action == 5:  # Move Y1 down
            new_y1 = min(self.height, new_y1 + alpha_h)  # Ensure new_y1 is not outside the bottom boundary
        elif action == 6:  # Move Y2 up
            new_y2 = max(0, new_y2 - alpha_h)  # Ensure new_y2 is not outside the top boundary
        elif action == 7:  # Move Y2 down
            new_y2 = min(self.height, new_y2 + alpha_h)  # Ensure new_y2 is not outside the bottom boundary

        # Check if the new coordinates are valid
        if self.check_valid_action():
            # Update the bounding box with the new coordinates
            self.bbox = [new_x1, new_y1, new_x2, new_y2]
        else:
            self.truncated = True


    def calculate_iou(self, bbox, target_box):
        # print("bbox: ", bbox)
        # print("target_box: ", target_box)
        # Calculate the IoU between the bounding box and the target box
        x1, y1, x2, y2 = bbox
        x1_gt, y1_gt, x2_gt, y2_gt = target_box

        intersection_x1 = max(x1, x1_gt)
        intersection_y1 = max(y1, y1_gt)
        intersection_x2 = min(x2, x2_gt)
        intersection_y2 = min(y2, y2_gt)

        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
        bbox_area = float((x2 - x1) * (y2 - y1))
        target_area = float((x2_gt - x1_gt) * (y2_gt - y1_gt))

        iou = intersection_area / (bbox_area + target_area - intersection_area)
        return iou


    def calculate_reward(self, previous_iou, current_iou):
        # Calculate the reward based on IoU improvement
        return np.sign(current_iou - previous_iou)

    def get_state(self):
        # Extract features from the current region using the VGG16 model
        # change bbox to int
        self.bbox = [int(i) for i in self.bbox]
        x1, y1, x2, y2 = self.bbox
        # Make sure the bounding box is within the image

        region = self.image[y1:y2, x1:x2]
        region = self.transform(region)
        region = torch.unsqueeze(region, 0)  # Add batch dimension
        features = self.vgg16(region)
        features = features.view(-1).detach().numpy()

        # Combine features (o) and history (h) into the state representation
        state = (features, self.history_vector)

        return state

    def check_valid_action(self):
        # Check if the action is valid
        x1, y1, x2, y2 = self.bbox
        if x1 < 0 or y1 < 0 or x2 > self.width or y2 > self.height:
            return False
        return True
    
    def reset(self):
        # Reset the environment to its initial state
        self.step_count = 0
        self.cumulative_reward = 0
        self.history_vector = np.zeros(9, dtype=np.uint8)
        self.bbox = [0, 0, self.width, self.height]
        self.truncated = False
        return self.get_state()

    def render(self):
        # Render the environment
        x1, y1, x2, y2 = self.bbox
        image = self.image.copy()
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.imshow(image)
        plt.axis('off')
        plt.show()