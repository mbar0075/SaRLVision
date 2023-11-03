import cv2
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ImageSegmentationEnv(gym.Env):
    def __init__(self, image, target_mask):
        super(ImageSegmentationEnv, self).__init__()
        
        self.image = image  # The input image
        self.target_mask = target_mask  # Ground truth segmentation mask (will not necessarily be provided)
        
        self.height, self.width = image.shape
        self.observation_space = spaces.Box(0, 255, shape=(self.height, self.width), dtype=np.uint8)  # Assuming grayscale image
        self.action_space = spaces.Discrete(9)  # 8 movement actions (up, down, left, right, or stop)
        self.max_steps = 100  # Maximum number of steps in an episode
        self.step_count = 0  # Current step count
        self.jump_intensity = 1 # Number of pixels to move when an action is taken
        self.cumulative_reward = 0  # Cumulative reward

        self.bbox = [0, 0, self.width, self.height]  # Initial bounding box [X1, Y1, X2, Y2]

    def step(self, action):
        previous_iou = self.calculate_iou(self.bbox, self.target_mask)

        # Update the bounding box based on the chosen action
        if action == 0:  # Move X1 left
            self.bbox[0] -= self.jump_intensity
        elif action == 1:  # Move X1 right
            self.bbox[0] += self.jump_intensity
        elif action == 2:  # Move X2 left
            self.bbox[2] -= self.jump_intensity
        elif action == 3:  # Move X2 right
            self.bbox[2] += self.jump_intensity
        elif action == 4:  # Move Y1 up
            self.bbox[1] -= self.jump_intensity
        elif action == 5:  # Move Y1 down
            self.bbox[1] += self.jump_intensity
        elif action == 6:  # Move Y2 up
            self.bbox[3] -= self.jump_intensity
        elif action == 7:  # Move Y2 down
            self.bbox[3] += self.jump_intensity
        elif action == 8:  # Stop
            return self.image, 0, False, True, {}

        # Calculate IoU and compute the reward
        current_iou = self.calculate_iou(self.bbox, self.target_mask)
        reward = current_iou - previous_iou

        # Update the observation, step count, and done flag
        observation = self.image
        self.step_count += 1
        truncated = terminated = self.step_count >= self.max_steps

        return observation, reward, truncated, terminated, {}

    def calculate_iou(self, bbox, mask):
        # Calculate the IoU between the bounding box and the mask
        x1, y1, x2, y2 = bbox
        bbox_mask = np.zeros_like(mask)
        bbox_mask[y1:y2, x1:x2] = 1
        intersection = np.logical_and(bbox_mask, mask)
        union = np.logical_or(bbox_mask, mask)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def reset(self):
        # Reset the environment to its initial state
        self.bbox = [0, 0, self.width, self.height]
        self.step_count = 0
        self.cumulative_reward = 0
        return self.image

    def render(self, mode='mask'):
        if(mode == 'mask'):
            plt.imshow(self.target_mask, cmap='gray')
            plt.axis('off')
            plt.show()
        elif(mode == 'image'):
            # Creating a bounding box around the image
            x1, y1, x2, y2 = self.bbox
            image = self.image.copy()
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.show()
        else:
            pass

    def close(self):
        pass    