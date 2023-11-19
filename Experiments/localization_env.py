import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

class ObjectLocalizationEnv(gym.Env):
    def __init__(self, image, target_box, max_steps=100, alpha=0.2, iou_threshold=0.9):
        super(ObjectLocalizationEnv, self).__init__()

        self.image = image
        self.target_box = target_box
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.observation_space = gym.spaces.Box(0, 1, shape=(3, 224, 224), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9)
        self.max_steps = max_steps
        self.step_count = 0
        self.alpha = alpha
        self.cumulative_reward = 0
        self.truncated = False
        self.iou_threshold = iou_threshold

        self.bbox = [0, 0, self.width, self.height]

        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16 = nn.Sequential(*list(self.vgg16.features.children())[:-1])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def step(self, action):
        if self.cumulative_reward < 0:
            return self.get_state(), 0.0, True, False, {}  # Stop if the cumulative reward is negative

        previous_iou = self.calculate_iou(self.bbox, self.target_box)

        if action < 8:  # Transformation actions
            self.transform_box(action)
        elif action == 8:  # Trigger action
            return self.get_state(), 0.0, True, False, {}

        if self.truncated:
            return self.get_state(), 0.0, False, True, {}

        current_iou = self.calculate_iou(self.bbox, self.target_box)
        reward = self.calculate_reward(previous_iou, current_iou)

        self.cumulative_reward += reward  # Update cumulative reward

        # if current_iou >= self.iou_threshold:
        #     return self.get_state(), self.cumulative_reward, True, False, {}  # Stop if IoU threshold is reached

        observation = self.get_state()
        self.step_count += 1
        terminated = self.step_count >= self.max_steps

        return observation, reward, terminated, self.truncated, {}

    def transform_box(self, action):
        x1, y1, x2, y2 = self.bbox
        alpha_w = int(self.alpha * (x2 - x1))
        alpha_h = int(self.alpha * (y2 - y1))

        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

        if action == 0:  # Move X1 left
            new_x1 = max(0, new_x1 - alpha_w)
        elif action == 1:  # Move X1 right
            new_x1 = min(self.width, new_x1 + alpha_w)
        elif action == 2:  # Move X2 left
            new_x2 = max(0, new_x2 - alpha_w)
        elif action == 3:  # Move X2 right
            new_x2 = min(self.width, new_x2 + alpha_w)
        elif action == 4:  # Move Y1 up
            new_y1 = max(0, new_y1 - alpha_h)
        elif action == 5:  # Move Y1 down
            new_y1 = min(self.height, new_y1 + alpha_h)
        elif action == 6:  # Move Y2 up
            new_y2 = max(0, new_y2 - alpha_h)
        elif action == 7:  # Move Y2 down
            new_y2 = min(self.height, new_y2 + alpha_h)

        if self.check_valid_action(new_x1, new_y1, new_x2, new_y2):
            self.bbox = [new_x1, new_y1, new_x2, new_y2]
        else:
            self.truncated = True

    def calculate_iou(self, bbox, target_box):
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
        # Giving reward a boost if the IoU is above a threshold
        if current_iou >= self.iou_threshold:
            return 10 * (current_iou - previous_iou)
        return current_iou - previous_iou  # Reward based on IoU improvement

    def get_state(self):
        self.bbox = [int(i) for i in self.bbox]
        x1, y1, x2, y2 = self.bbox

        region = self.image[y1:y2, x1:x2]
        region = cv2.resize(region, (224, 224))
        region = self.transform(region)
        features = self.vgg16(region)
        features = features.view(-1).detach().numpy()

        # Encode the state as a vector of features and history
        state = torch.cat((torch.tensor(features), torch.tensor(self.history_vector)))

        return state

    def check_valid_action(self, new_x1, new_y1, new_x2, new_y2):
        if new_x1 < 0 or new_y1 < 0 or new_x2 > self.width or new_y2 > self.height:
            return False
        return True

    def reset(self):
        self.step_count = 0
        self.cumulative_reward = 0
        self.history_vector = np.zeros(9, dtype=np.float32)
        self.bbox = [0, 0, self.width, self.height]
        self.truncated = False
        return self.get_state()

    def render(self, mode='image'):
        x1, y1, x2, y2 = self.bbox
        image = self.image.copy()
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if mode == 'image':
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        return image
