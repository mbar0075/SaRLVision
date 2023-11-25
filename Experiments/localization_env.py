import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

class ObjectLocalizationEnv(gym.Env):
    def __init__(self, image, original_image, target_box, max_steps=1000, alpha=0.2, iou_threshold=0.9, trigger_delay=10):
        super(ObjectLocalizationEnv, self).__init__()

        self.image = image
        self.original_image = original_image
        self.target_box = target_box
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.observation_space = gym.spaces.Box(0, 1, shape=(3, 224, 224), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)
        self.max_steps =self.width+self.height #max_steps
        self.step_count = 0
        self.alpha = alpha
        self.cumulative_reward = 0
        self.truncated = False
        self.iou_threshold = iou_threshold
        self.trigger_delay = trigger_delay
        self.history_vector = np.zeros(9, dtype=np.float32)

        self.bbox = [0, 0, self.width, self.height]

        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16 = nn.Sequential(*list(self.vgg16.features.children())[:-1])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def step(self, action):
        # if self.cumulative_reward < 0:
        #     print("Negative cumulative reward")
        #     return self.get_state(), 0.0, False, True, {}  # Stop if the cumulative reward is negative

        previous_iou = self.calculate_iou(self.bbox, self.target_box)

        if action < 4:  # Transformation actions
            self.transform_box(action)
        elif action == 4 and self.step_count >= self.trigger_delay:  # Trigger action after delay
            print("Trigger action")
            return self.get_state(), previous_iou, True, False, {}

        if self.truncated:
            print("Truncated")
            return self.get_state(), -previous_iou, False, True, {}

        current_iou = self.calculate_iou(self.bbox, self.target_box)
        reward = self.calculate_reward(previous_iou, current_iou)

        if self.truncated:
            print("Reward Truncated")

        self.cumulative_reward += reward  # Update cumulative reward
        observation = self.get_state()
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        if(self.step_count >= self.max_steps):
            print("Max steps reached")

        return observation, reward, terminated, self.truncated, {}


    # def transform_box(self, action):
    #     x1, y1, x2, y2 = self.bbox
    #     alpha_w = int(self.alpha * (x2 - x1))
    #     alpha_h = int(self.alpha * (y2 - y1))

    #     new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

    #     if action == 0:  # Move X1 left
    #         print("Move X1 left")
    #         new_x1 = max(0, new_x1 - alpha_w)
    #     elif action == 1:  # Move X1 right
    #         print("Move X1 right")
    #         new_x1 = min(self.width, new_x1 + alpha_w)
    #     elif action == 2:  # Move X2 left
    #         print("Move X2 left")
    #         new_x2 = max(0, new_x2 - alpha_w)
    #     elif action == 3:  # Move X2 right
    #         print("Move X2 right")
    #         new_x2 = min(self.width, new_x2 + alpha_w)
    #     elif action == 4:  # Move Y1 up
    #         print("Move Y1 up")
    #         new_y1 = max(0, new_y1 - alpha_h)
    #     elif action == 5:  # Move Y1 down
    #         print("Move Y1 down")
    #         new_y1 = min(self.height, new_y1 + alpha_h)
    #     elif action == 6:  # Move Y2 up
    #         print("Move Y2 up")
    #         new_y2 = max(0, new_y2 - alpha_h)
    #     elif action == 7:  # Move Y2 down
    #         print("Move Y2 down")
    #         new_y2 = min(self.height, new_y2 + alpha_h)

    #     if self.check_valid_action(new_x1, new_y1, new_x2, new_y2):
    #         self.bbox = [new_x1, new_y1, new_x2, new_y2]
    #     else:
    #         self.truncated = True
    def transform_box(self, action):
        x1, y1, x2, y2 = self.bbox
         # Start with a large step then decay over time
        alpha_w = int(10 / (1 + self.step_count / 100))  # Exponential decay for alpha_w
        alpha_h = int(10 / (1 + self.step_count / 100))  # Exponential decay for alpha_h

        if action == 0:  # Move X1 right
            # print("Move X1 right")
            x1 = min(x1 + alpha_w, self.width - alpha_w)
        elif action == 1:  # Move Y1 down
            # print("Move Y1 down")
            y1 = min(y1 + alpha_h, self.height - alpha_h)
        elif action == 2:  # Move X2 left
            # print("Move X2 left")
            x2 = max(x2 - alpha_w, x1 + alpha_w)
        elif action == 3:  # Move Y2 up
            # print("Move Y2 up")
            y2 = max(y2 - alpha_h, y1 + alpha_h)

        if self.check_valid_action(x1, y1, x2, y2):
            self.bbox = [x1, y1, x2, y2]
        else:
            self.truncated = True

        # self.render()


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
        # Getting the coordinates of the bounding boxes
        x1, y1, x2, y2 = self.bbox
        x1_gt, y1_gt, x2_gt, y2_gt = self.target_box

        # Calculating the areas
        bbox_area = (x2 - x1) * (y2 - y1)
        target_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

        # Giving reward based on IoU improvement
        reward = current_iou - previous_iou

        if bbox_area < target_area:
                penalty = 0.5 * (target_area - bbox_area)  # Adjust the penalty factor as needed
                
                # Normalizing the penalty
                penalty = min(penalty / target_area, 0.3)  # Adjust the divisor based on the range of your values
                
                reward -= penalty
                self.truncated = True

        return reward


    # def get_state(self):
    #     # Use a bounding box in the state
    #     # do not crop the image
    #     self.bbox = [int(i) for i in self.bbox]
    #     x1, y1, x2, y2 = self.bbox

    #     region = self.image[y1:y2, x1:x2]
    #     region = cv2.resize(region, (224, 224))
    #     region = self.transform(region)
    #     features = self.vgg16(region)
    #     features = features.view(-1).detach().numpy()

    #     # Encode the state as a vector of features and history
    #     state = torch.cat((torch.tensor(features), torch.tensor(self.history_vector)))

    #     return state
    def get_state(self):
        # Extracting a bounding box tensor
        x1, y1, x2, y2 = self.bbox
        bbox_tensor = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        # Get region of the original image without cropping
        region = self.image.copy()  # Copy the entire image
        region = cv2.resize(region, (224, 224))  # Resize without cropping
        region = self.transform(region)
        features = self.vgg16(region)
        features = features.view(-1).detach().numpy()

        # Encode the state as a vector of bounding box and features and history
        state = torch.cat((torch.tensor(features), bbox_tensor, torch.tensor(self.history_vector)))

        return state

    def check_valid_action(self, new_x1, new_y1, new_x2, new_y2):
        target_x1, target_y1, target_x2, target_y2 = self.target_box
        if new_x1 < 0 or new_y1 < 0 or new_x2 > self.width or new_y2 > self.height:
            return False
        if new_x1 >= new_x2 or new_y1 >= new_y2:
            return False
        if new_x1 >= target_x2 or new_y1 >= target_y2 or new_x2 <= target_x1 or new_y2 <= target_y1:
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
        image = self.original_image.copy()
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if mode == 'image':
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        return image
