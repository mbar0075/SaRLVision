#-------------------------------------------------------------------------------
# Name:        models.py
# Purpose:     Defining models for SaRLVision.
#
# Author:      Matthias Bartolo <matthias.bartolo@ieee.org>
#
# Created:     February 24, 2024
# Copyright:   (c) Matthias Bartolo 2024-
# Licence:     All rights reserved
#-------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
from torch.nn.init import  uniform_
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.applications.resnet_v2 import ResNet50V2, decode_predictions, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from keras.applications.efficientnet_v2 import EfficientNetV2B3, decode_predictions, preprocess_input
from keras.applications.xception import Xception, decode_predictions, preprocess_input
from keras.applications.inception_v3 import InceptionV3, decode_predictions, preprocess_input

from SaRLVision.utils import device

import warnings
warnings.filterwarnings("ignore")

"""
    Defining the target size of the input image for each model.
"""
VGG16_TARGET_SIZE = (224, 224)
RESNET50_TARGET_SIZE = (224, 224)
MOBILENETV2_TARGET_SIZE = (224, 224)
EFFICIENTNETV2_TARGET_SIZE = (300, 300)
XCEPTION_TARGET_SIZE = (299, 299)
INCEPTIONV3_TARGET_SIZE = (299, 299)


"""
    VGG16 Feature Extractor (Feature Learning Model).
"""
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        self.vgg16_model = vgg16(weights=VGG16_Weights.DEFAULT).to(device) # Loading the pretrained model
        self.vgg16_model.eval() # Setting the model in evaluation mode to not do dropout.
        self.features = self.vgg16_model.features  # Retrieving the feature extraction part of the model
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))  # Adding a global average pooling layer

    def forward(self, x):# Forwarding the input through the model
        x = self.features(x)  # Applying the feature extraction part of the model
        x = self.pooling(x)  # Applying the global average pooling
        return x
    
    
"""
    ResNet50 Feature Extractor (Feature Learning Model).
"""
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        self.resnet50_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device) # Loading the pretrained model
        self.resnet50_model.eval() # Setting the model in evaluation mode to not do dropout.
        self.features = nn.Sequential(*list(self.resnet50_model.children())[:-2])# Retrieving the image feature extraction part of the model (excluding the last two layers which are the average pooling and the fully connected layer)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))  # Adding a global average pooling layer

    def forward(self, x):# Forwarding the input through the model
        x = self.features(x) # Applying the image feature extraction part of the model
        x = self.pooling(x)
        return x
    
    
"""
    MobileNetV2 Feature Extractor (Feature Learning Model).
"""
class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileNetV2FeatureExtractor, self).__init__()
        self.mobilenetv2 = mobilenet_v2(pretrained=True).to(device) # Loading the pretrained model
        self.mobilenetv2.eval() # Setting the model in evaluation mode to not do dropout.
        self.features = self.mobilenetv2.features  # Retrieving the feature extraction part of the model
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))  # Adding a global average pooling layer

    def forward(self, x):# Forwarding the input through the model
        x = self.features(x)  # Applying the feature extraction part of the model
        x = self.pooling(x)  # Applying the global average pooling
        return x


"""
    Method to transform the input image to the input of the model.
"""
def transform_input(image, target_size):
    """
        Transforming the input image to the input of the model.
        
        Args:
            image: The input image.
            target_size: The target size of the image.
            
        Returns:
            The transformed image.
    """
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
    ])
    return transform(image)


"""
    Architecture of the Vanilla (Standard) DQN model.
"""
class DQN(nn.Module):
    """
    The DQN network that estimates the action-value function

    Args:
        ninputs: The number of inputs
        noutputs: The number of outputs

    Layers:
        1. Linear layer with ninputs neurons
        2. ReLU activation function
        3. Dropout layer with 0.2 dropout rate
        4. Linear layer with 1024 neurons
        5. ReLU activation function
        6. Dropout layer with 0.2 dropout rate
        7. Linear layer with 512 neurons
        8. ReLU activation function
        9. Dropout layer with 0.2 dropout rate
        10. Linear layer with 256 neurons
        11. ReLU activation function
        12. Dropout layer with 0.2 dropout rate
        13. Linear layer with 128 neurons
        14. Output layer with noutputs neurons
    """
    def __init__(self, ninputs, noutputs):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(ninputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, noutputs)
        )

    def forward(self, X):
        # Forward pass
        return self.classifier(X)

    def __call__(self, X):
        return self.forward(X)


"""
    Architecture of the Dueling DQN model.
"""
class DuelingDQN(nn.Module):
    """
    The dueling DQN network that estimates the action-value function

    Args:
        ninputs: The number of inputs
        noutputs: The number of outputs

    Layers:
        1. Linear layer with ninputs neurons
        2. ReLU activation function
        3. Dropout layer with 0.2 dropout rate
        4. Linear layer with 1024 neurons
        5. ReLU activation function
        6. Dropout layer with 0.2 dropout rate
        7. Linear layer with 512 neurons
        8. ReLU activation function
        9. Dropout layer with 0.2 dropout rate
        10. Linear layer with 256 neurons
        11. ReLU activation function
        12. Dropout layer with 0.2 dropout rate
        13. Linear layer with 128 neurons
        14. ReLU activation function
        
    Value Function:
        1. Linear layer with 128 neurons
        2. Linear layer with 1 neuron

    Advantage Function:
        1. Linear layer with 128 neurons
        2. Linear layer with noutputs neurons

    Output:
        The value and advantage functions combined into the Q function (Q = V + A - mean(A))
        The dimensions of the output are noutputs
    """
    def __init__(self, ninputs, noutputs):
        super(DuelingDQN, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(ninputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.valfunc = nn.Linear(128, 1)
        self.advfunc = nn.Linear(128, noutputs)

    def forward(self, X):
        # Forward pass through shared layers
        o = self.shared_layers(X)
        # Splitting the output into the value and advantage functions
        value = self.valfunc(o)
        adv = self.advfunc(o)
        # Returning the value and advantage functions combined into the Q function (Q = V + A - mean(A))
        return value + adv - adv.mean(dim=-1, keepdim=True)
