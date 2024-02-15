import math
import torch
import torch.nn as nn
from torch.nn.init import  uniform_
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights, mobilenet_v2
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.applications.resnet_v2 import ResNet50V2, decode_predictions, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from keras.applications.efficientnet_v2 import EfficientNetV2, decode_predictions, preprocess_input
from keras.applications.xception import Xception, decode_predictions, preprocess_input
from keras.applications.inception_v3 import InceptionV3, decode_predictions, preprocess_input

import warnings
warnings.filterwarnings("ignore")

VGG16_TARGET_SIZE = (224, 224)
RESNET50_TARGET_SIZE = (224, 224)
MOBILENETV2_TARGET_SIZE = (224, 224)
EFFICIENTNETV2_TARGET_SIZE = (224, 224)
XCEPTION_TARGET_SIZE = (299, 299)
INCEPTIONV3_TARGET_SIZE = (299, 299)

"""
This script was inspired from the Active Object Localization paper.
"""

"""
    VGG16 Feature Extractor.
"""
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16_model = vgg16(weights=VGG16_Weights.DEFAULT) # Loading the pretrained model
        vgg16_model.eval() # Setting the model in evaluation mode to not do dropout.
        self.features = list(vgg16_model.children())[0] # Retrieving the first child of the model, which is typically the feature extraciton part of the model
        self.classifier = nn.Sequential(*list(vgg16_model.classifier.children())[:-2]) # Retrieving the feature extraction part of the model, and removing the last two layers, which are typically the dropout and the last layer of the model
        # self.adaptive_pooling = nn.AdaptiveAvgPool2d((2, 2)) # Defining the adaptive pooling layer to be used to transform the output of the model to a fixed size

    def forward(self, x):# Forwarding the input through the model
        x = self.features(x) # Applying the feature extraction part of the model
        # x = self.adaptive_pooling(x) # Applying the adaptive pooling layer
        x = torch.flatten(x, 1) # Flattening the output of the model
        # print(x.shape) # Printing the shape of the output
        return x
    
    
"""
    ResNet50 Feature Extractor.
"""
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        resnet50_model = resnet50(weights=ResNet50_Weights.DEFAULT) # Loading the pretrained model
        resnet50_model.eval() # Setting the model in evaluation mode to not do dropout.
        modules = list(resnet50_model.children())[:-2] # Retrieving the first child of the model, which is typically the feature extraciton part of the model
        self.features = nn.Sequential(*modules) # Retrieving the feature extraction part of the model, and removing the last two layers, which are typically the dropout and the last layer of the model
        # self.adaptive_pooling = nn.AdaptiveAvgPool2d((2, 2)) # Defining the adaptive pooling layer to be used to transform the output of the model to a fixed size
    
    def forward(self, x):# Forwarding the input through the model
        x = self.features(x) # Applying the feature extraction part of the model
        # x = self.adaptive_pooling(x) # Applying the adaptive pooling layer
        x = torch.flatten(x, 1) # Flattening the output of the model
        # print(x.shape) # Printing the shape of the output
        return x
    
    
"""
    MobileNetV2 Feature Extractor.
"""
class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileNetV2FeatureExtractor, self).__init__()
        mobilenetv2 = mobilenet_v2(pretrained=True)
        mobilenetv2.eval() # Setting the model in evaluation mode to not do dropout.
        self.features = mobilenetv2.features  # Extract features using the predefined function
        # self.adaptive_pooling = nn.AdaptiveAvgPool2d((2, 2)) # Define the adaptive pooling layer
    
    def forward(self, x):
        x = self.features(x)  # Feature extraction
        # x = self.adaptive_pooling(x)  # Adaptive pooling
        x = torch.flatten(x, 1)  # Flatten the output
        # print(x.shape) # Printing the shape of the output
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
    Architecture of the DQN model.
"""
# class DQN(nn.Module):
#     def __init__(self, h, w, outputs):
#         super(DQN, self).__init__()# Defining the layers of the model
#         self.classifier = nn.Sequential(
#             nn.Linear( in_features= 81 + 25088, out_features=1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear( in_features= 1024, out_features=1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear( in_features= 1024, out_features=9)
#         )
#     def forward(self, x):
#         return self.classifier(x)

class DQN(nn.Module):
    """
        The DQN network that estimates the action-value function

        Args:
            ninputs: The number of inputs
            noutputs: The number of outputs

        Layers:
            1. Linear layer with 64 neurons
            2. Tanh activation function
            3. Linear layer with noutputs neurons
    """
    def __init__(self, ninputs, noutputs):
        super(DQN, self).__init__()
        self.a1 = nn.Linear(ninputs, 1024)
        self.a2 = nn.Linear(1024, noutputs)

    def forward(self, X):
        # Forward pass
        o = self.a1(X)
        o = torch.tanh(o)
        o = self.a2(o)
        return o

    def __call__(self, X):
        return self.forward(X)

class DQN2(nn.Module):
    """
    The DQN network that estimates the action-value function

    Args:
        ninputs: The number of inputs
        noutputs: The number of outputs

    Layers:
        1. Linear layer with ninputs neurons
        2. Batch Normalization
        3. ReLU activation function
        4. Linear layer with ninputs//2 neurons
        5. Batch Normalization
        6. ReLU activation function
        7. Linear layer with ninputs//4 neurons
        8. Batch Normalization
        9. ReLU activation function
        10. Linear layer with ninputs//8 neurons
        11. Batch Normalization
        12. ReLU activation function
        13. Linear layer with noutputs neurons
    """
    def __init__(self, ninputs, noutputs):
        super(DQN2, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(ninputs, ninputs),
            nn.BatchNorm1d(ninputs),
            nn.ReLU(),
            nn.Linear(ninputs, ninputs // 2),
            nn.BatchNorm1d(ninputs // 2),
            nn.ReLU(),
            nn.Linear(ninputs // 2, ninputs // 4),
            nn.BatchNorm1d(ninputs // 4),
            nn.ReLU(),
            nn.Linear(ninputs // 4, ninputs // 8),
            nn.BatchNorm1d(ninputs // 8),
            nn.ReLU(),
            nn.Linear(ninputs // 8, noutputs)
        )

    def forward(self, X):
        # Forward pass
        return self.classifier(X)

    def __call__(self, X):
        return self.forward(X)


class DuelingDQN(nn.Module):
    """
        The dueling DQN network that estimates the action-value function

        Args:
            ninputs: The number of inputs
            noutputs: The number of outputs

        Layers:
            1. Linear layer with 64 neurons
            2. Tanh activation function
            3. Linear layer with 32 neurons
            4. Linear layer with 1 neuron
            5. Linear layer with 32 neurons
            6. Linear layer with noutputs neurons
    """
    def __init__(self, ninputs, noutputs):
        super(DuelingDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(ninputs, 1024),
            nn.Tanh(),
            nn.Linear(1024, 32)
        )
        # The value function estimates the value of being in a particular state
        self.valfunc = nn.Linear(32, 1)
        # The advantage function estimates how good it is to take a particular action in a given state
        self.advfunc = nn.Linear(32, noutputs)

    def forward(self, X):
        # Forward pass through the network
        o = self.net(X)
        # Splitting the output into the value and advantage functions
        value = self.valfunc(o)
        adv = self.advfunc(o)
        # Returning the value and advantage functions combined into the Q function (Q = V + A - mean(A))
        return value + adv - adv.mean(dim=-1, keepdim=True)

    def __call__(self, X):
        return self.forward(X)
    
class NoisyLinear(nn.Module):
    """
        The noisy linear layer that adds noise to the weights of the linear layer

        Args:
            in_size: The number of inputs
            out_size: The number of outputs

        Layers:
            1. Linear layer with in_size inputs and out_size outputs
            2. Linear layer with in_size inputs and out_size outputs
            3. Linear layer with out_size outputs
            4. Linear layer with out_size outputs
    """
    def __init__(self, in_size, out_size):
        super(NoisyLinear, self).__init__()
        # Defining the parameters of the layer as trainable parameters (weights and biases mu and sigma)
        self.w_mu = nn.Parameter(torch.empty((out_size, in_size)))
        self.w_sigma = nn.Parameter(torch.empty((out_size, in_size)))
        self.b_mu = nn.Parameter(torch.empty((out_size)))
        self.b_sigma = nn.Parameter(torch.empty((out_size)))

        # Creating the noise tensors for the weights and biases of the layer (w_epsilon and b_epsilon)
        uniform_(self.w_mu, -math.sqrt(3 / in_size), math.sqrt(3 / in_size))
        uniform_(self.b_mu, -math.sqrt(3 / in_size), math.sqrt(3 / in_size))

        # Initializing the noise tensors with the same shape as the weights and biases
        nn.init.constant(self.w_sigma, 0.017)
        nn.init.constant(self.b_sigma, 0.017)

    def forward(self, x, sigma=0.1): # Sigma Controls the amount of noise was 1 before
        # Forward pass through the layer
        if self.training: # If the model is in training mode, add noise to the weights and biases
            w_noise = torch.normal(0, sigma, size=self.w_mu.size())
            b_noise = torch.normal(0, sigma, size=self.b_mu.size())
            return F.linear(x, self.w_mu + self.w_sigma * w_noise, self.b_mu + self.b_sigma * b_noise)
        else:# If the model is in evaluation mode, return the mean of the weights and biases
            return F.linear(x, self.w_mu, self.b_mu)
        
class NoisyDQN(nn.Module):
    """
        The noisy DQN network that estimates the action-value function

        Args:
            ninputs: The number of inputs
            noutputs: The number of outputs

        Layers:
            1. Noisy linear layer with 64 neurons
            2. Tanh activation function
            3. Noisy linear layer with noutputs neurons
    """
    def __init__(self, ninputs, noutputs):
        super(NoisyDQN, self).__init__()
        self.a1 = NoisyLinear(ninputs, 1024)
        self.a2 = NoisyLinear(1024, noutputs)

    def forward(self, X):
        # Forward pass
        o = self.a1(X)
        o = torch.tanh(o)
        o = self.a2(o)
        return o

    def __call__(self, X):
        return self.forward(X)