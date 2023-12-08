import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights, mobilenet_v2
import warnings
warnings.filterwarnings("ignore")

VGG16_TARGET_SIZE = (224, 224)
RESNET50_TARGET_SIZE = (224, 224)
MOBILENETV2_TARGET_SIZE = (224, 224)

"""
This file was inspired from the Active Object Localization paper.
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
        self.classifier = nn.Sequential(*list(vgg16_model.classifier.children())[:-2]) # Retrieving the classifier part of the model, and removing the last two layers, which are typically the dropout and the last layer of the model
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((7, 7)) # Defining the adaptive pooling layer to be used to transform the output of the model to a fixed size

    def forward(self, x):# Forwarding the input through the model
        x = self.features(x) # Applying the feature extraction part of the model
        x = self.adaptive_pooling(x) # Applying the adaptive pooling layer
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
        self.features = nn.Sequential(*modules) # Retrieving the classifier part of the model, and removing the last two layers, which are typically the dropout and the last layer of the model
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((7, 7)) # Defining the adaptive pooling layer to be used to transform the output of the model to a fixed size
    
    def forward(self, x):# Forwarding the input through the model
        x = self.features(x) # Applying the feature extraction part of the model
        x = self.adaptive_pooling(x) # Applying the adaptive pooling layer
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
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((7, 7)) # Define the adaptive pooling layer
    
    def forward(self, x):
        x = self.features(x)  # Feature extraction
        x = self.adaptive_pooling(x)  # Adaptive pooling
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
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()# Defining the layers of the model
        self.classifier = nn.Sequential(
            nn.Linear( in_features= 81 + 25088, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=9)
        )
    def forward(self, x):
        return self.classifier(x)