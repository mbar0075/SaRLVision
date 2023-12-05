import torch.nn as nn
import torchvision

"""
This file was inspired from the Active Object Localization paper.
"""

"""
    VGG16 Feature Extractor.
"""
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)# Loading the pretrained model
        vgg16.eval() # Setting the model in evaluation mode to not do dropout.
        self.features = list(vgg16.children())[0] # Getting the features of the model
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2]) # Getting the classifier of the model
    def forward(self, x):# Forwarding the input through the model
        x = self.features(x)
        return x
    
"""
    ResNet50 Feature Extractor.
"""
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)# Loading the pretrained model
        resnet50.eval() # Setting the model in evaluation mode to not do dropout.
        self.features = nn.Sequential(*list(resnet50.children())[:-1]) # Getting the features of the model
    def forward(self, x):# Forwarding the input through the model
        x = self.features(x)
        return x
    
"""
    EfficientNetB0 Feature Extractor.
"""
class EfficientNetB0FeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetB0FeatureExtractor, self).__init__()
        efficientnetb0 = torchvision.models.mobilenet_v2(pretrained=True)# Loading the pretrained model
        efficientnetb0.eval() # Setting the model in evaluation mode to not do dropout.
        self.features = nn.Sequential(*list(efficientnetb0.children())[:-1]) # Getting the features of the model
    def forward(self, x):# Forwarding the input through the model
        x = self.features(x)
        return x
    
"""
    MobileNetV2 Feature Extractor.
"""
class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileNetV2FeatureExtractor, self).__init__()
        mobilenetv2 = torchvision.models.mobilenet_v2(pretrained=True)# Loading the pretrained model
        mobilenetv2.eval() # Setting the model in evaluation mode to not do dropout.
        self.features = nn.Sequential(*list(mobilenetv2.children())[:-1]) # Getting the features of the model
    def forward(self, x):# Forwarding the input through the model
        x = self.features(x)
        return x
    
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