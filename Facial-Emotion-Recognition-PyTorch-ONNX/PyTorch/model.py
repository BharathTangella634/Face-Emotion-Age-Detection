import torch.nn as nn
import torch
from ResNet import Bottleneck, ResNet, ResNet50

import torchvision.models as models

# class ResNet50(nn.Module):
#     def __init__(self, num_classes=7, grayscale=True, freeze_layers=True):
#         super(ResNet50, self).__init__()
        
#         # Load pretrained ResNet-50 model
#         self.model = models.resnet50(pretrained=True)
        
#         # Modify first convolutional layer for grayscale images (if needed)
#         if grayscale:
#             self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

#         # Freeze layers (except last block and classifier)
#         if freeze_layers:
#             for param in self.model.parameters():
#                 param.requires_grad = False  # Freeze all layers
            
#             for param in self.model.layer4.parameters():
#                 param.requires_grad = True  # Unfreeze last block

#         # Replace the final fully connected (FC) layer for custom classification
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)

class Face_Emotion_CNN(nn.Module):
  def __init__(self):
    super(Face_Emotion_CNN, self).__init__()
    self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
    self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
    self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
    self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
    self.cnn6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
    self.cnn7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
    self.relu = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2, 1)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.cnn1_bn = nn.BatchNorm2d(8)
    self.cnn2_bn = nn.BatchNorm2d(16)
    self.cnn3_bn = nn.BatchNorm2d(32)
    self.cnn4_bn = nn.BatchNorm2d(64)
    self.cnn5_bn = nn.BatchNorm2d(128)
    self.cnn6_bn = nn.BatchNorm2d(256)
    self.cnn7_bn = nn.BatchNorm2d(256)
    self.fc1 = nn.Linear(1024, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 7)
    self.dropout = nn.Dropout(0.3)
    self.log_softmax = nn.LogSoftmax(dim=1)
    
  def forward(self, x):
    x = self.relu(self.pool1(self.cnn1_bn(self.cnn1(x))))
    x = self.relu(self.pool1(self.cnn2_bn(self.dropout(self.cnn2(x)))))
    x = self.relu(self.pool1(self.cnn3_bn(self.cnn3(x))))
    x = self.relu(self.pool1(self.cnn4_bn(self.dropout(self.cnn4(x)))))
    x = self.relu(self.pool2(self.cnn5_bn(self.cnn5(x))))
    x = self.relu(self.pool2(self.cnn6_bn(self.dropout(self.cnn6(x)))))
    x = self.relu(self.pool2(self.cnn7_bn(self.dropout(self.cnn7(x)))))
    
    x = x.view(x.size(0), -1)
    
    x = self.relu(self.dropout(self.fc1(x)))
    x = self.relu(self.dropout(self.fc2(x)))
    x = self.log_softmax(self.fc3(x))
    return x

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


import torch
import torch.nn as nn
import timm  # For SENet
from torchvision import transforms

# Define SENet with Pretrained VGGFace2 Weights
class SENet_FER(nn.Module):
    def __init__(self, num_classes=7):
        super(SENet_FER, self).__init__()
        
        # Load SENet-50 model pretrained on VGGFace2
        self.model = timm.create_model("senet154", pretrained=True)  
        
        # Unfreeze layers in layer3 and layer4
        for name, param in self.model.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True  # Fine-tune these layers
            else:
                param.requires_grad = False  # Keep others frozen

        # Modify the classifier head for facial emotion recognition
        in_features = self.model.fc.in_features  # Get original FC layer size
        self.model.fc = nn.Linear(in_features, num_classes)  # Replace with new classifier

    def forward(self, x):
        return self.model(x)





if __name__ == '__main__':
    bn_model = Face_Emotion_CNN()
    x = torch.randn(1,1,48,48)
    net = ResNet50(7).to('cuda')
    print(x.shape)
    print('Shape of output = ',bn_model(x).shape)
    print(bn_model)
    print('No of Parameters of the BatchNorm-CNN Model =',bn_model.count_parameters())
    print(net)
    print('No of Parameters of the BatchNorm-CNN Model =',sum(p.numel() for p in net.parameters() if p.requires_grad))