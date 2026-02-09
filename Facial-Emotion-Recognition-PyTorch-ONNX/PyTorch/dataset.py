import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.utils.data as utils
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import *



# emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
#                 4: 'sad', 5: 'surprise', 6: 'neutral'}
emotion_dict = {0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness',
                    4: 'Anger', 5: 'Disguest', 6: 'Fear'}


def load_fer2013(path_to_fer_csv):
    data = pd.read_csv(path_to_fer_csv)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (48,48))
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = data['emotions'].values
    return faces, emotions


def show_random_data(faces, emotions):
  idx = np.random.randint(len(faces))
  print(emotion_dict[emotions[idx]])
  print(faces[idx])
  print(faces[idx].shape)
  plt.imshow(faces[idx].reshape(48,48), cmap='gray')
  plt.show()

class EmotionDataset(utils.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index].reshape(48,48)
        x = Image.fromarray((x))
        if self.transform is not None:
            x = self.transform(x)
        y = self.y[index]
        return x, y


def get_dataloaders(path_to_fer_csv=r'/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial-Emotion-Recognition-PyTorch-ONNX/PyTorch/total_ferplus.csv', tr_batch_sz=1028, val_batch_sz=512):
    faces, emotions = load_fer2013(path_to_fer_csv)
    # show_random_data(faces, emotions)

    # for i in range(5):
    #     show_random_data(faces, emotions)
    
    train_X, val_X, train_y, val_y = train_test_split(faces, emotions, test_size=0.2,
                                                random_state = 1, shuffle=True)
    print("Train Data size :", len(train_X))
    print("val Data size :", len(val_X))

    train_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(30),
                        transforms.ToTensor(),
                        transforms.Normalize((0.507395516207, ),(0.255128989415, )) 
                        ])
     
    val_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.507395516207, ),(0.255128989415, ))
                        ])  

    train_dataset = EmotionDataset(train_X, train_y, train_transform)
    val_dataset = EmotionDataset(val_X, val_y, val_transform)

    trainloader = utils.DataLoader(train_dataset, tr_batch_sz)
    validloader = utils.DataLoader(val_dataset, val_batch_sz)
    print(len(trainloader))
    print(len(validloader))

    return trainloader, validloader


if __name__ == '__main__':

    trainloader, validloader = get_dataloaders()

    model_path = r"/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial-Emotion-Recognition-PyTorch-ONNX/PyTorch/best_model.pt"

    # model = ResNet50(num_classes=7, channels=1)
    # model_path = r"C:\Users\CSE RGUKT\Facial-Emotion-Recognition-PyTorch-ONNX\PyTorch\models\FER_trained_model.pt"
    # model = ResNet50(num_classes=7, channels=1)

    model = Face_Emotion_CNN()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Check if the file contains only state_dict
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)


    # Evaluation function
    def evaluate_model(model, dataloader):
        correct = 0
        total = 0
        # criterion = nn.CrossEntropyLoss()
        count = 0
        with torch.no_grad():
            for images, labels in dataloader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                if(count <= 2):
                    print(outputs[0])
                    # print(outputs)
                count += 1
                total += labels.size(0)

        accuracy = 100 * correct / total
        return accuracy

    # Calculate accuracy
    accuracy = evaluate_model(model, validloader)
    print(f"Accuracy on training dataset: {accuracy:.2f}%")




    # import torch
    # import torch.nn as nn
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from model import *

    # # Load Data
    # trainloader, validloader = get_dataloaders()

    # # Load Model
    # model_path = r"C:\Users\CSE RGUKT\Facial-Emotion-Recognition-PyTorch-ONNX\PyTorch\models\FER_trained_model.pt"
    # # model = ResNet50(num_classes=7, channels=1)
    # model = Face_Emotion_CNN()

    # state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # # Check if state_dict is wrapped in another dict
    # if isinstance(state_dict, dict) and 'state_dict' in state_dict:
    #     model.load_state_dict(state_dict['state_dict'])
    # else:
    #     model.load_state_dict(state_dict)

    # model.eval()  # Set model to evaluation mode

    # # Emotion labels mapping
    # emotion_dict = {
    #     0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness',
    #     4: 'Anger', 5: 'Disgust', 6: 'Fear'
    # }

    # # Function to Display Images
    # def show_images_with_predictions(model, dataloader, num_examples=10):
    #     fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # Grid for 10 images
    #     axes = axes.flatten()
    #     examples_shown = 0

    #     with torch.no_grad():
    #         for images, labels in dataloader:
    #             outputs = model(images)  # Get raw logits
    #             _, predicted = torch.max(outputs, 1)  # Get highest probability class
                
    #             # Loop through batch
    #             for i in range(len(labels)):
    #                 if examples_shown < num_examples:
    #                     actual_emotion = emotion_dict[int(labels[i].item())]
    #                     predicted_emotion = emotion_dict[int(predicted[i].item())]

    #                     # Convert tensor to numpy for visualization
    #                     img = images[i].squeeze().numpy()  # Remove channel dimension

    #                     # Plot image
    #                     axes[examples_shown].imshow(img, cmap='gray')
    #                     axes[examples_shown].set_title(f"Actual: {actual_emotion}\nPredicted: {predicted_emotion}")
    #                     axes[examples_shown].axis('off')

    #                     examples_shown += 1
                    
    #                 if examples_shown >= num_examples:
    #                     break  # Stop after showing required examples

    #             if examples_shown >= num_examples:
    #                 break  # Stop the outer loop

    #     plt.tight_layout()
    #     plt.show()

    # # Run Visualization
    # show_images_with_predictions(model, validloader)

    # from collections import Counter
    # import torch

    # # Function to count labels in dataloader
    # def count_labels(dataloader):
    #     label_counts = Counter()
        
    #     for _, labels in dataloader:
    #         label_counts.update(labels.numpy())  # Convert tensor to numpy and count
        
    #     return label_counts

    # # Get label counts for train and valid loaders
    # train_counts = count_labels(trainloader)
    # valid_counts = count_labels(validloader)

    # # Emotion labels mapping
    # emotion_dict = {
    #     0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness',
    #     4: 'Anger', 5: 'Disgust', 6: 'Fear'
    # }

    # # Display counts with labels
    # print("\n🔹 **Training Set Label Counts:**")
    # for label, count in train_counts.items():
    #     print(f"{emotion_dict[label]}: {count}")

    # print("\n🔹 **Validation Set Label Counts:**")
    # for label, count in valid_counts.items():
    #     print(f"{emotion_dict[label]}: {count}")




