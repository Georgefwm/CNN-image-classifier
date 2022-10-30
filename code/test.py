import torch
from torchvision import datasets, transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from torch.utils.data import random_split
import time

def main():


    desired_batch_size = 16

    test_transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor()
        ])

    # import dataset
    test_dataset = datasets.ImageFolder("traindata", transform=test_transform)

    # create loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=desired_batch_size,
                                                    shuffle=False, num_workers=2)

    # define our CNN
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.relu_leak = 0.05

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope=self.relu_leak)
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope=self.relu_leak),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope=self.relu_leak)
            )

            self.conv4 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope=self.relu_leak),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=self.relu_leak)
            )

            self.conv6 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=self.relu_leak),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.conv7 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=self.relu_leak)
            )

            self.conv8 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=self.relu_leak)
            )

            self.conv9 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=self.relu_leak),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.conv10 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(negative_slope=self.relu_leak)
            )

            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(negative_slope=self.relu_leak)
            )

            self.conv12 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(negative_slope=self.relu_leak),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.conv13 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(negative_slope=self.relu_leak)
            )

            self.conv14 = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(negative_slope=self.relu_leak)
            )

            self.conv15 = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(negative_slope=self.relu_leak),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.dropout_prob = 0.5

            self.fc1 = nn.Sequential(
                nn.Linear(in_features=4096, out_features=4096),
                nn.LeakyReLU(negative_slope=self.relu_leak),
                nn.Dropout(p=self.dropout_prob)
            )

            self.fc2 = nn.Sequential(
                nn.Linear(in_features=4096, out_features=4096),
                nn.LeakyReLU(negative_slope=self.relu_leak),
                nn.Dropout(p=self.dropout_prob)
            )

            self.fc3 = nn.Linear(in_features=4096, out_features=3)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.conv10(x)
            x = self.conv11(x)
            x = self.conv12(x)
            x = self.conv13(x)
            x = self.conv14(x)
            x = self.conv15(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    model_save_path = "model.pth"
    cnn_model = CNN()
    cnn_model.load_state_dict(torch.load(model_save_path))
    cnn_model.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = cnn_model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 900 validation images: {100 * correct // total} %')



    # prepare to count predictions for each class
    classes = test_dataset.classes
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    results = pd.DataFrame(columns=["true", "pred"])

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = cnn_model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                results.loc[len(results)] = [str(label), str(prediction)]

                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')



