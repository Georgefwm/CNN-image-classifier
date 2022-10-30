import torch
from torchvision import datasets, transforms, utils
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from torch.utils.data import random_split
import time

# First we define some useful functions
def get_img_mean_std(loader) -> float:
    """
    Generates an approximation of the mean and std deviation for a given dataset of images.
    Approximations are much, much faster to calculate than exact values and shouldn't be too far off.

    :param: loader dataset to have calculated
    :return: mean, std deviation
    """

    mean = 0.
    std = 0.
    total_image_count = 0

    for imgs, _ in loader:
        batch_img_count = imgs.size(0)
        imgs = imgs.view(batch_img_count, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        total_image_count += batch_img_count

    mean /= total_image_count
    std /= total_image_count

    return mean, std


def main():
    # Check if device has CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using: " + str(device))

    # define the base transformation to calculate mean and std deviation of the training set
    initial_train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    # set batch size
    desired_batch_size = 16

    # import dataset
    original_train_dataset = datasets.ImageFolder("traindata", transform=initial_train_transform)

    # create loader for generating mean/std deviation
    full_train_loader = torch.utils.data.DataLoader(original_train_dataset, batch_size=desired_batch_size, shuffle=False, num_workers=2)

    # show what classes have been identified
    classes = original_train_dataset.classes
    print("Classes: " + str(classes))

    print("Calculating mean and std in training set")
    train_mean, train_std = get_img_mean_std(full_train_loader)
    print("Done \nUsing: mean: " + str(train_mean) + ", std: " + str(train_std))

    # define train and validation transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(size=(150, 150), scale=(0.4, 0.9)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])

    # used for validation dataset as well
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # set new transform
    train_dataset = datasets.ImageFolder("traindata", transform=train_transform)

    # create validation set
    val_split_ratio = 0.2
    train_size = round(len(original_train_dataset) * (1 - val_split_ratio))
    val_size = round(len(original_train_dataset) * val_split_ratio)

    train_set, val_set = random_split(original_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(7))

    # create loaders for train and val sets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=desired_batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=desired_batch_size, shuffle=False, num_workers=2)

    # takes 'inspiration' from vgg architecture

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

    cnn_model = CNN()
    cnn_model.to(device)

    # define the loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.8)

    # set file name for model
    model_save_path = "model.pth"

    # iterate over the dataset
    best_val_loss = np.inf
    epoch_used = 0
    train_time_start = time.time()  # seconds

    # stop after this many epochs with no validation improvement
    early_stop_count = 5
    epochs_with_no_improv = 0

    cnn_train_history = pd.DataFrame(columns=["train_loss", "train_accuracy", "val_loss", "val_accuracy"])

    for epoch in range(40):
        # train the model
        cnn_model.train()

        train_accuracy = 0.0
        train_loss = 0.0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            train_accuracy += int(torch.sum(prediction == labels.data))

        train_accuracy = train_accuracy / total
        train_loss = train_loss / total

        # evaluate the model
        cnn_model.eval()

        val_accuracy = 0.0
        val_loss = 0.0
        total = 0

        for i, data in enumerate(val_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Forward Pass
            outputs = cnn_model(inputs)
            # Find the Loss
            loss = criterion(outputs, labels)
            # Calculate Loss
            val_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            val_accuracy += int(torch.sum(prediction == labels.data))

        val_accuracy = val_accuracy / total
        val_loss = val_loss / total

        print("Epoch: " + str(epoch+1) + "\n" +
              "Train loss: " + str(train_loss) + ", Train accuracy: " + str(train_accuracy) + "\n" +
              "Val loss: " + str(val_loss) + ", Val Accuracy: " + str(val_accuracy) + "\n")

        cnn_train_history.loc[len(cnn_train_history)] = [train_loss, train_accuracy, val_loss, val_accuracy]

        if val_loss < best_val_loss:
            print("Validation accuracy improved: " + str(best_val_loss) + " --> " + str(val_loss))
            print("Saving model \n")
            epoch_used = epoch
            best_val_loss = val_loss
            torch.save(cnn_model.state_dict(), model_save_path)
            epochs_with_no_improv = 0
        else:
            epochs_with_no_improv += 1

        # only stop if has been no improvement and the accuracies have deviated
        if epochs_with_no_improv >= early_stop_count:
            print("Stopping early, no validation loss improvement in " + str(early_stop_count) + " epochs \n")
            break

    train_time_end = time.time()
    train_time = (train_time_end - train_time_start) / 60

    print("Finished Training")
    print("Took " + str(round(train_time, 2)) + " minutes\n")

    # Generate Loss over epochs graph
    x = range(len(cnn_train_history))

    # plot params
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.rcParams['figure.dpi'] = 100
    plt.grid(visible=None)

    plt.plot(x, cnn_train_history["train_loss"].to_numpy(), label="Train")
    plt.plot(x, cnn_train_history["val_loss"].to_numpy(), label="Validation")
    plt.axvline(x=epoch_used, label="Selected point", color="g", linestyle="--")

    plt.title("CNN Loss", fontsize=18)
    plt.xlabel("Epochs", fontsize=14)
    plt.legend()

    plt.savefig("CNN Loss.png")
    print("Loss graph generated and saved as 'CNN Loss.png'")

    # Generate Accuracy over epochs graph
    x = range(len(cnn_train_history))

    # plot params
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.rcParams['figure.dpi'] = 100
    plt.grid(visible=None)

    plt.plot(x, cnn_train_history["train_accuracy"].to_numpy(), label="Train")
    plt.plot(x, cnn_train_history["val_accuracy"].to_numpy(), label="Validation")
    plt.axvline(x=epoch_used, label="Selected point", color="g", linestyle="--")

    plt.title("CNN Accuracy", fontsize=18)
    plt.xlabel("Epochs", fontsize=14)
    plt.legend()

    plt.savefig("CNN Accuracy.png")
    print("Accuracy graph generated and saved as 'CNN Accuracy.png'\n")

    print("Training script finished, ready for testing")
    return


if __name__ == "__main__":
    main()
