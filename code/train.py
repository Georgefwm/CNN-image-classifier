import torch
from torchvision import datasets, transforms
from torchvision.io import decode_image
from torchvision.io.image import ImageReadMode
import helper

# Check if device has CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using: " + device)

# import datasets
dataset = datasets.ImageFolder("traindata")
print(dataset)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)