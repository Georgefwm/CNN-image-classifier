import torch
from torchvision import datasets, transforms
from torchvision.io import decode_image
from torchvision.io.image import ImageReadMode
import helper

# import datasets

dataset = datasets.ImageFolder("traindata")
print(dataset)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)