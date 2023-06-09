import os
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

os.listdir('./meanstd/')
training_dataset_path = './meanstd/'

training_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(root = training_dataset_path, transform = training_transforms)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=32, shuffle=False)

def get_mean_and_std(loader):
    mean = 0
    std = 0
    total_image_count = 0
    for images, _ in tqdm(loader):
        image_count_in_a_batch = images.size(0)
        # print(images.shape)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        # print(images.shape)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_image_count += image_count_in_a_batch

    mean /= total_image_count
    std /= total_image_count

    print("Mean", mean)
    print("Std", std)
    return mean, std

get_mean_and_std(train_loader)