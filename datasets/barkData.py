import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class BarkDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = int(img_path.split('_')[-1].split('.')[0])  # Assuming file names are like: image_0.jpg, image_1.jpg, etc.

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloader(data_dir, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    dataset = BarkDataset(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
