import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class BarkDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for root, _, files in os.walk(data_dir) for fname in files if fname.endswith('.png')]

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

if __name__=="__main__":
    # Path to the main data folder
    data_folder = './data/'

    # Dictionary to store the count of images for each species
    species_count = {}

    # Loop through each sub-folder in the main data directory
    for species in os.listdir(data_folder):
        species_folder = os.path.join(data_folder, species)
        
        # Ensure we're looking at a directory (sub-folder)
        if os.path.isdir(species_folder)  and (not species=="tree_unknown"):
            # Count the number of files (images) in the species folder
            num_images = len([fname for fname in os.listdir(species_folder) if fname.endswith('.png')])
            
            # Store the count in the dictionary
            species_count[species] = num_images
    
    print(species_count)
    
    # Labels (species names) and sizes (number of images)
    labels = list(species_count.keys())
    sizes = list(species_count.values())

    # Create a pie chart
    plt.figure(figsize=(8, 8))  # Adjust figure size as needed
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)

    # Add a title
    plt.title('Distribution of Images by Tree Species')

    # Show the pie chart
    plt.savefig('dataDistribution.png')
