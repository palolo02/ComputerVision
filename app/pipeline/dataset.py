import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import numpy as np
from torchvision.transforms import transforms
from app.utils.my_utils import to_device

class FruitWrapedDataset:
    """ Default Fruit dataset for classification model with data augmentation"""
    def __init__(self, training_folder, testing_folder) -> None:
        self.train_dataset = ImageFolder(training_folder, 
                                         transform=T.Compose([
                                            T.RandomCrop(100, padding=4, padding_mode='reflect'), 
                                            T.Resize((100,100)),
                                            T.RandomHorizontalFlip(), 
                                            # T.RandomRotate
                                            # T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                                            # T.ColorJiTer(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                            T.ToTensor(), ])
                                         )
        self.test_dataset = ImageFolder(testing_folder, transform=T.Compose([T.ToTensor(), T.Resize((100,100)),]))
        

class FruitDataset(Dataset):
    """Fruit dataset for classification model"""

    def __init__(self, folder) -> None:        
        self.folder = folder                
        self.images = []
        self.masks = []        
        self.transform = T.Compose([T.ToTensor()])
    
    def __load_images__(self):
        self.images = [f"{self.folder}{img}" for img in os.listdir(self.folder)]
        print(self.images)


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """ Returns a tuple of image and mask within the dataset """        
        img_path = self.images[index]
        mask_path = self.masks[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
    
        image = self.transform(image)
        return (image, mask)
       
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
   

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
