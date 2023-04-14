import unittest
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from app.pipeline.dataset import FruitWrapedDataset, DeviceDataLoader
from app.models.fruit_model import Fruit360CnnModel
from app.utils.my_utils import get_default_device
from app.pipeline.evaluate import FruitModelTrainer

class TestModelTrainer(unittest.TestCase):
    # python3 -m unittest app.tests.test_pipeline.TestModelTrainer.test_pipeline
    def test_pipeline(self):
        print("Starting ===== test_pipeline ===== ")

        # Load dataset
        dataset = FruitWrapedDataset(training_folder="app/dataset/fruits-360/Training",
                                     testing_folder="app/dataset/fruits-360/Test")


        
        # Splitting the dataset
        torch.manual_seed(43)
        val_size = round(len(dataset.train_dataset) * 0.2)
        train_size = round(len(dataset.train_dataset) - val_size)
        train_ds, val_ds = random_split(dataset.train_dataset, [train_size, val_size])
        len(train_ds), len(val_ds)
        
        # Define batch size
        batch_size=600

        # Dataloader
        train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
        test_loader = DataLoader(dataset.test_dataset, batch_size*2, num_workers=4, pin_memory=True)


        device = get_default_device()
        print(device)


        train_dl = DeviceDataLoader(train_loader, device)
        valid_dl = DeviceDataLoader(val_loader, device)

        # Training model
        epochs = 1
        max_lr = 0.01
        grad_clip = 0.1
        weight_decay = 1e-4        
        model = Fruit360CnnModel()
        opt_func = torch.optim.Adam(model.parameters(), lr=max_lr)
        
        trainer = FruitModelTrainer(model=model, 
                                    optimizer=opt_func, 
                                    criterion=F.cross_entropy)

        trainer.train(train_loader=train_dl, 
                      val_loader=valid_dl, 
                      epochs=epochs)
        # print(os.getcwd())
        # print(model)
        print("----------------------")
    
        