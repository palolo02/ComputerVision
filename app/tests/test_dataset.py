import os
import unittest
from app.pipeline.dataset import FruitWrapedDataset


class TestDatasets(unittest.TestCase):

     # python3 -m unittest app.tests.test_dataset.TestDatasets.test_wraped_fruit_dataset
    def test_wraped_fruit_dataset(self):
        print("Starting ===== test_wraped_fruit_dataset ===== ")     
        dataset = FruitWrapedDataset(training_folder="app/dataset/fruits-360/Training",
                                     testing_folder="app/dataset/fruits-360/Test")
                
        print(f"{len(dataset.train_dataset)}")
        print(f"{len(dataset.test_dataset)}")        
        print(dataset.train_dataset)
        print(dataset.test_dataset)
        print("----------------------")

    # python3 -m unittest app.tests.test_dataset.TestDatasets.test_dataset_classes
    def test_dataset_classes(self):
        print("Starting ===== test_dataset_classes ===== ")    
                
        classes = os.listdir("app/dataset/fruits-360/Training")
        print(f'Total Number of Classes in Training: {len(classes)}')
        print(f'Classes Names: {classes}')

        classes = os.listdir("app/dataset/fruits-360/Test")
        print(f'Total Number of Classes in Test: {len(classes)}')
        print(f'Classes Names: {classes}')
        print("----------------------")

    # python3 -m unittest app.tests.test_dataset.TestDatasets.test_shape_img
    def test_shape_img(self):
        print("Starting ===== test_shape_img ===== ")   
        dataset = FruitWrapedDataset(training_folder="app/dataset/fruits-360/Training",
                                     testing_folder="app/dataset/fruits-360/Test")
        # Q: What is the shape of an image tensor from the dataset ?
        img, label = dataset.train_dataset[0]
        img_shape = img.shape
        print(img_shape)
        print("----------------------")

