import unittest
import os
from app.models.fruit_model import Fruit360CnnModel



class TestFruitModel(unittest.TestCase):
    
    # python3 -m unittest app.tests.test_model.TestFruitModel.test_fruit_model
    def test_fruit_model(self):
        print("Starting ===== test_fruit_model ===== ")    
        model = Fruit360CnnModel()
        print(os.getcwd())
        print(model)
        print("----------------------")
    
