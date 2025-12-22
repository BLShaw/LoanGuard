import unittest
import pandas as pd
import os
from src.data_loader import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data_failure(self):
        with self.assertRaises(FileNotFoundError):
            load_data("non_existent_file.csv")

if __name__ == '__main__':
    unittest.main()
