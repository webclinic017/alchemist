import os
import unittest
from alchemist.data.data_utils import *


class TestDataStorage(unittest.TestCase):

    def test_load_save_data(self):
        example_data = ["never", "gonna", "give", "you", "up"]
        # Test saving
        path = "cache/tests/test_saving_data"
        if os.path.isfile(path):
            os.remove(path)
        self.assertFalse(os.path.isfile(path))
        save_data(example_data, path)
        loaded_data = load_data(path)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(loaded_data, example_data)

    def test_data_backup(self):
        example_data = ("never", "gonna", "let", "you", "down")
        path = "cache/tests/test_data_backup"
        save_data(example_data, path, backup = True)
        os.remove(path)
        loaded_data = load_data(path)
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(os.path.isfile(path + "_backup"))
        self.assertEqual(loaded_data, example_data)


if __name__ == '__main__':
    unittest.main()
