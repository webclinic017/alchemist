import os
import pickle
import numpy as np
from torch.utils.data import Dataset

def save_data(data, fname, backup = False):
    # print("Storing data at " + fname)
    # Make sure that the folders the file is meant to be in exist
    path = fname[:fname.rindex("/")]
    if not os.path.exists(path):
        # print("Path " + path + " does not exist, creating necessary folders...")
        os.makedirs(path)
    # Then save the file
    with open(fname, "wb") as f:
        pickle.dump(data, f)
    #If needed, also save a backup
    if backup:
        # print("Creating backup...")
        fname += "_backup"
        with open(fname, "wb") as f:
            pickle.dump(data, f)
    # print("Data successfully stored")

def load_data(fname, save_backup = True):
    try:
        with open(fname, "rb") as f:
            data = pickle.load(f)
    # If the file is corrupted, attempt to load a backup
    except (pickle.UnpicklingError, AssertionError, EOFError,
            UnicodeDecodeError, FileNotFoundError):
        # print("Warning: data at " + fname + " corrupted or missing,",
        #       "loading backup...")
        old_fname = fname
        fname = fname + "_backup"
        with open(fname, "rb") as f:
            data = pickle.load(f)
        # save_backup in this case means replace currupted data with backup
        if save_backup:
            # print("Backup loaded successfully, storing in place of original...")
            save_data(data, old_fname, backup = False)

    return data

class Dataset (Dataset):
    # Dataset interacts correctly with data_loader
    def __init__(self, x_data, y_data):
        # Set parameters and things
        self.x_data = x_data
        self.y_data = y_data
        self.length = len(self.x_data)

    def __getitem__(self, index):
        return np.array([self.x_data[index]]), self.y_data[index]

    def __len__(self):
        return self.length
