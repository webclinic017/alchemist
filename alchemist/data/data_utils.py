import os
import pickle

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
