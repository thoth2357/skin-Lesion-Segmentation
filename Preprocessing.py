#importing modules
import os
import glob

class Preprocessing():
    def __init__(self, dataset_folder_path:str) -> None:
        self.DATASET_FOLDER_PATH = dataset_folder_path


    def load_dataset(self):
        """
        Loads the dataset from the given path in a sorted manner.
        """
        files_orig_image = sorted(filter(os.path.isfile, glob.glob(self.DATASET_FOLDER_PATH + "/org data/*.png")))
        files_ground_truth = sorted(filter(os.path.isfile, glob.glob(self.DATASET_FOLDER_PATH + "/GT/*.png")))
        
        
