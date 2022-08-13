#importing modules
import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

class Preprocessing():
    def __init__(self, dataset_folder_path:str) -> None:
        self.DATASET_FOLDER_PATH = dataset_folder_path


    def load_dataset(self) -> tuple(list[str], list[str]):
        """
        Loads the dataset from the given path in a sorted manner.
        """
        files_orig_image = sorted(filter(os.path.isfile, glob.glob(self.DATASET_FOLDER_PATH + "/org data/*.png")))
        files_ground_truth = sorted(filter(os.path.isfile, glob.glob(self.DATASET_FOLDER_PATH + "/GT/*.png")))
        return files_orig_image, files_ground_truth
    
    def convert_images_to_array(self, files_orig_image, files_ground_truth) -> tuple(np.ndarray, np.ndarray):
        """
        Converts the images to arrays.
        """
        self.X_train = np.array([np.array(Image.open(fname)) for fname in files_orig_image])
        self.Y_train = np.array([np.array(Image.open(fname)) for fname in files_ground_truth])
        return self.X_train, self.Y_train
    
    def train_test_split_image(self, X_train, Y_train, test_size) -> tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Splits the dataset into train and test set.
        """
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_train, Y_train, test_size=test_size)
        return self.X_train, self.X_test, self.Y_train, self.Y_test
