#importing modules
import os
import glob
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
class Dataset_Processing():
    """
    class to help with method to help us read,parse,convert and split the dataset for processing later
    """
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

class Plotting():
    """
    class to help with methods to help plot our dataset images and evaluation metrics
    """
    def __init__(self, figsize=(20,10)) -> None:
        self.FIGSIZE = figsize
    
    def plot_images(self, dataset, no_of_images=None,cmap=None, title="plot") -> None:
        """
        plots random images based on image number and type
        """
        if cmap is None:
            cmap = ''
        plt.figure(figsize=self.FIGSIZE)
        for i in range(no_of_images):
            plt.subplot(2, no_of_images, i+1)
            plt.imshow(dataset[i], cmap=cmap)
            # plt.title(title)
            plt.axis('off')
        plt.show()
    
class Image_Processing():
    '''
    class to help with method for processing our image by either augmenting,enhancing or doing any other processing
    '''
    def __init__(self) -> None:
        pass
    
    def horizontal_flip(self, x_image, y_image) -> np.ndarray:
        """
        horizontal flip the provided image
        """
        x_image = cv2.flip(x_image, 1)
        y_image = cv2.flip(y_image.astype('float32'), 1)
        return x_image, y_image.astype('int')
