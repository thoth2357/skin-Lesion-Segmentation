# importing modules
import os
import glob
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class Dataset_Processing:
    """
    class to help with method to help us read,parse,convert and split the dataset for processing later
    """

    def __init__(self, dataset_folder_path: str) -> None:
        self.DATASET_FOLDER_PATH = dataset_folder_path

    def load_dataset(self):  # type: ignore
        """
        Loads the dataset from the given path in a sorted manner.
        """
        files_orig_image = sorted(
            filter(
                os.path.isfile, glob.glob(self.DATASET_FOLDER_PATH + "/org data/*.jpg")
            )
        )
        files_ground_truth = sorted(
            filter(os.path.isfile, glob.glob(self.DATASET_FOLDER_PATH + "/GT/*.png"))
        )
        return files_orig_image, files_ground_truth

    def convert_images_to_array(
        self, files_orig_image, files_ground_truth
    ):  # type: ignore
        """
        Converts the images to arrays.
        """
        self.X_train = np.array(
            [np.array(Image.open(fname)) for fname in files_orig_image],
            dtype=object,
        )
        self.Y_train = np.array(
            [np.array(Image.open(fname)) for fname in files_ground_truth],
            dtype=object,
        )
        return self.X_train, self.Y_train

    def train_test_split_image(
        self, X_train, Y_train, test_size, *args
    ):  # type: ignore
        """
        Splits the dataset into train and test set.
        """
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X_train, Y_train, test_size=test_size
        )
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def train_test_split_validation(
        self, X_train, Y_train, test_size, random_state
    ):  # type: ignore
        """
       splits the dataset provided for us to get our validation set 
        """
        x_train, x_val, y_train, y_val = train_test_split(
            X_train, Y_train, test_size=test_size, random_state=random_state
        )
        return x_train, x_val, y_train, y_val

    def descriptives(self, *args) -> None:
        """
        Prints the descriptives of the dataset.
        """
        for arg in args:
            print("Length of Training Set:", len(arg))
            print("Length of Test Set:", len(arg))
            print("length of Validation Set:", len(arg))
            print("----"*10)
            print("Shape of Training Set:", arg.shape)
            print("Shape of Test Set:", arg.shape)
class Plotting:
    """
    class to help with methods to help plot our dataset images and evaluation metrics
    """

    def __init__(self, figsize=(20, 10)) -> None:
        self.FIGSIZE = figsize

    def plot_images(self, dataset, no_of_images=None, cmap=None, title=None) -> None:
        """
        plots random images based on image number and type
        """
        if cmap is None:
            cmap = ""
        plt.figure(figsize=self.FIGSIZE)
        for i in range(no_of_images):  # type: ignore
            plt.subplot(2, no_of_images / 2, i + 1)
            plt.imshow(dataset[i], cmap=cmap)
            if title != None:
                plt.title(title)
            plt.axis("off")
        plt.show()

    def plot_augmented_images(self, *args, **kwargs) -> None:
        """
        plots the augmented images
        """
        image_number = kwargs.get("image_number")
        plt.figure(figsize=(12, 12))
        for no, image in enumerate(args):
            plt.subplot(3, 2, no + 1)
            if image.split("_")[0] == "y":
                plt.imshow(image[image_number], cmap=plt.cm.binary_r)  # type: ignore # type: ignore
                plt.title(f"{image.split('_')[1]} Mask")
            else:
                plt.imshow(image[image_number])
                plt.title(f"{image.split('_')[1]} Image")


class Image_Processing:
    """
    class to help with method for processing our image by either augmenting,enhancing or doing any other processing
    """

    def __init__(self, x_train, y_train) -> None:
        pass

    def horizontal_flip(self, x_image, y_image):  # type: ignore # type: ignore
        """
        horizontal flip the provided image
        """
        x_image = cv2.flip(x_image, 1)
        y_image = cv2.flip(y_image.astype("float32"), 1)
        return x_image, y_image.astype("int")

    def random_rotation(self, x_image, y_image):  # type: ignore
        """
        random rotation of the image
        """
        rows, cols = x_image.shape[:2]
        rotation_angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
        x_image = cv2.warpAffine(x_image, M, (cols, rows))
        y_image = cv2.warpAffine(y_image.astype("float32"), M, (cols, rows))
        return x_image, y_image.astype("int")

    def random_noise(self, x_image, y_image):  # type: ignore
        """
        random noise of the image
        """
        x_image = x_image + np.random.normal(0, 0.1, x_image.shape)
        y_image = y_image + np.random.normal(0, 0.1, y_image.shape)
        return x_image, y_image.astype("int")

    def vertical_flip(self, x_image, y_image):  # type: ignore
        """
        vertical flip the provided image
        """
        x_image = cv2.flip(x_image, 0)
        y_image = cv2.flip(y_image.astype("float32"), 0)
        return x_image, y_image.astype("int")

    def random_translation(self, x_image, y_image):  # type: ignore
        """
        random translation of the image
        """
        rows, cols = x_image.shape[:2]
        translation_x = np.random.uniform(-10, 10)
        translation_y = np.random.uniform(-10, 10)
        M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])  # type: ignore
        x_image = cv2.warpAffine(x_image, M, (cols, rows))
        y_image = cv2.warpAffine(y_image.astype("float32"), M, (cols, rows))
        return x_image, y_image.astype("int")

    def random_zoom(self, x_image, y_image):  # type: ignore
        """
        random zoom of the image
        """
        rows, cols = x_image.shape[:2]
        zoom_x = np.random.uniform(0.9, 1.1)
        zoom_y = np.random.uniform(0.9, 1.1)
        M = np.float32([[zoom_x, 0, 0], [0, zoom_y, 0]])  # type: ignore # type: ignore
        x_image = cv2.warpAffine(x_image, M, (cols, rows))
        y_image = cv2.warpAffine(y_image.astype("float32"), M, (cols, rows))
        return x_image, y_image.astype("int")

    def random_shear(self, x_image, y_image):  # type: ignore
        """
        random shear of the image
        """
        rows, cols = x_image.shape[:2]
        shear_x = np.random.uniform(-10, 10)
        shear_y = np.random.uniform(-10, 10)
        M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])  # type: ignore
        x_image = cv2.warpAffine(x_image, M, (cols, rows))
        y_image = cv2.warpAffine(y_image.astype("float32"), M, (cols, rows))
        return x_image, y_image.astype("int")

    def random_brightness(self, x_image, y_image):  # type: ignore
        """
        random brightness of the image
        """
        x_image = x_image + np.random.normal(0, 0.1, x_image.shape)
        y_image = y_image + np.random.normal(0, 0.1, y_image.shape)
        return x_image, y_image.astype("int")

    def augment_image(
        self, x_train, y_train
    ):  # type: ignore
        """
        augment the image with random augmentation
        """
        (
            x_rotate,
            y_rotate,
            x_random_noise,
            y_random_noise,
            x_flip_h,
            y_flip_h,
            x_flip_v,
            y_flip_v,
            x_translate,
            y_translate,
            x_zoom,
            y_zoom,
            x_shear,
            y_shear,
            x_brightness,
            y_brightness,
        ) = []
        for idx, image in enumerate(x_train):
            x_image, y_image = self.random_rotation(x_train[idx], y_train[idx])
            x_rotate.append(x_image)
            y_rotate.append(y_image)
            x_image, y_image = self.horizontal_flip(x_train[idx], y_train[idx])
            x_flip_h.append(x_image)
            y_flip_h.append(y_image)
            x_image, y_image = self.vertical_flip(x_train[idx], y_train[idx])
            x_flip_v.append(x_image)
            y_flip_v.append(y_image)
            x_image, y_image = self.random_noise(x_train[idx], y_train[idx])
            x_random_noise.append(x_image)
            y_random_noise.append(y_image)
            x_image, y_image = self.random_translation(x_train[idx], y_train[idx])
            x_translate.append(x_image)
            y_translate.append(y_image)
            x_image, y_image = self.random_zoom(x_train[idx], y_train[idx])
            x_zoom.append(x_image)
            y_zoom.append(y_image)
            x_image, y_image = self.random_shear(x_train[idx], y_train[idx])
            x_shear.append(x_image)
            y_shear.append(y_image)
            x_image, y_image = self.random_brightness(x_train[idx], y_train[idx])
            x_brightness.append(x_image)
            y_brightness.append(y_image)
        return (
            np.array(x_rotate),
            np.array(y_rotate),
            np.array(x_random_noise),
            np.array(y_random_noise),
            np.array(x_flip_h),
            np.array(y_flip_h),
            np.array(x_flip_v),
            np.array(y_flip_v),
            np.array(x_translate),
            np.array(y_translate),
            np.array(x_zoom),
            np.array(y_zoom),
            np.array(x_shear),
            np.array(y_shear),
            np.array(x_brightness),
            np.array(y_brightness),
        )
