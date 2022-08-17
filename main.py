# importing modules
from matplotlib import pyplot as plt
from Preprocessing import Dataset_Processing, Plotting, Image_Processing
from Model import Fcn_Network
from Evaluation import Evaluation


def Main():
    """
    Main function to start program
    """
    # initializing the dataset processing class
    dataset_processing = Dataset_Processing(dataset_folder_path="dataset")

    # loading the dataset
    files_orig_image, files_ground_truth = dataset_processing.load_dataset()

    # converting the images to arrays
    X_train, Y_train = dataset_processing.convert_images_to_array(
        files_orig_image, files_ground_truth
    )

    # splitting the dataset into train and test set
    x_train, x_test, y_train, y_test = dataset_processing.train_test_split_image(
        X_train, Y_train, test_size=0.2
    )

    # plotting origial images
    plotting = Plotting()
    plotting.plot_images(X_train, no_of_images=8, cmap="None", title=None)

    # plotting ground truth images
    plotting.plot_images(Y_train, no_of_images=8, cmap=plt.cm.binary_r, title=None)

    # image processsing techniques and transformations
    # image augmentation
    image_processing = Image_Processing(x_train, y_train)
    (
        x_rotated,
        y_rotated,
        x_random_noise,
        y_random_noise,
        x_flipped_h,
        y_flipped_h,
        x_flipped_v,
        y_flipped_v,
        x_translated,
        y_translated,
        x_zoomed,
        y_zoomed,
        x_sheared,
        y_sheared,
        x_brightened,
        y_brightened,
    ) = image_processing.augment_images(x_train, y_train)


if __name__ == "__main__":
    Main()
