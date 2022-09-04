# importing modules
import warnings
from matplotlib import pyplot as plt
from Preprocessing import Dataset_Processing, Plotting, Image_Processing
from Model import Fcn_Network
from Evaluation import Evaluation
import numpy as np

warnings.filterwarnings("ignore")

def Main():
    """
    Main function to start program
    """
    # initializing the dataset processing class
    print("Initializing Dataset Processing Class")
    dataset_processing = Dataset_Processing(dataset_folder_path="dataset")

    # loading the dataset
    print("Loading the dataset")
    files_orig_image, files_ground_truth = dataset_processing.load_dataset()

    
    # converting the images to arrays
    print("Converting images to arrays")
    X_train, Y_train = dataset_processing.convert_images_to_array(
        files_orig_image, files_ground_truth
    )
    
    # splitting the dataset into train and test set
    print("splitting Dataset into train and test")
    x_train, x_test, y_train, y_test = dataset_processing.train_test_split_image(
        X_train, Y_train, test_size=0.2
    )

    print("plottting images")
    # plotting origial images
    plotting = Plotting()
    plotting.plot_images(X_train, no_of_images=8, cmap="", title="Original Images")

    # plotting ground truth images
    plotting.plot_images(Y_train, no_of_images=8, cmap=plt.cm.binary_r, title="Ground Truth")

    # image processsing techniques and transformations
    # image augmentation
    print("Performing Augmentation for Training Sets")
    image_processing = Image_Processing(x_train, y_train)
    (
        x_rotated,
        y_rotated,
        # x_random_noise,
        # y_random_noise,
        x_flipped_h,
        y_flipped_h,
        x_flipped_v,
        y_flipped_v,
        # x_translated,
        # y_translated,
        x_zoomed,
        y_zoomed,
        # x_sheared,
        # y_sheared,
        x_brightened,
        y_brightened,
    ) = image_processing.augment_image(x_train, y_train)
    print("Finished Augmentation for training sets")
    print("Performing Augmentation for Testing Sets")

    (
        x_rotated_test,
        y_rotated_test,
        # x_random_noise_test,
        # y_random_noise_test,
        x_flipped_h_test,
        y_flipped_h_test,
        x_flipped_v_test,
        y_flipped_v_test,
        # x_translated_test,
        # y_translated_test,
        x_zoomed_test,
        y_zoomed_test,
        # x_sheared_test,
        # y_sheared_test,
        x_brightened_test,
        y_brightened_test,
    ) = image_processing.augment_image(x_test, y_test)

    # plotting augmented images with their original images
    plotting.plot_augmented_images(
        x_rotated = x_rotated,
        y_rotated = y_rotated,
        # x_random_noise,
        # y_random_noise,
        x_flipped_h = x_flipped_h,
        y_flipped_h = y_flipped_h,
        x_flipped_v = x_flipped_v,
        y_flipped_v = y_flipped_v,
        # x_translated,
        # y_translated,
        x_zoomed = x_zoomed,
        y_zoomed = y_zoomed,
        # x_sheared,
        # y_sheared,
        x_brightened = x_brightened,
        y_brightened = y_brightened,
        image_number=40,
    )

    # joining all the transformations image arrays to the original training arrays for training set
    x_train_full = np.concatenate(
        [
            x_train,
            x_rotated,
            # x_random_noise,
            x_flipped_h,
            x_flipped_v,
            # x_translated,
            x_zoomed,
            # x_sheared,
            x_brightened,
        ]
    )

    y_train_full = np.concatenate(
        [
            y_train,
            y_rotated,
            # y_random_noise,
            y_flipped_h,
            y_flipped_v,
            # y_translated,
            y_zoomed,
            # y_sheared,
            y_brightened,
        ]
    )

    # making a validation set from the training set
    x_train, x_val, y_train, y_val = dataset_processing.train_test_split_validation(
        x_train_full, y_train_full, test_size=0.20, random_state=101
    )

    #check descriptives of training,testing and validation sets
    dataset_processing.descriptives(x_train, x_test, x_val)

    # defining loss and metrics to use for model training
    loss = [Evaluation.jaccard_distance]
    metrics = [Evaluation.dice_coefficient,Evaluation.precision,Evaluation.recall,Evaluation.accuracy]
    
    # running model network on the training set
    model_network = Fcn_Network(epochs=100, model_save_name="fcn_100_epochs_weights.h5",lr=0.003)
    model,hist = model_network.train_model(x_train, y_train, x_val, y_val, loss=loss, metrics=metrics)
if __name__ == "__main__":
    Main()
