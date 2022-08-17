#importing modules
from Preprocessing import Dataset_Processing, Plotting, Image_Processing
from Model import Fcn_Network
from Evaluation import Evaluation

def Main():
    """
    Main function to start program
    """
    #initializing the dataset processing class
    dataset_processing = Dataset_Processing(dataset_folder_path="dataset")
    
    #loading the dataset
    files_orig_image, files_ground_truth = dataset_processing.load_dataset()

    #converting the images to arrays
    X_train, Y_train = dataset_processing.convert_images_to_array(files_orig_image, files_ground_truth)

    #splitting the dataset into train and test set
    x_train, x_test, y_train, y_test = dataset_processing.train_test_split_image(X_train, Y_train, test_size=0.2)
if __name__ == '__main__':
    Main()