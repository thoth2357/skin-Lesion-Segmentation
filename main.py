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
    

if __name__ == '__main__':
    Main()