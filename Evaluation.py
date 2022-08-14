import numpy as np
class Evaluation():
    'class for providing us with methods to evaluate our segmentation'
    def __init__(self) -> None:
        pass
    def dice_coefficient(self, ground_truth, prediction) -> float:
        """
        Calculates the dice coefficient between two images.
        """
        ground_truth = ground_truth.flatten()
        prediction = prediction.flatten()
        intersection = np.sum(ground_truth * prediction)
        return (2. * intersection + 1) / (np.sum(ground_truth) + np.sum(prediction) + 1)
    def jaccard_index(self, ground_truth, prediction) -> float:
        """
        Calculates the jaccard index between two images.
        """
        ground_truth = ground_truth.flatten()
        prediction = prediction.flatten()
        intersection = np.sum(ground_truth * prediction)
        return intersection / (np.sum(ground_truth) + np.sum(prediction) - intersection)
    def mean_accuracy(self, ground_truth, prediction) -> float:
        """
        Calculates the mean accuracy between two images.
        """
        ground_truth = ground_truth.flatten()
        prediction = prediction.flatten()
        return np.sum(ground_truth == prediction) / len(ground_truth)
    def mean_accuracy_per_class(self, ground_truth, prediction) -> float:
        """
        Calculates the mean accuracy per class between two images.
        """
        ground_truth = ground_truth.flatten()
        prediction = prediction.flatten()
        return np.sum(ground_truth == prediction) / len(ground_truth)
    def mean_IU(self, ground_truth, prediction) -> float:
        """
        Calculates the mean IU between two images.
        """
        ground_truth = ground_truth.flatten()