import numpy as np
from keras import backend as K


class Evaluation:
    "class for providing us with methods to evaluate our segmentation"

    def __init__(self) -> None:
        pass

    def dice_coefficient(self, ground_truth, prediction) -> float:
        """
        Calculates the dice coefficient between two images.
        """
        ground_truth = ground_truth.flatten()
        prediction = prediction.flatten()
        intersection = np.sum(ground_truth * prediction)
        return (2.0 * intersection + 1) / (
            np.sum(ground_truth) + np.sum(prediction) + 1
        )

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

    def mean_IU(self, ground_truth, prediction) -> float:  # type: ignore
        """
        Calculates the mean IU between two images.
        """
        ground_truth = ground_truth.flatten()

    def precision(self, ground_truth, prediction) -> float:
        """ Calculates the precision, a metric for multi-label classification of
            how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(ground_truth * prediction, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(prediction, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(self, ground_truth, prediction) -> float:
        """ Calculates the recall, a metric for multi-label classification of
            how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(ground_truth * prediction, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(ground_truth, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def accuracy(self, ground_truth, prediction) -> float:
        """
        Calculates the mean accurcy rate across all predictions for binary classification problems.
        """
        return K.mean(K.equal(ground_truth, K.round(prediction)))
