import numpy as np
from knn import KNN


############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    real_label_sum = 0
    predicted_label_sum = 0
    agg = 0
    for real_value, predicted_value in zip(real_labels, predicted_labels):
        agg = agg + real_value * predicted_value
        real_label_sum = real_label_sum + real_value
        predicted_label_sum = predicted_label_sum + predicted_value
    return 2 * (float(agg) / float(real_label_sum + predicted_label_sum))


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        cubed_sum = 0
        for x, y in zip(point1, point2):
            cubed_sum = cubed_sum + np.absolute(x - y) ** 3
        return float(np.cbrt(cubed_sum))

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        squared_sum = 0
        for x, y in zip(point1, point2):
            squared_sum = squared_sum + (x - y) ** 2
        return float(np.sqrt(squared_sum))

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        norm_point1 = 0
        norm_point2 = 0
        product_sum = 0
        for x, y in zip(point1, point2):
            norm_point1 = norm_point1 + x ** 2
            norm_point2 = norm_point2 + y ** 2
            product_sum = product_sum + x * y
        if norm_point1 == 0 or norm_point2 == 0:
            return 1
        return float(1 - float(product_sum) / float(np.sqrt(norm_point1) * np.sqrt(norm_point2)))


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist
        (this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        best_f1_score = 0
        for distance_func in distance_funcs:
            for k in range(1, min(31, len(x_train) + 1), 2):
                model = KNN(k, distance_funcs[distance_func])
                model.train(x_train, y_train)
                y_val_pred = model.predict(x_val)
                calculated_f1_score = f1_score(y_val, y_val_pred)
                if calculated_f1_score > best_f1_score:
                    self.best_k = k
                    self.best_distance_function = distance_func
                    self.best_model = model
                    best_f1_score = calculated_f1_score

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them.
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        best_f1_score = 0
        for scaling_class in scaling_classes:
            scalar = scaling_classes[scaling_class]()
            scaled_x_train = scalar.__call__(x_train)
            scaled_x_val = scalar.__call__(x_val)
            for distance_func in distance_funcs:
                for k in range(1, min(31, len(x_train) + 1), 2):
                    model = KNN(k, distance_funcs[distance_func])
                    model.train(scaled_x_train, y_train)
                    y_val_pred = model.predict(scaled_x_val)
                    calculated_f1_score = f1_score(y_val, y_val_pred)
                    if calculated_f1_score > best_f1_score:
                        self.best_k = k
                        self.best_distance_function = distance_func
                        self.best_scaler = scaling_class
                        self.best_model = model
                        best_f1_score = calculated_f1_score


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        total_data = len(features)
        total_features = len(features[0])
        normalized_features = [[0] * total_features for x in range(total_data)]
        for x in range(total_data):
            norm = 0
            for y in range(total_features):
                norm += (features[x][y] ** 2)
            norm = np.sqrt(norm)
            if norm == 0:
                normalized_features[x] = features[x]
                continue
            for y in range(total_features):
                normalized_features[x][y] = 0 if features[x][y] == 0 else features[x][y] / norm
        return normalized_features


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
        For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
        This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
        The minimum value of this feature is thus min=-1, while the maximum value is max=2.
        So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
        leading to 1, 0, and 0.333333.
        If max happens to be same as min, set all new values to be zero for this feature.
        (For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        total_data = len(features)
        total_features = len(features[0])
        normalized_features = [[0] * total_features for _ in range(total_data)]
        minimum_feature_list = [float("inf")] * total_features
        maximum_feature_list = [float("-inf")] * total_features

        for y in range(total_data):
            for x in range(total_features):
                val = features[y][x]
                maximum_feature_list[x] = max(maximum_feature_list[x], val)
                minimum_feature_list[x] = min(minimum_feature_list[x], val)

        for y in range(total_data):
            for x in range(total_features):
                max_min_diff = maximum_feature_list[x] - minimum_feature_list[x]
                normalized_features[y][x] = 0 if max_min_diff == 0 else ((features[y][x] - minimum_feature_list[x]) / max_min_diff)
        return normalized_features
