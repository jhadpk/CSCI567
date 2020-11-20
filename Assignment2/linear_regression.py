import numpy as np
import pandas as pd


############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######

def transpose_not_needed(X, w):
    columns_X = np.size(X, 1)
    rows_w = np.size(w, 0)
    return columns_X == rows_w


def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    prediction = np.dot(X, w) if transpose_not_needed(X, w) else np.dot(X.transpose(), w)
    squared_error = np.square(np.subtract(prediction, y))
    err = np.mean(squared_error, dtype=np.float64)
    return err


###### Part 1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    Returns:
    - w: a numpy array of shape (D, )
    """
    X_transpose = X.transpose()
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_X_inverse = np.linalg.inv(X_transpose_X)
    X_transpose_y = np.dot(X_transpose, y)
    w = np.dot(X_transpose_X_inverse, X_transpose_y)
    return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
    X_transpose = X.transpose()
    X_transpose_X = np.dot(X_transpose, X)
    lambda_I = lambd * np.identity(np.size(X, 1))
    X_transpose_y = np.dot(X_transpose, y)
    w = np.dot(np.linalg.inv(np.add(X_transpose_X, lambda_I)), X_transpose_y)
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    bestlambda = None
    min_mse = float("inf")
    for power in range(-14, 1):
        lmbda = 2 ** power
        w = regularized_linear_regression(Xtrain, ytrain, lmbda)
        mse = mean_square_error(w, Xval, yval)
        if mse < min_mse:
            bestlambda = lmbda
            min_mse = mse
    return bestlambda

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    X_temp = X
    for power in range(2, p + 1):
        X = np.concatenate((X, np.power(X_temp, power)), axis=1)
    return X


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""
