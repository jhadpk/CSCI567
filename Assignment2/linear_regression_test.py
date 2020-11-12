from linear_regression import linear_regression_noreg, regularized_linear_regression, tune_lambda, mean_square_error, mapping_data
from data_loader import data_processing_linear_regression
import numpy as np
import pandas as pd

filename = 'winequality-white.csv'


print("\n======== Part 1.1 and Part 1.2 ========")
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, False, 0)
w = linear_regression_noreg(Xtrain, ytrain)
print("dimensionality of the model parameter is ", w.shape, ".", sep="")
print("model parameter is ", np.array_str(w))
mse = mean_square_error(w, Xtrain, ytrain)
print("MSE on train is %.5f" % mse)
mse = mean_square_error(w, Xval, yval)
print("MSE on val is %.5f" % mse)
mse = mean_square_error(w, Xtest, ytest)
print("MSE on test is %.5f" % mse)

print("\n======== Part 1.3 ========")
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, False, 0)
w = regularized_linear_regression(Xtrain, ytrain, 0.1)
print("dimensionality of the model parameter is ", w.shape, ".", sep="")
print("model parameter is ", np.array_str(w))
mse = mean_square_error(w, Xtrain, ytrain)
print("MSE on train is %.5f" % mse)
mse = mean_square_error(w, Xval, yval)
print("MSE on val is %.5f" % mse)
mse = mean_square_error(w, Xtest, ytest)
print("MSE on test is %.5f" % mse)


print("\n======== Part 1.4========")
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, False, 0)
bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval)
print("Best Lambda =  " + str(bestlambd))
w = regularized_linear_regression(Xtrain, ytrain, bestlambd)
print("dimensionality of the model parameter is ", len(w), ".", sep="")
print("model parameter is ", np.array_str(w))
mse = mean_square_error(w, Xtrain, ytrain)
print("MSE on train is %.5f" % mse)
mse = mean_square_error(w, Xval, yval)
print("MSE on val is %.5f" % mse)
mse = mean_square_error(w, Xtest, ytest)
print("MSE on test is %.5f" % mse)


print("\n======== Part 1.5 ========")
power = 6
for i in range(2, power):
    Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, True, i)
    bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval)
    print('best lambd is ' + str(bestlambd))
    w = regularized_linear_regression(Xtrain, ytrain, bestlambd)
    print('when power = ' + str(i))
    mse = mean_square_error(w, Xtrain, ytrain)
    print("MSE on train is %.5f" % mse)
    mse = mean_square_error(w, Xval, yval)
    print("MSE on val is %.5f" % mse)
    mse = mean_square_error(w, Xtest, ytest)
    print("MSE on test is %.5f" % mse)
    print("-----------------")



