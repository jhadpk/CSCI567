import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
    - w0: initial weight vector (a numpy array)
    - b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    # one = np.ones((N, 1))
    updated_X = np.insert(X, 0, 1, axis=1) # inserting 1 as first dimension to X
    updated_W = np.insert(w, 0, b, axis=0) # inserting b to W
    y = np.where(y == 0, -1, 1)

    if loss == "perceptron":
        for i in range(max_iterations + 1):
            y_pred = binary_predict(updated_X, updated_W, 10101010)  # XTw
            y_pred = np.where(y_pred == 0, -1, 1)
            y_wt_x = (y) * y_pred
            y_wt_x = np.where(y_wt_x <= 0, 1, 0)  # Indicator function implementation
            indicator_y = y_wt_x * y
            indicator_y_x = np.dot(indicator_y, updated_X)
            eeta_by_N = step_size / N
            sub_gradient_component = eeta_by_N * (indicator_y_x)
            updated_W = np.add(updated_W, sub_gradient_component)

    elif loss == "logistic":
        for i in range(max_iterations + 1):
            y_wt_x = y * np.dot(updated_X, updated_W)
            sigma_of_y_wt_x = sigmoid(-y_wt_x)
            sigma_of_y_wt_x_y = sigma_of_y_wt_x * y
            sigma_of_y_wt_x_y_x = np.dot(sigma_of_y_wt_x_y, updated_X)
            eeta_by_N = step_size / N
            updated_W = updated_W + (eeta_by_N) * sigma_of_y_wt_x_y_x

    else:
        raise "Undefined loss function."

    b = updated_W[0]
    w = np.delete(updated_W, 0)
    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """
    value = 1 / (1 + np.exp(-z))
    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape
    if b != 10101010:
        X = np.insert(X, 0, 1, axis=1)
        w = np.insert(w, 0, b, axis=0)
    z = np.dot(X, w)
    preds = np.sign(z)
    preds = np.where(preds == -1, 0, 1)
    assert preds.shape == (N,)
    return preds


def softmax(x):
    numerator = np.exp(x - np.amax(x))
    denominator = np.sum(numerator, axis=1)
    return (numerator.transpose() / denominator).transpose()


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes

    Implement multinomial logistic regression for multiclass
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.

    You may find it useful to use a special (one-hot) representation of the labels,
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION

    one = np.ones((N, 1))
    updated_X = np.append(X, one, axis=1)
    updated_w = np.append(w, np.array([b]).T, axis=1)

    if gd_type == "sgd":
        for it in range(max_iterations):
            n = np.random.choice(N)
            w_Xt = np.matmul(updated_w, updated_X[n].T)
            numerator = np.subtract(w_Xt, np.amax(w_Xt, axis=0))
            numerator = [np.exp(item) for item in numerator]
            denominator = np.sum(numerator)
            prob = [item / denominator for item in numerator]
            prob[y[n]] -= 1
            sgd_component = np.matmul(np.array([prob]).T, np.array([updated_X[n]]))
            updated_w = updated_w - (step_size) * sgd_component

    elif gd_type == "gd":
        y = np.eye(C)[y]  # taking the yth row of identity matrix
        for i in range(max_iterations):
            X_wt = updated_X.dot(updated_w.transpose())
            numerator = np.exp(X_wt - np.amax(X_wt))
            denominator = np.sum(numerator, axis=1)
            z = (numerator.transpose() / denominator).transpose()
            z = z - y
            gd_component = np.dot(z.transpose(), updated_X)
            updated_w = updated_w - (step_size / N) * gd_component

    else:
        raise "Undefined algorithm."

    b = updated_w[:, -1]
    w = np.delete(updated_w, -1, 1)
    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape

    updated_w = np.insert(w, 0, b, axis=1)
    updated_X = np.insert(X, 0, 1, axis=1)
    w_transpose_X = np.dot(updated_X, updated_w.transpose())
    preds = np.argmax(w_transpose_X, axis=1)

    assert preds.shape == (N,)
    return preds
