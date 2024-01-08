import numpy as np


def softmax(scores):
    """
    Computes the numerically stable softmax, checkout: https://stackoverflow.com/questions/42599498/numerically-stable-softmax

    """
    scores = (scores.T - np.max(scores, axis=1)).T
    return np.exp(scores) / np.array(np.sum(np.exp(scores), axis=1))[:, None]


def cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)=-inf when initial coefficients are huge
    loss = -np.sum(y_true * np.log(y_pred + epsilon))
    return loss / float(y_pred.shape[0])


def ce_loss_value(Z, rho):
    """
    computes the value of the ce loss after computing the softmax value of the scores

    Parameters
    ----------
    Z           tuple of numpy.arrays containing training data with shape = ((n_rows, n_cols), (n_rows, n_classes))
    rho         numpy.array of coefficients with shape = (n_cols*n_classes,)

    Returns
    -------
    loss_value  scalar = -1/n_rows * sum(Y*log( softmax(X*rho))

    """
    X, Y = Z
    rho = rho.reshape((X.shape[-1], Y.shape[-1]))
    scores = X.dot(rho)
    probs = softmax(scores)
    loss_value = cross_entropy(Y, probs)
    return loss_value


def ce_loss_value_and_slope(Z, rho):
    """
    computes the value and slope of the ce loss
    this function should only be used when generating cuts in cutting-plane algorithms
    (computing both the value and the slope at the same time is slightly cheaper)
    see: https://www.michaelpiseno.com/blog/2021/softmax-gradient/

        Parameters
    ----------
    Z           tuple of numpy.arrays containing training data with shape = ((n_rows, n_cols), (n_rows, n_classes))
    rho         numpy.array of coefficients with shape = (n_cols*n_classes,)

    Returns
    -------
    loss_value  scalar = -1/n_rows * sum(Y*log( softmax(X*rho))
    loss_slope: (n_cols * n_classes x 1) vector = 1/n_rows * sum(-Z*rho ./ (1+exp(-Z*rho))

    """
    X, Y = Z
    rho = rho.reshape((X.shape[-1], Y.shape[-1]))
    scores = X.dot(rho)

    probs = softmax(scores)

    loss_value = cross_entropy(Y, probs)
    dScores = probs - Y
    loss_slope = (X.T.dot(dScores)) / float(
        Y.shape[0])
    loss_slope = loss_slope.reshape(-1)
    return loss_value, loss_slope


def ce_loss_value_from_scores(Z, scores):
    """
    computes the logistic loss value from a vector of scores in a numerically stable way
    where scores = Z.dot(rho)

    see also: http://stackoverflow.com/questions/20085768/

    this function is used for heuristics (discrete_descent, sequential_rounding).
    to save computation when running the heuristics, we store the scores and
    call this function to compute the loss directly from the scores
    this reduces the need to recompute the dot product.

    Parameters
    ----------
    scores  numpy.array of scores = Z.dot(rho)

    Returns
    -------
    loss_value  scalar = 1/n_rows * sum(log( 1 .+ exp(-Z*rho))

    """
    _, Y = Z
    probs = softmax(scores)
    loss_value = cross_entropy(Y, probs)
    return loss_value
