import numpy as np

class Perceptron(object):
    """
    Perceptron classifier

    Parameters:
    eta: learning rate (float)
    n_iter: number of training passes (int)
    random_state: seed for weight initialization (int)

    Attributes:
    w_: weights after fitting (1D array [# of features])
    errors_: number of misclassified samples (int)
    """
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, Y):
        """
        Fit training data

        Parameters:
        X: training data (array [# of samples x # of features])
        Y: "correct" values (array [# of samples])

        Returns:
        self: updated weights (perceptron object)
        """
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.normal(loc = 0.0, scale = 0.01, size=(X.shape[1] + 1))

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                update = self.eta * (target - self.predict(xi))
                self.w_[0] += update
                self.w_[1:] += update * xi
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        net input from a sample
        """
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        """
        return label using classifier step function
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
