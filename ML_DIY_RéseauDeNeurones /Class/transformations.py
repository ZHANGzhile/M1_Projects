import numpy as np
from module import Module


class TanH(Module):
    def __init__(self):
        super().__init__()
        self.name = "tanh"

    def forward(self, data):
        return np.tanh(data)

    def backward_delta(self, data, delta):
        return delta * (1 - self.forward(data) ** 2)

    def zero_grad(self):
        pass

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_update_gradient(self, data, delta):
        pass

    def predict(self, X):
        pass


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.name = "sigmoid"

    def forward(self, data):
        return 1 / (1 + np.exp(-data))

    def backward_delta(self, data, delta):
        return delta * (1 - self.forward(data)) * self.forward(data)

    def zero_grad(self):
        pass

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_update_gradient(self, data, delta):
        pass

    def predict(self, X):
        raise NotImplementedError


class SoftMax(Module):
    def __init__(self):
        super().__init__()

    def update_parameters(self, gradient_step: float = 0.001):
        # pas de parametres donc on fait rien
        return None

    def backward_update_gradient(self, X, delta):
        # Softmax n'a pas de parametres donc pas besoin de faire ca.
        pass

    def backward_delta(self, X, delta):
        """Calcul la derivee de l'erreur par rapport aux inputs

        :param X: batch x layer_size
        :param delta: batch x layer_size
        :return: batch x layer_size
        """

        expX = np.exp(X)
        softmax = expX / np.sum(expX, axis=1).reshape(-1, 1)

        return softmax * (1 - softmax) * delta

    def forward(self, X):
        """

        :param X: np.array shape : (batch_size , layer_size)
        :return: np.array shape : (batch_size , layer_size)
        """

        expX = np.exp(X)
        return expX / np.sum(expX, axis=1).reshape(-1, 1)

    def zero_grad(self):
        pass

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_update_gradient(self, data, delta):
        pass

    def predict(self, X):
        raise NotImplementedError


class ReLU(Module):
    # leaky relu
    # X -> X si X>0 et alpha*X si X<0
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def update_parameters(self, gradient_step: float = 0.001):
        return None

    def backward_update_gradient(self, X: np.array, delta: np.array):
        # ReLu n'a pas de parametres donc pas besoin de faire ca.
        pass

    def backward_delta(self, X: np.array, delta: np.array):
        """Calcul la derivee de l'erreur

        :param X: batch x layer_size
        :param delta: batch x layer_size
        :return: batch x layer_size
        """

        return np.where(X > 0, 1, self.alpha) * delta

    def forward(self, X: np.array):
        """

        :param X: np.array shape : (batch_size , layer_size)
        :return: np.array shape : (batch_size , layer_size)
        """

        return np.where(X > 0, X, self.alpha * X)

    def zero_grad(self):
        pass

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_update_gradient(self, data, delta):
        pass

    def predict(self, X):
        raise NotImplementedError
