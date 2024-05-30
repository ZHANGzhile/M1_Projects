import matplotlib.pyplot as plt
import numpy as np
from module import Module
from tqdm import tqdm


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        self.inputs = []

    def forward(self, X):
        self.inputs = [X]
        for module in self.modules:
            X = module.forward(X)
            self.inputs.append(X)
        return X

    def backward_update_gradient(self, input, delta):
        for i, module in enumerate(reversed(self.modules)):
            module.backward_update_gradient(self.inputs[-i - 2], delta)
            delta = module.backward_delta(self.inputs[-i - 2], delta)

    def backward_delta(self, input, delta):
        for module in reversed(self.modules):
            delta = module.backward_delta(module.forward(input), delta)
            input = module.forward(input)
        return delta

    def update_parameters(self, learning_rate=1e-3):
        for module in self.modules:
            module.update_parameters(learning_rate)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()


class Optim(object):
    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        # Forward pass
        output = self.net.forward(batch_x)

        # Compute loss
        loss_value = np.mean(self.loss.forward(batch_y, output))

        # Backward pass
        gradient = self.loss.backward(batch_y, output)
        self.net.zero_grad()
        self.net.backward_update_gradient(batch_x, gradient)
        self.net.update_parameters(self.eps)

        return loss_value

    def SGD(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: None,
        epochs: int,
        plot: bool = False,
        x_test: np.ndarray = None,
        y_test: np.ndarray = None,
        shuffle: bool = True,
    ):
        num_samples = x_train.shape[0]
        if batch_size == None:
            batch_size = num_samples

        costs = []
        acc_train = []
        acc_test = []

        for epoch in tqdm(range(epochs)):
            cost = 0
            if shuffle:
                idx = np.random.permutation(num_samples)
                x_train = x_train[idx]
                y_train = y_train[idx]

            for i in range(0, num_samples, batch_size):
                batch_x = x_train[i : i + batch_size]
                batch_y = y_train[i : i + batch_size]
                cost += np.mean(self.step(batch_x, batch_y))
                #print(cost)

            cost /= num_samples // batch_size
            #print(cost)

            if plot:
                costs.append(cost)
                acc_train.append(self.score(x_train, y_train) * 100)
                if x_test is not None:
                    acc_test.append(self.score(x_test, y_test) * 100)

        if plot:
            self.plot(costs, acc_train, acc_test)

    def score(self, X, y):
        assert X.shape[0] == y.shape[0], ValueError()
        if len(y.shape) != 1:  # eventual y with OneHot encoding
            y = y.argmax(axis=1)
        y_hat = np.argmax(self.net.forward(X), axis=1)
        return np.where(y == y_hat, 1, 0).mean()

    def plot(self, costs, acc_train, acc_test):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.plot(costs)

        # representer l'accuracy de train et test
        ax2.set_title("Training and Test Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy (%)")
        ax2.plot(acc_train, label="Train", color="red")
        ax2.plot(acc_test, label="Test", color="blue")
        ax2.legend()
        plt.show()
        
