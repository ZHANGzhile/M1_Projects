from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def forward(self, y, yhat):
        pass

    @abstractmethod
    def backward(self, y, yhat):
        pass


class Module(ABC):
    def __init__(self):
        self._parameters = {}
        self._gradient = {}

    @abstractmethod
    def zero_grad(self):
        ## Annule gradient
        pass

    @abstractmethod
    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step * self._gradient

    @abstractmethod
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    @abstractmethod
    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

    def predict(self, X):
        ## Calcule la prediction
        pass
