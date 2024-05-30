import matplotlib.pyplot as plt
import numpy as np
from module import Module
from tqdm import tqdm
from encapsulage import Sequential
from lineaire import Linear
from transformations import TanH, Sigmoid

class AutoEncoder_simple(Module):
    def __init__(self, DIM_IN, fc_act):
        super().__init__()
        self.encoder = Sequential(
            Linear(DIM_IN, 128),
            fc_act,
            Linear(128, 64),
            fc_act
        )
        self.decoder = Sequential(
            Linear(64, 128),
            fc_act,
            Linear(128, DIM_IN),
            Sigmoid()
        )
    def forward(self, x):
        encoded = x
        for layer in self.encoder.modules:
            encoded = layer.forward(encoded)

        decoded = encoded
        for layer in self.decoder.modules:
            decoded = layer.forward(decoded)

        return decoded
    
    def backward_delta(self, x, y):
        pass

    def backward_update_gradient(self, x, y):
        pass
    
    def zero_grad(self):
        pass


class AutoEncoder_complex(Module):
    def __init__(self, DIM_IN, fc_act):
        super().__init__()
        self.encoder = Sequential(
            Linear(DIM_IN, 512),
            fc_act,
            Linear(512, 256),
            fc_act,
            Linear(256, 128),
            fc_act,
            Linear(128, 64),
            fc_act
        )
        self.decoder = Sequential(
            Linear(64, 128),
            fc_act,
            Linear(128, 256),
            fc_act,
            Linear(256, 512),
            fc_act,
            Linear(512, DIM_IN),
            Sigmoid()
        )
    def forward(self, x):
        encoded = x
        for layer in self.encoder.modules:
            encoded = layer.forward(encoded)

        decoded = encoded
        for layer in self.decoder.modules:
            decoded = layer.forward(decoded)

        return decoded
    
    def backward_delta(self, x, y):
        pass

    def backward_update_gradient(self, x, y):
        pass
    
    def zero_grad(self):
        pass


class AutoEncoder_complex2(Module):
    def __init__(self, DIM_IN, fc_act):
        super().__init__()
        self.encoder = Sequential(
            Linear(DIM_IN, 128),
            fc_act,
            Linear(128, 64),
            fc_act,
            Linear(64, 32),
            fc_act
        )
        self.decoder = Sequential(
            Linear(32, 64),
            fc_act,
            Linear(64, 128),
            fc_act,
            Linear(128, DIM_IN),
            Sigmoid()
        )
    def forward(self, x):
        encoded = x
        for layer in self.encoder.modules:
            encoded = layer.forward(encoded)

        decoded = encoded
        for layer in self.decoder.modules:
            decoded = layer.forward(decoded)

        return decoded
    
    def backward_delta(self, x, y):
        pass

    def backward_update_gradient(self, x, y):
        pass
    
    def zero_grad(self):
        pass