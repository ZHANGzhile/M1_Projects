import numpy as np
from module import Module


class Linear(Module):
    def __init__(self, input_dim, output_dim, init_type="normal", bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.include_bias = bias
        self.init_type = init_type

        if self.init_type == "normal":
            self._parameters["weight"] = (
                np.random.randn(self.input_dim, self.output_dim) * 0.01
            )
            self._parameters["bias"] = (
                np.random.randn(1, self.output_dim) if self.include_bias else None
            )
        elif self.init_type == "zeros":
            self._parameters["weight"] = np.zeros((self.input_dim, self.output_dim))
            self._parameters["bias"] = (
                np.zeros((1, self.output_dim)) if self.include_bias else None
            )
        else:
            raise ValueError(f"Unknown initialization type: {self.init_type}")

        self._gradient["weight"] = np.zeros_like(self._parameters["weight"])
        self._gradient["bias"] = (
            np.zeros_like(self._parameters["bias"])
            if self._parameters["bias"] is not None
            else None
        )

    def forward(self, x):
        # print(x.shape)
        assert (
            x.shape[1] == self.input_dim
        ), "Input x must have shape (batch_size, input_dim)"
        output = np.dot(x, self._parameters["weight"])
        if self._parameters["bias"] is not None:
            output += self._parameters["bias"]
        return output

    def backward_update_gradient(self, input, delta):
        assert (
            input.shape[1] == self.input_dim
        ), "Input must have shape (batch_size, input_dim)"
        assert (
            delta.shape[1] == self.output_dim
        ), "Delta must have shape (batch_size, output_dim)"
        self._gradient["weight"] += np.dot(input.T, delta)
        if self._parameters["bias"] is not None:
            self._gradient["bias"] += np.sum(delta, axis=0, keepdims=True)

    def backward_delta(self, input, delta):
        assert (
            input.shape[1] == self.input_dim
        ), "Input must have shape (batch_size, input_dim)"
        assert (
            delta.shape[1] == self.output_dim
        ), "Delta must have shape (batch_size, output_dim)"
        return np.dot(delta, self._parameters["weight"].T)

    def zero_grad(self):
        self._gradient["weight"].fill(0)
        if self._parameters["bias"] is not None:
            self._gradient["bias"].fill(0)

    def update_parameters(self, learning_rate=1e-3):
        self._parameters["weight"] -= learning_rate * self._gradient["weight"]
        if self._parameters["bias"] is not None:
            self._parameters["bias"] -= learning_rate * self._gradient["bias"]
