from torch import nn

from ..mytyping import ActivationName, Tensor
from ._CNN_modules import add_activation


class MLP(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        middle_dimension: int,
        output_dimension: int,
        num_layers: int,
        activation: ActivationName,
    ) -> None:
        super().__init__()
        self.input_dimension = input_dimension
        self.middle_dimension = middle_dimension
        self.output_dimension = output_dimension
        self.actfunc_str = activation

        self.layers = self._make_layers(
            num_layers - 1, input_dimension, output_dimension, activation
        )
        self.final_layer = nn.Linear(middle_dimension, output_dimension)
        self.activation = add_activation(activation)
        self.output_activation = (
            nn.Sigmoid() if output_dimension == 1 else nn.Softmax()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.output_activation(self.final_layer(x))
        return x

    @staticmethod
    def _make_layers(
        num_layers: int,
        input_dim: int,
        output_dim: int,
        activation: ActivationName,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Linear(input_dim, output_dim),
            add_activation(activation),
        ]
        _l = [
            nn.Linear(output_dim, output_dim)
            if i % 2 == 0
            else add_activation(activation)
            for i in range(2 * (num_layers - 1))
        ]
        return nn.Sequential(*(layers + _l))
