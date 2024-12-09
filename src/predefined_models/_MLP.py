from collections.abc import Generator, Sequence

from torch import nn

from ..mytyping import ActivationName, Tensor
from ._CNN_modules import add_activation


class MLP(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        middle_dimensions: Sequence[int],
        output_dimension: int,
        activation: ActivationName,
    ) -> None:
        super().__init__()
        self.input_dimension = input_dimension
        self.middle_dimensions = middle_dimensions
        self.output_dimension = output_dimension
        self.actfunc_str = activation

        self.layers = self._make_layers(
            input_dimension, middle_dimensions, activation
        )
        self.final_layer = nn.Linear(middle_dimensions[-1], output_dimension)
        self.activation = add_activation(activation)
        self.output_activation = (
            nn.Sigmoid()
            if output_dimension == 1
            else nn.Softmax(output_dimension)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.output_activation(self.final_layer(x))
        return x

    @staticmethod
    def _make_layers(
        input_dim: int,
        mid_dims: Sequence[int],
        activation: ActivationName,
    ) -> nn.Sequential:
        def _gen_each_layer() -> Generator[nn.Module, None, None]:
            for indim, outdim in inout_pairs:
                yield nn.Linear(indim, outdim)
                yield add_activation(activation)

        dims = [input_dim] + list(mid_dims)
        inout_pairs = zip(dims, dims[1:])
        layers = list(_gen_each_layer())

        return nn.Sequential(*layers)
