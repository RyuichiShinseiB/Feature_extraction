from collections.abc import Generator, Sequence

from torch import nn

from ..mytyping import ActFunc, ActFuncName, Tensor
from ._CNN_modules import add_actfunc


class MLP(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        middle_dimensions: Sequence[int],
        output_dimension: int,
        actfunc: ActFuncName,
        output_actfunc: ActFuncName | None,
        dropout_rate: float | None = None,
    ) -> None:
        super().__init__()
        self.input_dimension = input_dimension
        self.middle_dimensions = middle_dimensions
        self.output_dimension = output_dimension
        self.actfunc_str = actfunc
        self.dropout_rate = dropout_rate or 0.0

        self.layers = self._make_layers(
            input_dimension, middle_dimensions, actfunc
        )
        self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        self.final_layer = nn.Linear(middle_dimensions[-1], output_dimension)
        self.actfunc = add_actfunc(actfunc)
        if output_actfunc is None:
            self.output_actfunc: ActFunc = (
                nn.Sigmoid() if output_dimension == 1 else nn.Softmax(1)
            )
        else:
            self.output_actfunc = add_actfunc(output_actfunc)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.output_actfunc(self.final_layer(x))
        return x

    def get_final_actfunc(self, x: Tensor) -> Tensor:
        x = self.final_layer(self.layers(x))
        return x

    def _make_layers(
        self,
        input_dim: int,
        mid_dims: Sequence[int],
        actfunc: ActFuncName,
    ) -> nn.Sequential:
        def _gen_each_layer() -> Generator[nn.Module, None, None]:
            for indim, outdim in inout_pairs:
                yield nn.Linear(indim, outdim)
                yield add_actfunc(actfunc)
                # yield nn.Dropout(self.dropout_rate)

        dims = [input_dim] + list(mid_dims)
        inout_pairs = zip(dims, dims[1:])
        layers = list(_gen_each_layer())

        return nn.Sequential(*layers)
