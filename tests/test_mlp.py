from torch import nn

from src.predefined_models._MLP import MLP


def test_make_layers() -> None:
    num_layers = 10
    input_dim = 10
    output_dim = 20
    activation = "relu"
    layers = MLP._make_layers(num_layers, input_dim, output_dim, activation)

    first_layer = layers[0]
    first_layer_act = layers[1]
    second_layer = layers[2]

    first_layer_params = tuple(first_layer.parameters())[0]
    second_layer_params = tuple(second_layer.parameters())[0]

    assert len(layers) == 2 * num_layers
    assert isinstance(first_layer_act, nn.ReLU)
    assert first_layer_params.shape == (output_dim, input_dim)
    assert second_layer_params.shape == (output_dim, output_dim)
