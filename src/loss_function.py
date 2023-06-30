# Standard Library
from typing import Callable, Literal

# Third Party Library
import torch

# Local Library
from . import Model, Tensor


def calc_loss(
    input_data: Tensor,
    reconst_loss: Callable[[Tensor, Tensor], Tensor],
    latent_loss: Callable[[Tensor, Tensor], Tensor] | None,
    model: Model,
) -> tuple[Tensor, Tensor]:
    x_pred, _latent = model(input_data)
    loss = reconst_loss(x_pred, input_data)
    if latent_loss is not None:
        loss += latent_loss(_latent[-2], _latent[-1])
    return loss, x_pred


class LossFunction:
    def __init__(
        self,
        reconst_loss_type: Literal["bce", "mse"],
        var_calc_type: Literal["softplus", "general"] | None,
    ) -> None:
        self.reconst: torch.nn.BCELoss | torch.nn.MSELoss
        self.latent: LatentLoss | None

        if reconst_loss_type == "bce":
            self.reconst = torch.nn.BCELoss()
        elif reconst_loss_type == "mse":
            self.reconst = torch.nn.MSELoss()
        else:
            raise RuntimeError("Please select another loss function")

        if var_calc_type is not None:
            self.latent = LatentLoss(var_calc_type)
        else:
            self.latent = None


class LatentLoss:
    def __init__(self, var_calc_type: Literal["softplus", "general"]) -> None:
        if var_calc_type == "softplus":
            self.latent_loss = self.softplus_latent
        elif var_calc_type == "standard":
            self.latent_loss = self.general_latent
        else:
            raise RuntimeError("'softplus' or 'general' can be selected.")

    def __call__(self, mean: Tensor, var: Tensor) -> Tensor:
        return self.latent_loss(mean, var)

    @staticmethod
    def softplus_latent(mean: Tensor, var: Tensor) -> Tensor:
        eps = 1e-5
        return -0.5 * torch.mean(
            torch.sum(1 + torch.log(var + eps) - mean**2 - var, dim=1).view(
                -1
            )
        )

    @staticmethod
    def general_latent(mean: Tensor, log_var: Tensor) -> Tensor:
        return -0.5 * torch.mean(
            torch.sum(1 + log_var - mean**2 - log_var.exp(), dim=1)
        )
