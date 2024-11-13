# Standard Library
from typing import Callable, Literal

# Third Party Library
import torch
from torch import nn

# Local Library
from .mytyping import Model, Tensor


def torch_log(x: Tensor) -> Tensor:
    return torch.log(torch.clamp(x, min=1e-10))


def calc_loss(
    input_data: Tensor,
    reconst_loss: Callable[[Tensor, Tensor], Tensor],
    latent_loss: Callable[[Tensor, Tensor], Tensor] | None,
    model: Model,
) -> tuple[dict[Literal["reconst", "kldiv"], Tensor], Tensor]:
    loss: dict[Literal["reconst", "kldiv"], Tensor] = {}
    x_pred, _latent = model(input_data)
    loss["reconst"] = reconst_loss(
        x_pred.flatten(start_dim=1), input_data.flatten(start_dim=1)
    )
    if latent_loss is not None:
        loss["kldiv"] = latent_loss(_latent[-2], _latent[-1])
    return loss, x_pred


class LossFunction:
    def __init__(
        self,
        reconst_loss_type: Literal["bce", "mse", "None"],
        var_calc_type: Literal["softplus", "general"] | None,
    ) -> None:
        self.latent_loss: LatentLoss | None

        if reconst_loss_type == "bce":
            # self.reconst_loss = self.bce_loss
            self.reconst_loss: nn.BCELoss | nn.MSELoss = nn.BCELoss()
        elif reconst_loss_type == "mse":
            # self.reconst_loss = self.mse_loss
            self.reconst_loss = nn.MSELoss()
        else:
            raise ValueError("Please select another loss function")

        if var_calc_type is not None:
            self.latent_loss = LatentLoss(var_calc_type)
        else:
            self.latent_loss = None

    @staticmethod
    def bce_loss(reconst_data: Tensor, input_data: Tensor) -> Tensor:
        return -torch.mean(
            torch.sum(
                input_data * torch_log(reconst_data)
                + (1 - input_data) * torch_log(1 - reconst_data),
                dim=1,
            )
        )

    @staticmethod
    def mse_loss(reconst_data: Tensor, input_data: Tensor) -> Tensor:
        return -torch.mean(
            torch.sum(
                (reconst_data - input_data) ** 2,
                dim=1,
            )
        )


class LatentLoss(nn.Module):
    def __init__(self, var_calc_type: Literal["softplus", "general"]) -> None:
        super().__init__()
        if var_calc_type == "softplus":
            self.latent_loss = self.softplus_latent
        elif var_calc_type == "standard":
            self.latent_loss = self.general_latent
        else:
            raise ValueError("'softplus' or 'general' can be selected.")

    def forward(self, mean: Tensor, var: Tensor) -> Tensor:
        return self.latent_loss(mean, var)

    @staticmethod
    def softplus_latent(mean: Tensor, var: Tensor) -> Tensor:
        return -0.5 * torch.mean(
            # torch.sum(1 + 2 * torch_log(std) - mean**2 - std**2, dim=1)
            torch.sum(1 + torch_log(var) - mean**2 - var, dim=1)
        )

    @staticmethod
    def general_latent(mean: Tensor, log_var: Tensor) -> Tensor:
        return -0.5 * torch.mean(
            torch.sum(1 + log_var - mean**2 - log_var.exp(), dim=1)
        )
