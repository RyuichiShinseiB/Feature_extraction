# Standard Library
from typing import Callable, Literal, overload

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


class LossFunction(nn.Module):
    def __init__(
        self,
        reconst_loss_type: Literal["bce", "mse", "ce", "None"],
        var_calc_type: Literal["softplus", "general"] | None,
    ) -> None:
        super().__init__()
        self.latent_loss: LatentLoss | None

        self.reconst_loss: nn.BCELoss | nn.MSELoss | nn.CrossEntropyLoss
        if reconst_loss_type == "bce":
            # self.reconst_loss = self.bce_loss
            self.reconst_loss = nn.BCELoss()
        elif reconst_loss_type == "mse":
            # self.reconst_loss = self.mse_loss
            self.reconst_loss = nn.MSELoss()
        elif reconst_loss_type == "ce":
            self.reconst_loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Please select another loss function")

        if var_calc_type is not None:
            self.latent_loss = LatentLoss(var_calc_type)
        else:
            self.latent_loss = None

    @overload
    def forward(self, pred: Tensor, t: Tensor) -> tuple[Tensor, None]:
        ...

    @overload
    def forward(
        self, pred: Tensor, t: Tensor, latent_params: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        ...

    def forward(
        self,
        pred: Tensor,
        t: Tensor,
        latent_params: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        reconst_error: Tensor = self.reconst_loss(pred, t)
        if self.latent_loss is not None and latent_params is not None:
            kldiv = self.latent_loss.forward(
                latent_params[-2], latent_params[-1]
            )
            return reconst_error, kldiv
        return reconst_error, None

    @staticmethod
    def calc_weight(val: Tensor | None) -> float:
        if val is None:
            return 0

        if val < 0.2:
            weight = 1e-2
        elif val < 0.4:
            weight = 1e-3
        elif val < 0.6:
            weight = 1e-4
        elif val < 0.8:
            weight = 1e-5
        else:
            weight = 1e-6

        # if val < 0.3:
        #     weight = 1e-4
        # elif val < 0.5:
        #     weight = 1e-5
        # elif val < 0.7:
        #     weight = 1e-6
        # elif val < 0.9:
        #     weight = 1e-7
        # else:
        #     weight = 0.0

        return weight


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
        return -0.5 * torch.sum(1 + var.log() - var - mean**2)

    @staticmethod
    def general_latent(mean: Tensor, log_var: Tensor) -> Tensor:
        return -0.5 * torch.mean(1 + log_var - log_var.exp() - mean**2)
