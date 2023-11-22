# Standard Library
from typing import TypeAlias

# Third Party Library
import numpy.typing as npt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import RobustScaler, StandardScaler

ReductionMeth: TypeAlias = PCA | KernelPCA
Scaler: TypeAlias = RobustScaler | StandardScaler


class DimReduction:
    def __init__(
        self, reduction_meth: ReductionMeth, scaler: Scaler | None
    ) -> None:
        self.reduction_meth = reduction_meth
        self.scaler = scaler
        self.fitting_flags = {
            "reduction_meth": False,
            "scaler": False,
        }

    def fit(self, x: npt.NDArray) -> None:
        if self.scaler is not None:
            x = self.scaler.fit_transform(x)
        self.reduction_meth.fit(x)
        self.fitting_flags["reduction_meth"] = True

    def fit_transform(self, x: npt.NDArray) -> npt.NDArray:
        if self.scaler is not None:
            x = self.scaler.fit_transform(x)
        x_fitted = self.reduction_meth.fit_transform(x)
        self.fitting_flags["reduction_meth"] = True
        return x_fitted

    def transform(self, x: npt.NDArray) -> npt.NDArray:
        if self.scaler is not None:
            x = self.scaler.fit_transform(x)
        x_transformed = self.reduction_meth.transform(x)
        self.fitting_flags["reduction_meth"] = True

        return x_transformed


if __name__ == "__main__":
    obj = DimReduction(
        PCA,
        StandardScaler,
    )
    obj.reduction_meth
    obj.scaler
