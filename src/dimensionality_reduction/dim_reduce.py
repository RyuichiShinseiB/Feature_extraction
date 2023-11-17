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

    def whatis_reduction_meth_name(self) -> str:
        return str(type(self.reduction_meth))

    def whatis_scaler_name(self) -> str:
        return str(type(self.scaler))

    def fit_reduction_meth(self, x: npt.NDArray) -> None:
        if self.scaler is not None:
            self.scaler.fit(x)
        self.reduction_meth.fit(x)
        self.fitting_flags["reduction_meth"] = True

    def fit_transform_reduction_meth(self, x: npt.NDArray) -> None:
        if self.scaler is not None:
            if not self.fitting_flags["scaler"]:
                self.scaler.fit(x)
        self.reduction_meth.fit_transform(x)
        self.fitting_flags["reduction_meth"] = True


if __name__ == "__main__":
    obj = DimReduction(
        PCA,
        StandardScaler,
    )
    print(obj.whatis_reduction_meth_name)
    print(obj.whatis_scaler_name())
