# Third Party Library
import numpy as np
import numpy.typing as npt


class MinMax:
    def __init__(self) -> None:
        self.min = np.nan
        self.max = np.nan

    def fit(self, x: npt.NDArray) -> None:
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)

    def transform(self, x: npt.NDArray) -> npt.NDArray:
        return (x - self.min) / (self.max - self.min)

    def fit_transform(self, x: npt.NDArray) -> npt.NDArray:
        if np.isnan([self.min, self.max]).any():
            self.fit(x)
        return self.transform(x)


class Centering:
    def __init__(self) -> None:
        self.mean = np.nan

    def fit(self, x: npt.NDArray) -> None:
        self.mean = x.mean(axis=0)

    def transform(self, x: npt.NDArray) -> npt.NDArray:
        return x - self.mean

    def fit_transform(self, x: npt.NDArray) -> npt.NDArray:
        if np.isnan(self.mean):
            self.fit(x)
        return self.transform(x)
