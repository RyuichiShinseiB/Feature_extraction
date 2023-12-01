# Standard Library
import pickle
from pathlib import Path
from typing import TypeAlias, cast

# Third Party Library
import optuna
import torch.nn as nn
from sklearn.mixture import BayesianGaussianMixture

_TorchModel = nn.Module
Model: TypeAlias = BayesianGaussianMixture | _TorchModel


class ModelRegistry:
    def __init__(self, study_name: str) -> None:
        self._models: dict[int, Model] = {}
        save_model = Path("./model") / f".{study_name}_best_model"
        if not save_model.exists():
            save_model.mkdir(parents=True, exist_ok=True)
        self.best_model_file = save_model / "model.pickle"

    def register(self, trial_id: int, model: Model) -> None:
        self._models[trial_id] = model

    def retrieve(self, trial_id: int) -> Model | None:
        return self._models.get(trial_id)

    def save_best_model(self, best_trial_id: int) -> None:
        best_model = self._models[best_trial_id]
        if best_model is not None:
            with open(self.best_model_file, "wb") as f:
                pickle.dump(best_model, f)
        else:
            print("Best model is None.")

    def load_best_model(self) -> Model | None:
        if self.best_model_file.exists():
            with open(self.best_model_file, "rb") as f:
                loaded_model = pickle.load(f)
            if isinstance(loaded_model, (BayesianGaussianMixture, nn.Module)):
                return cast(Model, loaded_model)
            else:
                return None
        else:
            return None


class StudyCallback:
    def __init__(self, model_registry: ModelRegistry) -> None:
        self._model_registry = model_registry

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        if study.best_trial.number == trial.number:
            self._model_registry.save_best_model(study.best_trial.number)
