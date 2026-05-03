"""AutoML and hyperparameter search utilities."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping


Objective = Callable[[Mapping[str, Any]], float]


@dataclass
class SearchSpace:
    """Typed search space with continuous and categorical parameters."""

    floats: dict[str, tuple[float, float]] = field(default_factory=dict)
    ints: dict[str, tuple[int, int]] = field(default_factory=dict)
    categoricals: dict[str, tuple[Any, ...]] = field(default_factory=dict)

    def sample(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key, (low, high) in self.floats.items():
            params[key] = random.uniform(low, high)
        for key, (low, high) in self.ints.items():
            params[key] = random.randint(low, high)
        for key, values in self.categoricals.items():
            params[key] = random.choice(values)
        return params


@dataclass
class TrialResult:
    """One hyperparameter trial."""

    params: dict[str, Any]
    score: float
    trial_id: int


class HyperparameterTuner:
    """Optuna-first tuner with random/evolutionary fallbacks."""

    def __init__(self, objective: Objective, search_space: SearchSpace, direction: str = "maximize") -> None:
        self.objective = objective
        self.search_space = search_space
        self.direction = direction

    def random_search(self, trials: int) -> list[TrialResult]:
        results = []
        for idx in range(trials):
            params = self.search_space.sample()
            results.append(TrialResult(params, self.objective(params), idx))
        return sorted(results, key=lambda item: item.score, reverse=self.direction == "maximize")

    def evolutionary_search(self, trials: int, population: int = 8, mutation_rate: float = 0.2) -> list[TrialResult]:
        evaluated = self.random_search(max(1, population))
        for idx in range(population, trials):
            parent = random.choice(evaluated[: max(1, population // 2)])
            child = dict(parent.params)
            fresh = self.search_space.sample()
            for key, value in fresh.items():
                if random.random() < mutation_rate:
                    child[key] = value
            evaluated.append(TrialResult(child, self.objective(child), idx))
            evaluated.sort(key=lambda item: item.score, reverse=self.direction == "maximize")
            evaluated = evaluated[:population]
        return evaluated

    def optuna_search(self, trials: int, study_name: str = "plasformers") -> list[TrialResult]:
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("Optuna search requires optional dependency 'optuna'") from exc

        def suggest(trial: Any) -> dict[str, Any]:
            params: dict[str, Any] = {}
            for key, (low, high) in self.search_space.floats.items():
                params[key] = trial.suggest_float(key, low, high, log=key.endswith("lr"))
            for key, (low, high) in self.search_space.ints.items():
                params[key] = trial.suggest_int(key, low, high)
            for key, values in self.search_space.categoricals.items():
                params[key] = trial.suggest_categorical(key, list(values))
            return params

        def wrapped(trial: Any) -> float:
            return self.objective(suggest(trial))

        study = optuna.create_study(study_name=study_name, direction=self.direction)
        study.optimize(wrapped, n_trials=trials)
        return [
            TrialResult(dict(trial.params), float(trial.value), trial.number)
            for trial in sorted(study.trials, key=lambda t: t.value or 0.0, reverse=self.direction == "maximize")
        ]

    @staticmethod
    def save_results(results: Iterable[TrialResult], path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([result.__dict__ for result in results], indent=2), encoding="utf-8")
        return out


def default_detection_space() -> SearchSpace:
    """Search space covering LR, augmentation, loss, and scaling factors."""

    return SearchSpace(
        floats={
            "optimization.base_lr": (1e-5, 5e-3),
            "optimization.weight_decay": (1e-5, 0.1),
            "augmentation.copy_paste.prob": (0.0, 0.7),
            "augmentation.mixup.prob": (0.0, 0.4),
            "loss.bbox_weight": (3.0, 12.0),
            "loss.dfl_weight": (0.5, 3.0),
            "model.width_mult": (0.35, 1.25),
            "model.depth_mult": (0.33, 1.20),
        },
        ints={"assignment.topk": (5, 30)},
        categoricals={"loss.bbox": ("ciou", "siou")},
    )
