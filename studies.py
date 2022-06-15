from typing import List, Tuple

import optuna
from optuna import Study


def create_single_objective_studies() -> List[Study]:
    studies = []
    storage = optuna.storages.InMemoryStorage()

    # Single-objective study
    study = optuna.create_study(study_name="single", storage=storage)

    def objective_single(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", 0, 10)
        x2 = trial.suggest_float("x2", 0, 10)
        return (x1 - 2) ** 2 + (x2 - 5) ** 2

    study.optimize(objective_single, n_trials=50)
    studies.append(study)

    # Single-objective study with dynamic search space
    study = optuna.create_study(
        study_name="single-dynamic", storage=storage, direction="maximize"
    )

    def objective_single_dynamic(trial: optuna.Trial) -> float:
        category = trial.suggest_categorical("category", ["foo", "bar"])
        if category == "foo":
            return (trial.suggest_float("x1", 0, 10) - 2) ** 2
        else:
            return -((trial.suggest_float("x2", -10, 0) + 5) ** 2)

    study.optimize(objective_single_dynamic, n_trials=50)
    studies.append(study)
    return studies


def create_multi_objective_studies() -> List[Study]:
    studies = []
    storage = optuna.storages.InMemoryStorage()

    # Multi-objective study
    def objective_multi(trial: optuna.Trial) -> Tuple[float, float]:
        x = trial.suggest_float("x", 0, 5)
        y = trial.suggest_float("y", 0, 3)
        v0 = 4 * x**2 + 4 * y**2
        v1 = (x - 5) ** 2 + (y - 5) ** 2
        return v0, v1

    study = optuna.create_study(
        study_name="Multi-objective study with static search space",
        storage=storage,
        directions=["minimize", "minimize"],
    )
    study.optimize(objective_multi, n_trials=50)
    studies.append(study)

    # Multi-objective study with dynamic search space
    study = optuna.create_study(
        study_name="Multi-objective study with dynamic search space",
        storage=storage,
        directions=["minimize", "minimize"],
    )

    def objective_multi_dynamic(trial: optuna.Trial) -> Tuple[float, float]:
        category = trial.suggest_categorical("category", ["foo", "bar"])
        if category == "foo":
            x = trial.suggest_float("x1", 0, 5)
            y = trial.suggest_float("y1", 0, 3)
            v0 = 4 * x**2 + 4 * y**2
            v1 = (x - 5) ** 2 + (y - 5) ** 2
            return v0, v1
        else:
            x = trial.suggest_float("x2", 0, 5)
            y = trial.suggest_float("y2", 0, 3)
            v0 = 2 * x**2 + 2 * y**2
            v1 = (x - 2) ** 2 + (y - 3) ** 2
            return v0, v1

    study.optimize(objective_multi_dynamic, n_trials=50)
    studies.append(study)

    return studies


def create_intermediate_value_studies() -> List[Study]:
    # See https://github.com/optuna/optuna/blob/master/tests/visualization_tests/matplotlib_tests/test_intermediate_plot.py
    studies = []
    storage = optuna.storages.InMemoryStorage()

    def objective_simple(
        trial: optuna.Trial, report_intermediate_values: bool
    ) -> float:
        if report_intermediate_values:
            trial.report(1.0, step=0)
            trial.report(2.0, step=1)
        return 0.0

    def fail_objective(_: optuna.Trial) -> float:
        raise ValueError

    study = optuna.create_study(study_name="study with 1 trial", storage=storage)
    study.optimize(lambda t: objective_simple(t, True), n_trials=1)
    studies.append(study)

    study = optuna.create_study(
        study_name="study with only 1 trial that has no intermediate value",
        storage=storage,
    )
    study.optimize(lambda t: objective_simple(t, False), n_trials=1)
    studies.append(study)

    study = optuna.create_study(study_name="only failed trials", storage=storage)
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    studies.append(study)

    study = optuna.create_study(study_name="no trials", storage=storage)
    studies.append(study)
    return studies
