import os
import time
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import optuna
from optuna import Study


StudiesType = Union[Study, Sequence[Study]]


def objective_single(trial: optuna.Trial) -> float:
    x1 = trial.suggest_float("x1", 0, 10)
    x2 = trial.suggest_float("x2", 0, 10)
    return (x1 - 2) ** 2 + (x2 - 5) ** 2


def objective_single_dynamic(trial: optuna.Trial) -> float:
    category = trial.suggest_categorical("category", ["foo", "bar"])
    if category == "foo":
        return (trial.suggest_float("x1", 0, 10) - 2) ** 2
    else:
        return -((trial.suggest_float("x2", -10, 0) + 5) ** 2)


def create_single_objective_studies() -> List[Tuple[str, StudiesType]]:
    studies: List[Tuple[str, StudiesType]] = []
    storage = optuna.storages.InMemoryStorage()

    # Single-objective study
    study = optuna.create_study(
        study_name="A single objective study with 2-dimensional static search space",
        storage=storage,
    )

    study.optimize(objective_single, n_trials=50)
    studies.append((study.study_name, study))

    # Single-objective study with dynamic search space
    study = optuna.create_study(
        study_name="A single-objective study with 3-dimensional dynamic search space",
        storage=storage,
        direction="maximize",
    )

    study.optimize(objective_single_dynamic, n_trials=50)
    studies.append((study.study_name, study))

    # Single-objective study with None categorical value
    study = optuna.create_study(
        study_name="A single objective study that suggests None as a categorical parameter",
        storage=storage,
    )

    def objective_single_none_categorical(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -100, 100)
        trial.suggest_categorical("y", ["foo", None])
        return x**2

    study.optimize(objective_single_none_categorical, n_trials=10)
    studies.append((study.study_name, study))

    # No trials single-objective study
    optuna.create_study(study_name="A single objective study that has no trials", storage=storage)
    return studies


def create_multiple_single_objective_studies() -> List[Tuple[str, StudiesType]]:
    studies: List[Tuple[str, StudiesType]] = []
    storage = optuna.storages.InMemoryStorage()

    # Single-objective study
    _static: List[Study] = []
    for i in range(2):
        study = optuna.create_study(
            study_name=f"static{i}",
            storage=storage,
        )
        study.optimize(objective_single, n_trials=50)
        _static.append(study)
    title = "Two single objective studies with 2-dimensional static search space"
    studies.append((title, _static))

    # Single-objective study with dynamic search space
    _dynamic: List[Study] = []
    for i in range(2):
        study = optuna.create_study(
            study_name=f"dynamic{i}",
            storage=storage,
            direction="maximize",
        )
        study.optimize(objective_single_dynamic, n_trials=50)
        _dynamic.append(study)
    title = "Two single objective studies with 3-dimensional dynamic search space"
    studies.append((title, _dynamic))

    return studies


def create_multi_objective_studies() -> List[Tuple[str, StudiesType]]:
    studies: List[Tuple[str, StudiesType]] = []
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
    studies.append((study.study_name, study))

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
    studies.append((study.study_name, study))

    return studies


def create_intermediate_value_studies() -> List[Tuple[str, StudiesType]]:
    studies: List[Tuple[str, StudiesType]] = []
    storage = optuna.storages.InMemoryStorage()

    def objective_simple(trial: optuna.Trial, report_intermediate_values: bool) -> float:
        if report_intermediate_values:
            trial.report(1.0, step=0)
            trial.report(2.0, step=1)
        return 0.0

    def objective_single_inf_report(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        if trial.number % 3 == 0:
            trial.report(float("inf"), 1)
        elif trial.number % 3 == 1:
            trial.report(float("-inf"), 1)
        else:
            trial.report(float("nan"), 1)

        if x > 0:
            raise optuna.TrialPruned()
        else:
            return x**2

    def fail_objective(_: optuna.Trial) -> float:
        raise ValueError

    study = optuna.create_study(study_name="Study with 1 trial", storage=storage)
    study.optimize(lambda t: objective_simple(t, True), n_trials=1)
    studies.append((study.study_name, study))

    study = optuna.create_study(
        study_name="Study that is pruned after 'inf', '-inf', or 'nan'", storage=storage
    )
    study.optimize(objective_single_inf_report, n_trials=50)
    studies.append((study.study_name, study))

    study = optuna.create_study(
        study_name="Study with only 1 trial that has no intermediate value",
        storage=storage,
    )
    study.optimize(lambda t: objective_simple(t, False), n_trials=1)
    studies.append((study.study_name, study))

    study = optuna.create_study(study_name="Study that has only failed trials", storage=storage)
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    studies.append((study.study_name, study))

    study = optuna.create_study(study_name="Study that has no trials", storage=storage)
    studies.append((study.study_name, study))
    return studies


def create_pytorch_study() -> Optional[Study]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        import torch.utils.data
        from torchvision import datasets
        from torchvision import transforms
    except ImportError:
        print("create_pytorch_studies is skipped because torch/torchvision is not found")
        return None

    DEVICE = torch.device("cpu")
    BATCHSIZE = 128
    CLASSES = 10
    DIR = os.getcwd()
    EPOCHS = 10
    N_TRAIN_EXAMPLES = BATCHSIZE * 30
    N_VALID_EXAMPLES = BATCHSIZE * 10

    def define_model(trial: optuna.Trial) -> "torch.nn.Module":
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers: List["torch.nn.Module"] = []

        in_features = 28 * 28
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
            layers.append(nn.Dropout(p))

            in_features = out_features
        layers.append(nn.Linear(in_features, CLASSES))
        layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)

    def get_mnist() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        # Load FashionMNIST dataset.
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=BATCHSIZE,
            shuffle=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
            batch_size=BATCHSIZE,
            shuffle=True,
        )

        return train_loader, valid_loader

    def objective(trial: optuna.Trial) -> float:
        # Generate the model.
        model = define_model(trial).to(DEVICE)

        # Generate the optimizers.
        optimizer_name: str = trial.suggest_categorical(
            "optimizer", ["Adam", "RMSprop", "SGD"]
        )  # type: ignore
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # Get the FashionMNIST dataset.
        train_loader, valid_loader = get_mnist()

        # Training of the model.
        for epoch in range(EPOCHS):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # Limiting training data for faster epochs.
                if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                    break

                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

            # Validation of the model.
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_loader):
                    # Limiting validation data.
                    if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                        break
                    data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                    output = model(data)
                    # Get the index of the max log-probability.
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)  # type: ignore

            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy

    study = optuna.create_study(
        direction="maximize", study_name="pytorch_simple.py in optuna-example"
    )
    study.optimize(objective, n_trials=50, timeout=600)
    return study


def create_terminator_studies() -> List[Tuple[str, StudiesType]]:
    studies: list[tuple[str, StudiesType]] = []
    storage = optuna.storages.InMemoryStorage()

    def objective_terminator_simple(trial):
        x = trial.suggest_float("x", -100, 100)
        y = trial.suggest_categorical("y", [-1, 0, 1])
        return x ** 2 + y

    study = optuna.create_study(
        study_name="Terminator Simple",
        sampler=optuna.samplers.TPESampler(seed=10),
        storage=storage,
    )
    study.optimize(objective_terminator_simple, n_trials=10)
    studies.append((study.study_name, study))
    return studies


def create_terminator_studies_with_validation_score() -> List[Tuple[str, StudiesType]]:
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_wine
    from sklearn.model_selection import KFold
    from lightgbm import LGBMClassifier
    from optuna.terminator.erroreval import report_cross_validation_scores

    studies: list[tuple[str, StudiesType]] = []
    storage = optuna.storages.InMemoryStorage()

    def objective_lightgbm(trial):
        X, y = load_wine(return_X_y=True)

        clf = LGBMClassifier(
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            num_leaves=trial.suggest_int("num_leaves", 2, 256),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            subsample_freq=trial.suggest_int("subsample_freq", 1, 7),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        )

        scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True))
        report_cross_validation_scores(trial, scores)
        return scores.mean()

    study = optuna.create_study(
        study_name="Terminator LightGBM",
        storage=storage,
    )
    study.optimize(objective_lightgbm, n_trials=50, timeout=600)
    studies.append((study.study_name, study))
    return studies


def create_single_objective_studies_for_timeline() -> List[Tuple[str, StudiesType]]:
    studies: List[Tuple[str, StudiesType]] = []
    storage = optuna.storages.InMemoryStorage()

    def objective_timeline(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0, 1)
        time.sleep(x * 0.1)
        if x > 0.8:
            raise ValueError()
        if x > 0.4:
            raise optuna.TrialPruned()
        return x**2

    # Single-objective study
    study = optuna.create_study(
        study_name="A single objective study consuming time",
        storage=storage,
    )

    study.enqueue_trial({"x": 0.3})  # Add a COMPLETE trial.
    study.enqueue_trial({"x": 0.9})  # Add a FAIL trial.
    study.enqueue_trial({"x": 0.5})  # Add a PRUNED trial.
    study.optimize(objective_timeline, n_trials=50, n_jobs=2, catch=(ValueError,))
    studies.append((study.study_name, study))

    # Single-objective study
    study = optuna.create_study(
        study_name=(
            "A single objective study consuming time and "
            "the order of legends is different from the order of trials"
        ),
        storage=storage,
    )
    study.enqueue_trial({"x": 0.9})  # Add a FAIL trial.
    study.enqueue_trial({"x": 0.5})  # Add a PRUNED trial.
    study.enqueue_trial({"x": 0.3})  # Add a COMPLETE trial.
    study.optimize(objective_timeline, n_trials=50, n_jobs=2, catch=(ValueError,))
    studies.append((study.study_name, study))

    # No trials single-objective study
    study = optuna.create_study(
        study_name="A single objective study that has no trials", storage=storage
    )
    studies.append((study.study_name, study))
    return studies
