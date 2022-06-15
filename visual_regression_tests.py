import argparse
import os
import warnings
from typing import List, Tuple

import matplotlib.pylab as plt
import optuna
import optuna.visualization as plotly_visualization
import optuna.visualization.matplotlib as matplotlib_visualization
from jinja2 import Environment, FileSystemLoader
from optuna import Study
from optuna.exceptions import ExperimentalWarning

try:
    from optuna_fast_fanova import FanovaImportanceEvaluator
except ImportError:
    from optuna.importance import FanovaImportanceEvaluator

warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)

figsize = (8, 6)
dpi = 100
plt.rcParams["figure.figsize"] = figsize

template_dirs = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")]
plot_functions = [
    "plot_edf",
    "plot_optimization_history",
    "plot_param_importances",
    "plot_parallel_coordinate",
    "plot_slice",
]

parser = argparse.ArgumentParser()
parser.add_argument("func", help="plot function name", choices=plot_functions)
parser.add_argument("--dir", help="output directory", default="tmp")
args = parser.parse_args()


def create_studies() -> List[Study]:
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


def generate_optimization_history_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    files = []
    for study in studies:
        plotly_filepath = os.path.join(base_dir, f"{study.study_name}-plotly.png")
        plotly_fig = plotly_visualization.plot_optimization_history(study)
        plotly_fig.update_layout(
            width=figsize[0] * dpi, height=figsize[1] * dpi, margin={"l": 10, "r": 10}
        )
        plotly_fig.write_image(plotly_filepath)

        matplotlib_filepath = os.path.join(
            base_dir, f"{study.study_name}-matplotlib.png"
        )
        matplotlib_visualization.plot_optimization_history(study)
        plt.savefig(matplotlib_filepath, bbox_inches="tight", dpi=dpi)

        files.append((study, plotly_filepath, matplotlib_filepath))
    return files


def generate_edf_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    files = []
    for study in studies:
        plotly_filepath = os.path.join(base_dir, f"{study.study_name}-plotly.png")
        plotly_fig = plotly_visualization.plot_edf(study)
        plotly_fig.update_layout(
            width=figsize[0] * dpi, height=figsize[1] * dpi, margin={"l": 10, "r": 10}
        )
        plotly_fig.write_image(plotly_filepath)

        matplotlib_filepath = os.path.join(
            base_dir, f"{study.study_name}-matplotlib.png"
        )
        matplotlib_visualization.plot_edf(study)
        plt.savefig(matplotlib_filepath, bbox_inches="tight", dpi=dpi)

        files.append((study, plotly_filepath, matplotlib_filepath))
    return files


def generate_slice_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    files = []
    for study in studies:
        plotly_filepath = os.path.join(base_dir, f"{study.study_name}-plotly.png")
        plotly_fig = plotly_visualization.plot_slice(study)
        plotly_fig.update_layout(
            width=figsize[0] * dpi, height=figsize[1] * dpi, margin={"l": 10, "r": 10}
        )
        plotly_fig.write_image(plotly_filepath)

        matplotlib_filepath = os.path.join(
            base_dir, f"{study.study_name}-matplotlib.png"
        )
        matplotlib_visualization.plot_slice(study)
        plt.savefig(matplotlib_filepath, bbox_inches="tight", dpi=dpi)

        files.append((study, plotly_filepath, matplotlib_filepath))
    return files


def generate_param_importances_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    files = []
    seed = 0
    for study in studies:
        plotly_filepath = os.path.join(base_dir, f"{study.study_name}-plotly.png")
        plotly_fig = plotly_visualization.plot_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=seed))
        plotly_fig.update_layout(
            width=figsize[0] * dpi, height=figsize[1] * dpi, margin={"l": 10, "r": 10}
        )
        plotly_fig.write_image(plotly_filepath)

        matplotlib_filepath = os.path.join(
            base_dir, f"{study.study_name}-matplotlib.png"
        )
        matplotlib_visualization.plot_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=seed))
        plt.savefig(matplotlib_filepath, bbox_inches="tight", dpi=dpi)

        files.append((study, plotly_filepath, matplotlib_filepath))
    return files


def generate_parallel_coordinate_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    files = []
    for study in studies:
        plotly_filepath = os.path.join(base_dir, f"{study.study_name}-plotly.png")
        plotly_fig = plotly_visualization.plot_parallel_coordinate(study)
        plotly_fig.update_layout(
            width=figsize[0] * dpi, height=figsize[1] * dpi, margin={"l": 10, "r": 10}
        )
        plotly_fig.write_image(plotly_filepath)

        matplotlib_filepath = os.path.join(
            base_dir, f"{study.study_name}-matplotlib.png"
        )
        matplotlib_visualization.plot_parallel_coordinate(study)
        plt.savefig(matplotlib_filepath, bbox_inches="tight", dpi=dpi)

        files.append((study, plotly_filepath, matplotlib_filepath))
    return files


def main():
    studies = create_studies()
    base_dir = os.path.abspath(args.dir)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    if args.func == "plot_optimization_history":
        plot_files = generate_optimization_history_plots(studies, base_dir)
    elif args.func == "plot_edf":
        plot_files = generate_edf_plots(studies, base_dir)
    elif args.func == "plot_slice":
        plot_files = generate_slice_plots(studies, base_dir)
    elif args.func == "plot_param_importances":
        plot_files = generate_param_importances_plots(studies, base_dir)
    elif args.func == "plot_parallel_coordinate":
        plot_files = generate_parallel_coordinate_plots(studies, base_dir)
    else:
        assert False, "must not reach here"

    # Render HTML
    env = Environment(loader=FileSystemLoader(template_dirs))
    template = env.get_template("index.html")

    with open(os.path.join(base_dir, "index.html"), "w") as f:
        f.write(template.render(
            funcname=f"{args.func}()",
            plot_files=plot_files)
        )
    print("index.html:", os.path.join(base_dir, "index.html"))


if __name__ == "__main__":
    main()
