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

from studies import create_single_objective_studies

try:
    from optuna_fast_fanova import FanovaImportanceEvaluator
except ImportError:
    from optuna.importance import FanovaImportanceEvaluator

warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)

template_dirs = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")]
plot_functions = [
    "plot_contour",
    "plot_edf",
    "plot_optimization_history",
    "plot_param_importances",
    "plot_parallel_coordinate",
    "plot_slice",
]

parser = argparse.ArgumentParser()
parser.add_argument("func", help="plot function name", choices=plot_functions)
parser.add_argument("--output-dir", help="output directory (default: %(default)s)", default="tmp")
parser.add_argument("--width", help="plot width (default: %(default)s)", type=int, default=800)
parser.add_argument("--height", help="plot height (default: %(default)s)", type=int, default=600)
args = parser.parse_args()

dpi = 100
plt.rcParams["figure.figsize"] = (args.width / dpi, args.height / dpi)


def generate_optimization_history_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    files = []
    for study in studies:
        plotly_filepath = os.path.join(base_dir, f"{study.study_name}-plotly.png")
        plotly_fig = plotly_visualization.plot_optimization_history(study)
        plotly_fig.update_layout(
            width=args.width,
            height=args.height,
            margin={"l": 10, "r": 10}
        )
        plotly_fig.write_image(plotly_filepath)

        matplotlib_filepath = os.path.join(
            base_dir, f"{study.study_name}-matplotlib.png"
        )
        matplotlib_visualization.plot_optimization_history(study)
        plt.savefig(matplotlib_filepath, bbox_inches="tight", dpi=dpi)

        files.append((study, plotly_filepath, matplotlib_filepath))
    return files


def generate_contour_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    files = []
    for study in studies:
        plotly_filepath = os.path.join(
            base_dir, f"{study._study_id}-{study.study_name}-plotly.png"
        )
        try:
            plotly_fig = plotly_visualization.plot_contour(study)
            plotly_fig.update_layout(
                width=args.width,
                height=args.height,
                margin={"l": 10, "r": 10},
            )
            plotly_fig.write_image(plotly_filepath)
        except:
            plotly_fig = ""

        matplotlib_filepath = os.path.join(
            base_dir, f"{study._study_id}-{study.study_name}-matplotlib.png"
        )
        try:
            matplotlib_visualization.plot_contour(study)
            plt.savefig(matplotlib_filepath, bbox_inches="tight", dpi=dpi)
        except:
            matplotlib_filepath = ""

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
            width=args.width,
            height=args.height,
            margin={"l": 10, "r": 10}
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
            width=args.width,
            height=args.height,
            margin={"l": 10, "r": 10}
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
        plotly_fig = plotly_visualization.plot_param_importances(
            study, evaluator=FanovaImportanceEvaluator(seed=seed)
        )
        plotly_fig.update_layout(
            width=args.width,
            height=args.height,
            margin={"l": 10, "r": 10}
        )
        plotly_fig.write_image(plotly_filepath)

        matplotlib_filepath = os.path.join(
            base_dir, f"{study.study_name}-matplotlib.png"
        )
        matplotlib_visualization.plot_param_importances(
            study, evaluator=FanovaImportanceEvaluator(seed=seed)
        )
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
            width=args.width,
            height=args.height,
            margin={"l": 10, "r": 10}
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
    base_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    if args.func == "plot_optimization_history":
        studies = create_single_objective_studies()
        plot_files = generate_optimization_history_plots(studies, base_dir)
    elif args.func == "plot_contour":
        studies = create_single_objective_studies()
        plot_files = generate_contour_plots(studies, base_dir)
    elif args.func == "plot_edf":
        studies = create_single_objective_studies()
        plot_files = generate_edf_plots(studies, base_dir)
    elif args.func == "plot_slice":
        studies = create_single_objective_studies()
        plot_files = generate_slice_plots(studies, base_dir)
    elif args.func == "plot_param_importances":
        studies = create_single_objective_studies()
        plot_files = generate_param_importances_plots(studies, base_dir)
    elif args.func == "plot_parallel_coordinate":
        studies = create_single_objective_studies()
        plot_files = generate_parallel_coordinate_plots(studies, base_dir)
    else:
        assert False, "must not reach here"

    # Render HTML
    env = Environment(loader=FileSystemLoader(template_dirs))
    template = env.get_template("index.html")

    with open(os.path.join(base_dir, "index.html"), "w") as f:
        f.write(template.render(funcname=f"{args.func}()", plot_files=plot_files))
    print("index.html:", os.path.join(base_dir, "index.html"))


if __name__ == "__main__":
    main()
