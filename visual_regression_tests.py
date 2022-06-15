import argparse
import os
from os.path import relpath
from typing import Callable
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING
import warnings

from jinja2 import Environment
from jinja2 import FileSystemLoader
import matplotlib.pylab as plt
import optuna
from optuna import Study
from optuna.exceptions import ExperimentalWarning
import optuna.visualization as plotly_visualization
import optuna.visualization.matplotlib as matplotlib_visualization

from studies import create_intermediate_value_studies
from studies import create_multi_objective_studies
from studies import create_single_objective_studies


try:
    from optuna_fast_fanova import FanovaImportanceEvaluator
except ImportError:
    from optuna.importance import FanovaImportanceEvaluator

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    import plotly.graph_objs as go

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
    "plot_intermediate_values",
    "plot_pareto_front",
]

parser = argparse.ArgumentParser()
parser.add_argument("func", help="plot function name", choices=plot_functions)
parser.add_argument("--output-dir", help="output directory (default: %(default)s)", default="tmp")
parser.add_argument("--width", help="plot width (default: %(default)s)", type=int, default=800)
parser.add_argument("--height", help="plot height (default: %(default)s)", type=int, default=600)
parser.add_argument("--relpath", help="Use relative path for img link", action='store_true')
args = parser.parse_args()

dpi = 100
plt.rcParams["figure.figsize"] = (args.width / dpi, args.height / dpi)


def generate_plot_files(
    studies: List[Study],
    base_dir: str,
    plotly_plot: Callable[[Study], "go.Figure"],
    matplotlib_plot: Callable[[Study], "Axes"],
) -> List[Tuple[Study, str, str]]:
    files = []
    for study in studies:
        plotly_filepath = os.path.join(
            base_dir, f"{study._study_id}-{study.study_name}-plotly.png"
        )
        try:
            plotly_fig = plotly_plot(study)
            plotly_fig.update_layout(
                width=args.width,
                height=args.height,
                margin={"l": 10, "r": 10},
            )
            plotly_fig.write_image(plotly_filepath)
        except:  # NOQA
            plotly_filepath = ""

        matplotlib_filepath = os.path.join(
            base_dir, f"{study._study_id}-{study.study_name}-matplotlib.png"
        )
        try:
            matplotlib_plot(study)
            plt.savefig(matplotlib_filepath, bbox_inches="tight", dpi=dpi)
        except:  # NOQA
            matplotlib_filepath = ""

        files.append((study, plotly_filepath, matplotlib_filepath))
    return files


def generate_optimization_history_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_optimization_history,
        matplotlib_visualization.plot_optimization_history,
    )


def generate_contour_plots(studies: List[Study], base_dir: str) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_contour,
        matplotlib_visualization.plot_contour,
    )


def generate_edf_plots(studies: List[Study], base_dir: str) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_edf,
        matplotlib_visualization.plot_edf,
    )


def generate_slice_plots(studies: List[Study], base_dir: str) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_slice,
        matplotlib_visualization.plot_slice,
    )


def generate_param_importances_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    seed = 0
    return generate_plot_files(
        studies,
        base_dir,
        lambda s: plotly_visualization.plot_param_importances(
            s, evaluator=FanovaImportanceEvaluator(seed=seed)
        ),
        lambda s: matplotlib_visualization.plot_slice(
            s, evaluator=FanovaImportanceEvaluator(seed=seed)
        ),
    )


def generate_parallel_coordinate_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_parallel_coordinate,
        matplotlib_visualization.plot_parallel_coordinate,
    )


def generate_intermediate_value_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_intermediate_values,
        matplotlib_visualization.plot_intermediate_values,
    )


def generate_pareto_front_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_pareto_front,
        matplotlib_visualization.plot_pareto_front,
    )


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
    elif args.func == "plot_intermediate_values":
        studies = create_intermediate_value_studies()
        plot_files = generate_intermediate_value_plots(studies, base_dir)
    elif args.func == "plot_pareto_front":
        studies = create_multi_objective_studies()
        plot_files = generate_pareto_front_plots(studies, base_dir)
    else:
        assert False, "must not reach here"

    # Render HTML
    env = Environment(loader=FileSystemLoader(template_dirs))
    template = env.get_template("index.html")

    if args.relpath:
        plot_files = [(study, relpath(file1, base_dir), relpath(file2, base_dir)) for study, file1, file2 in plot_files]

    with open(os.path.join(base_dir, "index.html"), "w") as f:
        f.write(template.render(funcname=f"{args.func}()", plot_files=plot_files))

    if args.relpath:
        print("Output Directory:", base_dir)
        print("Please above command to check the output.")
        print("")
        print(f"  python3 -m http.server --directory {base_dir}")
        print("")
    else:
        print("index.html:", os.path.join(base_dir, "index.html"))


if __name__ == "__main__":
    main()
