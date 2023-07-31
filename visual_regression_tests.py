import argparse
import functools
import logging
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
import warnings

from jinja2 import Environment
from jinja2 import FileSystemLoader
import matplotlib.pylab as plt
import optuna
from optuna import Study
from optuna.exceptions import ExperimentalWarning

from studies import create_intermediate_value_studies
from studies import create_multi_objective_studies
from studies import create_multiple_single_objective_studies
from studies import create_pytorch_study
from studies import create_single_objective_studies
from studies import create_single_objective_studies_for_timeline
from studies import create_terminator_studies
from studies import create_terminator_studies_with_validation_score
from studies import StudiesType
import optuna.visualization as plotly_visualization
import optuna.visualization.matplotlib as matplotlib_visualization


FigureType = TypeVar("FigureType")

try:
    from optuna_fast_fanova import FanovaImportanceEvaluator
except ImportError:
    from optuna.importance import FanovaImportanceEvaluator

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    import plotly.graph_objs as go

warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", help="output directory (default: %(default)s)", default="tmp")
parser.add_argument("--width", help="plot width (default: %(default)s)", type=int, default=800)
parser.add_argument("--height", help="plot height (default: %(default)s)", type=int, default=600)
parser.add_argument("--heavy", help="create studies that takes long time", action="store_true")
args = parser.parse_args()

template_dirs = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")]
abs_output_dir = os.path.abspath(args.output_dir)
dpi = 100
plt.rcParams["figure.figsize"] = (args.width / dpi, args.height / dpi)


def wrap_plot_func(
    plot_func: Callable[[Study], FigureType]
) -> Callable[[StudiesType], FigureType]:
    @functools.wraps(plot_func)
    def wrapper(studies: StudiesType) -> FigureType:
        assert isinstance(studies, Study)
        return plot_func(studies)

    return wrapper


def stringify_plot_kwargs(plot_kwargs: Dict[str, Any]) -> str:
    items = []
    for key, value in plot_kwargs.items():
        items.append(f"{key}={value}")
    return ",".join(items)


def generate_plot_files(
    studies: List[Tuple[str, StudiesType]],
    base_dir: str,
    plotly_plot: Callable[[StudiesType], "go.Figure"],
    matplotlib_plot: Callable[[StudiesType], "Axes"],
    filename_prefix: str,
) -> List[Tuple[str, str, str]]:
    files = []
    for title, study in studies:
        plotly_filename = f"{filename_prefix}-{title}-plotly.png"
        plotly_filepath = os.path.join(base_dir, plotly_filename)
        matplotlib_filename = f"{filename_prefix}-{title}-matplotlib.png"
        matplotlib_filepath = os.path.join(base_dir, matplotlib_filename)
        try:
            plotly_fig = plotly_plot(study)
            plotly_fig.update_layout(
                width=args.width,
                height=args.height,
                margin={"l": 10, "r": 10},
            )
            plotly_fig.write_image(plotly_filepath)
        except Exception as e:  # NOQA
            logging.exception(f"Exception occurred while executing plotly visualization function")
            plotly_filename = ""

        try:
            matplotlib_plot(study)
            plt.savefig(matplotlib_filepath, bbox_inches="tight", dpi=dpi)
        except Exception as e:  # NOQA
            logging.exception(f"Exception occurred while executing matplotlib visualization function")
            matplotlib_filename = ""

        files.append((title, plotly_filename, matplotlib_filename))
    return files


def generate_optimization_history_plots(
    studies: List[Tuple[str, StudiesType]],
    base_dir: str,
    plot_kwargs: Dict[str, Any],
) -> List[Tuple[str, str, str]]:
    filename_prefix = "history"
    if len(plot_kwargs) > 0:
        filename_prefix = f"{filename_prefix}-{stringify_plot_kwargs(plot_kwargs)}"
    return generate_plot_files(
        studies,
        base_dir,
        lambda s: plotly_visualization.plot_optimization_history(s, **plot_kwargs),
        lambda s: matplotlib_visualization.plot_optimization_history(s, **plot_kwargs),
        filename_prefix=filename_prefix,
    )


def generate_contour_plots(
    studies: List[Tuple[str, StudiesType]], base_dir: str, plot_kwargs: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    filename_prefix = "contour"
    if len(plot_kwargs) > 0:
        filename_prefix = f"{filename_prefix}-{stringify_plot_kwargs(plot_kwargs)}"
    return generate_plot_files(
        studies,
        base_dir,
        wrap_plot_func(lambda s: plotly_visualization.plot_contour(s, **plot_kwargs)),
        wrap_plot_func(lambda s: matplotlib_visualization.plot_contour(s, **plot_kwargs)),
        filename_prefix=filename_prefix,
    )


def generate_edf_plots(
    studies: List[Tuple[str, StudiesType]], base_dir: str, plot_kwargs: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    filename_prefix = "edf"
    if len(plot_kwargs) > 0:
        filename_prefix = f"{filename_prefix}-{stringify_plot_kwargs(plot_kwargs)}"
    return generate_plot_files(
        studies,
        base_dir,
        lambda s: plotly_visualization.plot_edf(s, **plot_kwargs),
        lambda s: matplotlib_visualization.plot_edf(s, **plot_kwargs),
        filename_prefix=filename_prefix,
    )


def generate_slice_plots(
    studies: List[Tuple[str, StudiesType]], base_dir: str, plot_kwargs: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    filename_prefix = "slice"
    if len(plot_kwargs) > 0:
        filename_prefix = f"{filename_prefix}-{stringify_plot_kwargs(plot_kwargs)}"
    return generate_plot_files(
        studies,
        base_dir,
        wrap_plot_func(lambda s: plotly_visualization.plot_slice(s, **plot_kwargs)),
        wrap_plot_func(lambda s: matplotlib_visualization.plot_slice(s, **plot_kwargs)),
        filename_prefix=filename_prefix,
    )


def generate_param_importances_plots(
    studies: List[Tuple[str, StudiesType]], base_dir: str, plot_kwargs: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    seed = 0
    filename_prefix = "importance"
    if len(plot_kwargs) > 0:
        filename_prefix = f"{filename_prefix}-{stringify_plot_kwargs(plot_kwargs)}"
    return generate_plot_files(
        studies,
        base_dir,
        wrap_plot_func(
            lambda s: plotly_visualization.plot_param_importances(
                s, evaluator=FanovaImportanceEvaluator(seed=seed), **plot_kwargs
            )
        ),
        wrap_plot_func(
            lambda s: matplotlib_visualization.plot_param_importances(
                s, evaluator=FanovaImportanceEvaluator(seed=seed), **plot_kwargs
            )
        ),
        filename_prefix=filename_prefix,
    )


def generate_parallel_coordinate_plots(
    studies: List[Tuple[str, StudiesType]], base_dir: str, plot_kwargs: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    filename_prefix = "parcoords"
    if len(plot_kwargs) > 0:
        filename_prefix = f"{filename_prefix}-{stringify_plot_kwargs(plot_kwargs)}"
    return generate_plot_files(
        studies,
        base_dir,
        wrap_plot_func(lambda s: plotly_visualization.plot_parallel_coordinate(s, **plot_kwargs)),
        wrap_plot_func(
            lambda s: matplotlib_visualization.plot_parallel_coordinate(s, **plot_kwargs)
        ),
        filename_prefix=filename_prefix,
    )


def generate_intermediate_value_plots(
    studies: List[Tuple[str, StudiesType]], base_dir: str, plot_kwargs: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    filename_prefix = "intermediate"
    if len(plot_kwargs) > 0:
        filename_prefix = f"{filename_prefix}-{stringify_plot_kwargs(plot_kwargs)}"
    return generate_plot_files(
        studies,
        base_dir,
        wrap_plot_func(lambda s: plotly_visualization.plot_intermediate_values(s, **plot_kwargs)),
        wrap_plot_func(
            lambda s: matplotlib_visualization.plot_intermediate_values(s, **plot_kwargs)
        ),
        filename_prefix=filename_prefix,
    )


def generate_pareto_front_plots(
    studies: List[Tuple[str, StudiesType]], base_dir: str, plot_kwargs: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    filename_prefix = "pareto-front"
    if len(plot_kwargs) > 0:
        filename_prefix = f"{filename_prefix}-{stringify_plot_kwargs(plot_kwargs)}"
    return generate_plot_files(
        studies,
        base_dir,
        wrap_plot_func(lambda s: plotly_visualization.plot_pareto_front(s, **plot_kwargs)),
        wrap_plot_func(lambda s: matplotlib_visualization.plot_pareto_front(s, **plot_kwargs)),
        filename_prefix=filename_prefix,
    )


def generate_timeline_plots(
    studies: List[Tuple[str, StudiesType]], base_dir: str, plot_kwargs: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    filename_prefix = "timeline"
    if len(plot_kwargs) > 0:
        filename_prefix = f"{filename_prefix}-{stringify_plot_kwargs(plot_kwargs)}"
    return generate_plot_files(
        studies,
        base_dir,
        wrap_plot_func(lambda s: plotly_visualization.plot_timeline(s, **plot_kwargs)),
        wrap_plot_func(lambda s: matplotlib_visualization.plot_timeline(s, **plot_kwargs)),
        filename_prefix=filename_prefix,
    )


def generate_terminator_plots(
    studies: List[Tuple[str, StudiesType]], base_dir: str, plot_kwargs: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    filename_prefix = "terminator-improvement"
    if len(plot_kwargs) > 0:
        filename_prefix = f"{filename_prefix}-{stringify_plot_kwargs(plot_kwargs)}"
    return generate_plot_files(
        studies,
        base_dir,
        wrap_plot_func(lambda s: plotly_visualization.plot_terminator_improvement(s, **plot_kwargs)),
        wrap_plot_func(lambda s: matplotlib_visualization.plot_terminator_improvement(s, **plot_kwargs)),
        filename_prefix=filename_prefix,
    )


def main() -> None:
    if not os.path.exists(abs_output_dir):
        os.mkdir(abs_output_dir)

    env = Environment(loader=FileSystemLoader(template_dirs))
    plot_results_template = env.get_template("plot_results.html")
    list_pages_template = env.get_template("list_pages.html")

    print("Creating single objective studies")
    single_objective_studies = create_single_objective_studies()
    multiple_single_objective_studies = create_multiple_single_objective_studies()
    print("Creating multi objective studies")
    multi_objective_studies = create_multi_objective_studies()
    print("Creating studies that have intermediate values")
    intermediate_value_studies = create_intermediate_value_studies()
    print("Creating studies for timeline plot")
    single_objective_studies_for_timeline = create_single_objective_studies_for_timeline()
    print("Creating studies for optuna terminator")
    terminator_studies = create_terminator_studies()
    terminator_studies_with_val = create_terminator_studies_with_validation_score()

    if args.heavy:
        print("Creating pytorch study")
        pytorch_study = create_pytorch_study()
        assert pytorch_study is not None
        single_objective_studies.append((pytorch_study.study_name, pytorch_study))
        intermediate_value_studies.insert(0, (pytorch_study.study_name, pytorch_study))

    single_objective_studies_with_multi_studies: List[Tuple[str, StudiesType]] = [
        *single_objective_studies,
        *multiple_single_objective_studies,
    ]

    pages: List[Tuple[str, str]] = []
    for funcname, studies, generate, plot_kwargs in [
        (
            "plot_optimization_history",
            single_objective_studies_with_multi_studies,
            generate_optimization_history_plots,
            {},
        ),
        (
            "plot_optimization_history",
            multiple_single_objective_studies,
            generate_optimization_history_plots,
            {"error_bar": True},
        ),
        ("plot_slice", single_objective_studies, generate_slice_plots, {}),
        ("plot_contour", single_objective_studies, generate_contour_plots, {}),
        (
            "plot_parallel_coordinate",
            single_objective_studies,
            generate_parallel_coordinate_plots,
            {},
        ),
        (
            "plot_intermediate_values",
            intermediate_value_studies,
            generate_intermediate_value_plots,
            {},
        ),
        ("plot_pareto_front", multi_objective_studies, generate_pareto_front_plots, {}),
        ("plot_param_importances", single_objective_studies, generate_param_importances_plots, {}),
        (
            "plot_edf",
            single_objective_studies_with_multi_studies,
            generate_edf_plots,
            {},
        ),
        ("plot_timeline", single_objective_studies_for_timeline, generate_timeline_plots, {}),
        ("plot_terminator_improvement", terminator_studies, generate_terminator_plots, {}),
        ("plot_terminator_improvement", terminator_studies_with_val, generate_terminator_plots, {"plot_error": True}),
    ]:
        assert isinstance(plot_kwargs, Dict)
        plot_files = generate(studies, abs_output_dir, plot_kwargs)

        plot_kwargs_str = stringify_plot_kwargs(plot_kwargs)
        filename = (
            f"{funcname}.html"
            if len(plot_kwargs_str) == 0
            else f"{funcname}-{plot_kwargs_str}.html"
        )

        with open(os.path.join(abs_output_dir, filename), "w") as f:
            f.write(
                plot_results_template.render(
                    funcname=f"{funcname}({plot_kwargs_str})", plot_files=plot_files
                )
            )

        pages.append((f"{funcname}({plot_kwargs_str})", filename))

    with open(os.path.join(abs_output_dir, "index.html"), "w") as f:
        f.write(list_pages_template.render(pages=pages))

    print("Generated to:", os.path.join(abs_output_dir, "index.html"))


if __name__ == "__main__":
    main()
