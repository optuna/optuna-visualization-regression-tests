# optuna-visualization-regression-tests

This tool continuously runs on GitHub Actions with Optuna's master branch.
The output artifact is pushed to gh-pages branch.
You can check the output from the below link.

https://c-bata.link/optuna-visualization-regression-tests/index.html

## Usage

```
$ pip install -r requirements.txt
$ python visual_regression_tests.py --output-dir /tmp/foo --heavy
Generated to: /tmp/foo/index.html
```

## Example Output

### `plot_optimization_history()`

<img width="1715" alt="plot_optimization_history" src="https://user-images.githubusercontent.com/5564044/173838248-8f9599ec-c7cb-482c-9fd3-be6c856520d0.png">

### `plot_slice()`

<img width="1715" alt="plot_slice" src="https://user-images.githubusercontent.com/5564044/173838319-24433136-bd59-47d5-afdb-2694aafe354d.png">

### `plot_pareto_front()`

<img width="1715" alt="plot_pareto_front" src="https://user-images.githubusercontent.com/5564044/173838426-34f87856-411f-41e9-8e0f-dee5b6329a45.png">

### `plot_contour()`

<img width="1715" alt="plot_contour" src="https://user-images.githubusercontent.com/5564044/173839582-b22da40d-34f0-40c6-baf9-c2e541b7f9a7.png">

### `plot_edf()`

### `plot_intermediate_value()`

### `plot_parallel_coordinate()`

### `plot_param_importances()`

