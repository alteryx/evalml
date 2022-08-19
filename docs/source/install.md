---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Install

EvalML is available for Python 3.8 and 3.9. It can be installed from [pypi](https://pypi.org/project/evalml/), [conda-forge](https://anaconda.org/conda-forge/evalml), or from [source](https://github.com/alteryx/evalml).

To install EvalML on your platform, run one of the following commands:

````{tab} PyPI
```console
$ pip install evalml
```
````

````{tab} Conda
```console
$ conda install -c conda-forge evalml
```
````

````{tab} PyPI (M1 Mac)
```console
# See the EvalML with core dependencies only section
$ pip install evalml --no-dependencies
$ pip install -r core-requirements.txt
```
````

````{tab} Conda (M1 Mac)
```console
# See the EvalML with core dependencies only section
$ conda install -c conda-forge evalml-core
```
````

## EvalML with core dependencies only

EvalML includes several optional dependencies. The `xgboost` and `catboost` packages support pipelines built around those modeling libraries. The `plotly` and `ipywidgets` packages support plotting functionality in automl searches. These dependencies are recommended, and are included with EvalML by default but are not required in order to install and use EvalML.

EvalML's core dependencies are listed in `core-requirements.txt` in the source code, while the default collection of requirements is specified in `setup.cfg`'s `install_requires`.

To install EvalML with only the core-required dependencies with pypi, first download the EvalML source [from pypi](https://pypi.org/project/evalml/#files) or [github](https://github.com/alteryx/evalml) to access the requirements files before running the following command.

````{tab} PyPI
```console
$ pip install evalml --no-dependencies
$ pip install -r core-requirements.txt
```
````

````{tab} Conda
```console
$ conda install -c conda-forge evalml-core
```
````

## Add-ons

EvalML allows users to install add-ons individually or all at once:

- **Update Checker**: Receive automatic notifications of new EvalML releases
- **Time Series**: Use EvalML with Facebook's Prophet library for time series support.

````{tab} PyPI
```{tab} All Add-ons
```console
$ pip install evalml[complete]
```
```{tab} Time Series Support
```console
$ pip install evalml[prophet]
```
```{tab} Update Checker
```console
$ pip install evalml[update_checker]
```
````
````{tab} Conda
```{tab} Update Checker
```console
$ conda install -c conda-forge alteryx-open-src-update-checker
```
````

## Time Series support with Facebook's Prophet

To support the `Prophet` time series estimator, be sure to install it as an extra requirement. Please note that this may take a few minutes.
Prophet is currently only supported via pip installation in EvalML for Mac with CmdStan as a backend.
```shell
pip install evalml[prophet]
```
Another option for installing Prophet with CmdStan as a backend is to use `make installdeps-prophet`.

Note: In order to do this, you must have the EvalML repo cloned and you must be in the top level folder `<your_directory>/evalml/` to execute this command.
This command will do the following:
- Pip install `cmdstanpy==0.9.68`
- Execute the `install_cmdstan.py` script found within your `site-packages/cmdstanpy` which builds `cmdstan` in your `site-packages`.
- Install `Prophet==1.0.1` with the `CMDSTAN` and `STAN_BACKEND` environment variables set.

If the `site-packages` path is incorrect or you'd like to specify a different one, just run `make installdeps-prophet SITE_PACKAGES_DIR="<path_to_your_site_packages>"`.

If you'd like to have more fine-tuned control over the installation steps for Prophet, such as specifying the backend, follow these steps:

````{tab} PyStan (default)
```console
$ pip install prophet==1.0.1
```
````
````{tab} CmdStanPy backend
```console
$ pip install cmdstanpy==0.9.68
$ python <path_to_installed_cmdstanpy>/install_cmdstan.py --dir <path_to_build_cmdstan> -v <version_to_use>
$ CMDSTAN=<path_to_build_cmdstan>/cmdstan-<version_to_use> STAN_BACKEND=CMDSTANPY pip install prophet==1.0.1

```
````

## Windows Additional Requirements & Troubleshooting

If you are using `pip` to install EvalML on Windows, it is recommended you first install the following packages using conda:
* `numba` (needed for `shap` and prediction explanations). Install with `conda install -c conda-forge numba`
* `graphviz` if you're using EvalML's plotting utilities. Install with `conda install -c conda-forge python-graphviz`

The [XGBoost](https://pypi.org/project/xgboost/) library may not be pip-installable in some Windows environments. If you are encountering installation issues, please try installing XGBoost from [Github](https://xgboost.readthedocs.io/en/latest/build.html) before installing EvalML or install evalml with conda.

## Mac Additional Requirements & Troubleshooting

In order to run on Mac, [LightGBM](https://pypi.org/project/lightgbm/) requires the `OpenMP` library to be installed, which can be done with [HomeBrew](https://brew.sh/) by running:

```bash
brew install libomp
```

Additionally, `graphviz` can be installed by running:

```bash
brew install graphviz
```

### Installing EvalML on an M1 Mac

Not all of EvalML's dependencies support Apple's new M1 chip. For this reason, `pip` or `conda` installing EvalML will
fail. The core set of EvalML dependencies can be installed in the M1 chip, so we recommend you install EvalML with core
dependencies.

Alternatively, there is experimental support for M1 chips with the Rosetta terminal. After setting up a Rosetta terminal, you should be able to `pip` or `conda` install EvalML.

For Docker fans, an included `Dockerfile.arm` can be built and run to provide an environment for testing.  Details are included within.

+++
