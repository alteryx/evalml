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

EvalML is available for Python 3.8 with experimental support 3.9. It can be installed with pip or conda.

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

For CmdStanPy as a backend:
1. `pip install cmdstanpy==0.9.68`
2. `python <path_to_installed_cmdstanpy>/install_cmdstan.py --dir <path_to_build_cmdstan> -v <version_to_use>`
3. `CMDSTAN=<path_to_build_cmdstan>/cmdstan-<version_to_use> STAN_BACKEND=CMDSTANPY pip install prophet==1.0.1`

For PyStan as a backend (PyStan is used by default):
1. `pip install prophet==1.0.1`


## Pip with all dependencies

To install evalml with pip, run the following command:

```bash
pip install evalml
```

## Pip with core dependencies

EvalML includes several optional dependencies. The `xgboost` and `catboost` packages support pipelines built around those modeling libraries. The `plotly` and `ipywidgets` packages support plotting functionality in automl searches. These dependencies are recommended, and are included with EvalML by default but are not required in order to install and use EvalML.

EvalML's core dependencies are listed in `core-requirements.txt` in the source code, and optional requirements are isted in `requirements.txt`.

To install EvalML with only the core required dependencies, download the EvalML source [from pypi](https://pypi.org/project/evalml/#files) to access the requirements files. Then run the following:

```bash
pip install evalml --no-dependencies
pip install -r core-requirements.txt
```

### Add-ons
You can install add-ons individually or all at once by running:
```bash
pip install evalml[complete]
```

**Time Series Support** <br>

Add time series support with Facebook's Prophet
```bash
pip install evalml[prophet]
```
Please note that this may take a few minutes. Prophet is currently only supported via pip installation in EvalML.

**Update checker** <br>

Receive automatic notifications of new EvalML releases
```bash
pip install evalml[update_checker]
```

## Conda with all dependencies

To install evalml with conda run the following command:

```bash
conda install -c conda-forge evalml
```

## Conda with core dependencies 

To install evalml with only core dependencies run the following command:

```bash
conda install -c conda-forge evalml-core
```

## Windows

Additionally, if you are using `pip` to install EvalML, it is recommended you first install the following packages using conda:
* `numba` (needed for `shap` and prediction explanations). Install with `conda install -c conda-forge numba`
* `graphviz` if you're using EvalML's plotting utilities. Install with `conda install -c conda-forge python-graphviz`

The [XGBoost](https://pypi.org/project/xgboost/) library may not be pip-installable in some Windows environments. If you are encountering installation issues, please try installing XGBoost from [Github](https://xgboost.readthedocs.io/en/latest/build.html) before installing EvalML or install evalml with conda.

## Mac

In order to run on Mac, [LightGBM](https://pypi.org/project/lightgbm/) requires the `OpenMP` library to be installed, which can be done with [HomeBrew](https://brew.sh/) by running 

```bash
brew install libomp
```

Additionally, `graphviz` can be installed by running

```bash
brew install graphviz
```

## Python 3.9 support

Evalml can still be installed with pip in python 3.9 but note that `sktime`, one of our dependencies, will not be installed because that library does not yet support python 3.9. This means the ``PolynomialDetrending`` component will not be usable in python 3.9. You can try to install `sktime` [from source](https://www.sktime.org/en/latest/installation.html#building-from-source) in python 3.9 to use the ``PolynomialDetrending`` component but be warned that we only test it in python 3.8.

+++

 
````{tab} Python
```python
print("Hello World!")
```
````

````{tab} C++
```c++
#include <iostream>

int main() {
  std::cout << "Hello World!" << std::endl;
}
```
````
