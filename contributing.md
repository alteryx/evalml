## Contributing to the Codebase

#### 0. Look at Open Issues 
We currently utilize GitHub Issues as our project management tool for EvalML. Please do the following:
* Look at our [open issues](https://github.com/alteryx/evalml/issues)
* Find an unclaimed issue by looking for an empty `Assignees` field.
* If this is your first time contributing, issues labeled ``good first issue`` are a good place to start.
* If your issue is labeled `needs design` or `spike` it is recommended you provide a design document for your feature
  prior to submitting a pull request (PR).
* Connect your PR to your issue by adding the following comment in the PR body: `Fixes #<issue-number>`


#### 1. Clone repo
The code is hosted on GitHub, so you will need to use Git to clone the project and make changes to the codebase. Once you have obtained a copy of the code, you should create a development environment that is separate from your existing Python environment so that you can make and test changes without compromising your own work environment. Additionally, you must make sure that the version of Python you use is at least 3.8. Using `conda` you can use `conda create -n evalml python=3.8` and `conda activate evalml` before the following steps.
* clone with `git clone https://github.com/alteryx/evalml.git`
* install in edit mode with:
    ```bash
    # move into the repo
    cd evalml
    # installs the repo in edit mode, meaning changes to any files will be picked up in python. also installs all dependencies.
    make installdeps-dev
    ```

Note that if you're on Mac, there are a few extra steps you'll want to keep track of.
* In order to run on Mac, [LightGBM requires the OpenMP library to be installed](https://evalml.alteryx.com/en/stable/install.html#Mac), which can be done with HomeBrew by running `brew install libomp`
* We've seen some installs get the following warning when importing evalml: "UserWarning: Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError". [A known workaround](https://stackoverflow.com/a/61531555/841003) is to run `brew reinstall readline xz` before installing the python version you're using via pyenv. If you've already installed a python version in pyenv, consider deleting it and reinstalling. v3.8.2 is known to work.

#### 2. Implement your Pull Request

* Implement your pull request. If needed, add new tests or update the documentation.
* Before submitting to GitHub, verify the tests run and the code lints properly
  ```bash
  # runs linting
  make lint

  # will fix some common linting issues automatically, if the above command failed
  make lint-fix

  # runs all the unit tests locally
  make test
  ```

* If you made changes to the documentation, build the documentation to view locally.
  ```bash
  # go to docs and build
  cd docs
  make html

  # view docs locally
  open build/html/index.html
  ```

* Before you commit, a few lint fixing hooks will run. You can also manually run these.
  ```bash
  # run linting hooks only on changed files
  pre-commit run

  # run linting hooks on all files
  pre-commit run --all-files
  ```

Note that if you're building docs locally, the warning suppression code at `docs/source/disable-warnings.py` will not run, meaning you'll see python warnings appear in the docs where applicable. To suppress this, add `warnings.filterwarnings('ignore')` to `docs/source/conf.py`.

#### 3. Submit your Pull Request

* Once your changes are ready to be submitted, make sure to push your changes to GitHub before creating a pull request. Create a pull request, and our continuous integration will run automatically.

* Be sure to include unit tests (and docstring tests, if applicable) for your changes; these tests you write will also be run as part of the continuous integration.

* If your changes alter the following please fix them as well:
    * Docstrings - if your changes render docstrings invalid
    * API changes - if you change the API update `docs/source/api_reference.rst`
    * Documentation - run the documentation notebooks locally to ensure everything is logical and works as intended

* Update the "Future Release" section at the top of the release notes (`docs/source/release_notes.rst`) to include an entry for your pull request. Write your entry in past tense, i.e. "added fizzbuzz impl."

* Please create your pull request initially as [a "Draft" PR](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/about-pull-requests#draft-pull-requests). This signals the team to ignore it and to allow you to develop. When the checkin tests are passing and you're ready to get your pull request reviewed and merged, please convert it to a normal PR for review.

* We use GitHub Actions to run our PR checkin tests. On creation of the PR and for every change you make to your PR, you'll need a maintainer to click "Approve and run" on your PR. This is a change [GitHub made in April 2021](https://github.blog/2021-04-22-github-actions-update-helping-maintainers-combat-bad-actors/).

* We ask that all contributors sign our contributor license agreement (CLA) the first time they contribute to evalml. The CLA assistant will place a message on your PR; follow the instructions there to sign the CLA.

Add a description of your PR to the subsection that most closely matches your contribution:
    * Enhancements: new features or additions to EvalML.
    * Fixes: things like bugfixes or adding more descriptive error messages.
    * Changes: modifications to an existing part of EvalML.
    * Documentation Changes
    * Testing Changes

If your work includes a [breaking change](https://en.wiktionary.org/wiki/breaking_change), please add a description of what has been affected in the "Breaking Changes" section below the latest release notes. If no "Breaking Changes" section yet exists, please create one as follows. See past release notes for examples of this.
```
.. warning::

    **Breaking Changes**

    * Description of your breaking change
```

### 4. Updating our conda package

We maintain a conda package [package](https://anaconda.org/conda-forge/evalml) to give users more options of how to install EvalML.
Conda packages are created from recipes, which are yaml config files that list a package's dependencies and tests. Here is 
EvalML's latest published [recipe](https://github.com/conda-forge/evalml-core-feedstock/blob/master/recipe/meta.yaml).
GitHub repositories containing conda recipes are called `feedstocks`.

If you opened a PR to EvalML that modifies the packages in `dependencies` within `pyproject.toml`, or if the latest dependency bot
updates the latest version of one of our packages, you will see a CI job called `build_conda_pkg`. This section describes
what `build_conda_pkg` does and what to do if you see it fails in your pr. 

#### What is build_conda_pkg?
`build_conda_pkg` clones the PR branch and builds the conda package from that branch. Since the conda build process runs our
entire suite of unit tests, `build_conda_pkg` checks that our conda package actually supports the proposed change of the PR.
We added this check to eliminate surprises. Since the conda package is released after we release to PyPi, it's possible that
we released a dependency version that is not compatible with our conda recipe. It would be a pain to try to debug this at
release-time since the PyPi release includes many possible PRs that could have introduced that change.

#### How does `build_conda_pkg` work?
`build_conda_pkg` will clone the `master` branch of the feedstock as well as you EvalML PR branch. It will
then replace the recipe in the `master` branch of the feedstock with the current
latest [recipe](https://github.com/alteryx/evalml/blob/make-it-easier-to-fix-build-conda-pkg/.github/meta.yaml) in EvalML.
It will also modify the [source](https://github.com/alteryx/evalml/blob/make-it-easier-to-fix-build-conda-pkg/.github/meta.yaml#L7)
field of the local copy of the recipe and point it at the local EvalML clone of your PR branch.
This has the effect of building our conda package against your PR branch!

#### Why does `build_conda_pkg` use a recipe in EvalML as opposed to the recipe in the feedstock `master` branch?
One important fact to know about conda is that any change to the `master` branch of a feedstock will
result in a new version of the conda package being published to the world!

With this in mind, let's say your PR requires modifying our dependencies. 
If we made a change to `master`, an updated version of EvalML's latest conda package would
be released. This means people who installed the latest version of EvalML prior to this PR would get different dependency versions
than those who installed EvalML after the PR got merged on GitHub. This is not desirable, especially because the PR would not get shipped
to PyPi until the next release happens. So there would also be a discrepancy between the PyPi and conda versions.

By using a recipe stored in the EvalML repo, we can keep track of the changes that need to be made for the next release without
having to publish a new conda package. Since the recipe is also "unique" to your PR, you are free to make whatever changes you
need to make without disturbing other PRs. This would not be the case if `build_conda_pkg` ran from the `master` branch of the
feedstock.

#### What to do if you see `build_conda_pkg` is red on your PR?
It depends on the kind of PR:

**Case 1: You're adding a completely new dependency**

In this case, `build_conda_pkg` is failing simply because a dependency is missing. Adding the dependency to the recipe should
make the check green. To add the dependency, modify the recipe located at `.github/meta.yaml`.  

If you see that adding the dependency causes the build to fail, possibly because of conflicting versions, then iterate until
the build passes. The team will verify if your changes make sense during PR review.

**Case 2: The latest dependency bot created a PR**
If the latest dependency bot PR fails `build_conda_pkg`, it means our code doesn't support the latest version
of one of our dependencies. This means that we either have to cap the max allowed version in our requirements file
or update our code to support that version. If we opt for the former, then just like in Case 1, make the corresponding change
to the recipe located at `.github/meta.yaml`

#### What about the `check_versions` CI check?
This check verifies that the allowed versions listed in `pyproject.toml` match those listed in
the conda recipe so that the PyPi requirements and conda requirements don't get out of sync.

## Code Style Guide

* Keep things simple. Any complexity must be justified in order to pass code review.
* Be aware that while we love fancy python magic, there's usually a simpler solution which is easier to understand!
* Make PRs as small as possible! Consider breaking your large changes into separate PRs. This will make code review easier, quicker, less bug-prone and more effective.
* In the name of every branch you create, include the associated issue number if applicable.
* If new changes are added to the branch you're basing your changes off of, consider using `git rebase -i base_branch` rather than merging the base branch, to keep history clean.
* Always include a docstring for public methods and classes. Consider including docstrings for private methods too. We use the [Google docstring convention](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings), and use the [`sphinx.ext.napoleon`](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) extension to parse our docstrings.
* Although not explicitly enforced by the Google convention, keep the following stylistic conventions for docstrings in mind:
    - First letter of each argument description should be capitalized.
    - Docstring sentences should end in periods. This includes descriptions for each argument.
    - Types should be written in lower-case. For example, use "bool" instead of "Bool".
    - Always add the default value in the description of the argument, if applicable. For example, "Defaults to 1."
* Use [PascalCase (upper camel case)](https://en.wikipedia.org/wiki/Camel_case#Variations_and_synonyms) for class names, and [snake_case](https://en.wikipedia.org/wiki/Snake_case) for method and class member names.
* To distinguish private methods and class attributes from public ones, those which are private should be prefixed with an underscore
* Any code which doesn't need to be public should be private. Use `@staticmethod` and `@classmethod` where applicable, to indicate no side effects.
* Only call public methods in unit tests.
* All code must have unit test coverage. Use mocking and monkey-patching when necessary.
* Keep unit tests as fast as possible. In particular, avoid calling `fit`. Mocking can help with this.
* When you're working with code which uses a random number generator, make sure your unit tests set a random seed.
* Use `np.testing.assert_almost_equal` when comparing floating-point numbers, to avoid numerical precision issues, particularly cross-platform.
* Use `os.path` tools to keep file paths cross-platform.
* Our rule of thumb is to favor traditional inheritance over a mixin pattern.

## GitHub Issue Guide

* Make the title as short and descriptive as possible.
* Make sure the body is concise and gets to the point quickly.
* Check for duplicates before filing.
* For bugs, a good general outline is: problem summary, reproduction steps, symptoms and scope, root cause if known, proposed solution(s), and next steps.
* If the issue writeup or conversation get too long and hard to follow, consider starting a design document.
* Use the appropriate labels to help your issue get triaged quickly.
* Make your issues as actionable as possible. If they track open discussions, consider prefixing the title with "[Discuss]", or refining the issue further before filing.
