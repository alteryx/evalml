## Contributing to the Codebase

#### 0. Access ZenHub
We currently utilize ZenHub as our project management tool for EvalML. Please do the following:
* Access ZenHub directly through GitHub (using the [extension](https://www.zenhub.com/extension)) or [here](https://app.zenhub.com/)
* Be assigned or assign yourself work from the `Sprint Backlog` and then `Development Backlog`
* Connect your PR to your issue so it behaves as one
    * Hit `Connect this pull request with an existing issue` at the bottom of the PR and attach the issue
* Move your issue to the correct pipeline(column)
    * In Progress for issues in progress (including work after review)
    * Review/QA when needing review or QA
    * Closed when finished

More details about ZenHub and its best practices can be found [here](https://bit.ly/379iFB9).


#### 1. Clone repo
The code is hosted on GitHub, so you will need to use Git to clone the project and make changes to the codebase. Once you have obtained a copy of the code, you should create a development environment that is separate from your existing Python environment so that you can make and test changes without compromising your own work environment.
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

#### 3. Submit your Pull Request

* Once your changes are ready to be submitted, make sure to push your changes to GitHub before creating a pull request. Create a pull request, and our continuous integration will run automatically.

* Be sure to include unit tests for your changes; the unit tests you write will also be run as part of the continuous integration.

* If your changes alter the following please fix them as well:
    * Docstrings - if your changes render docstrings invalid
    * API changes - if you change the API update `docs/source/api_reference.rst`
    * Documentation - run the documentation notebooks locally to ensure everything is logical and works as intended

* Update the "Future Release" section at the top of the release notes (`docs/source/release_notes.rst`) to include an entry for your pull request. Write your entry in past tense, i.e. "added fizzbuzz impl."

* Please create your pull request initially as [a "Draft" PR](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/about-pull-requests#draft-pull-requests). This signals the team to ignore it and to allow you to develop. When the checkin tests are passing and you're ready to get your pull request reviewed and merged, please convert it to a normal PR for review.

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

## Code Style Guide

* Keep things simple. Any complexity must be justified in order to pass code review.
* Be aware that while we love fancy python magic, there's usually a simpler solution which is easier to understand!
* Make PRs as small as possible! Consider breaking your large changes into separate PRs. This will make code review easier, quicker, less bug-prone and more effective.
* In the name of every branch you create, include the associated issue number if applicable.
* If new changes are added to the branch you're basing your changes off of, consider using `git rebase -i base_branch` rather than merging the base branch, to keep history clean.
* Always include a docstring for public methods and classes. Consider including docstrings for private methods too. Our docstring convention is [`sphinx.ext.napoleon`](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html).
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
