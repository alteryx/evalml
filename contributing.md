## Contributing to the Codebase

#### 0. Access ZenHub
We currently utilize ZenHub as our project management tool for EvalML. Please follow the following:
* Access ZenHub directly through GitHub (using the [extension](https://www.zenhub.com/extension)) or [here](https://app.zenhub.com/)
* Be assigned or assign yourself work from the `Sprint Backlog` and then `Development Backlog`
* Move your issue to the correct pipeline(column)
    * In Progress for issues in progress (including work after review)
    * Review/QA when needing review or QA
    * Closed when finished

More details about ZenHub and it's best practices can be found [here](https://bit.ly/379iFB9).


#### 1. Clone repo
The code is hosted on GitHub, so you will need to use Git to clone the project and make changes to the codebase. Once you have obtained a copy of the code, you should create a development environment that is separate from your existing Python environment so that you can make and test changes without compromising your own work environment.
* clone with `git clone https://github.com/FeatureLabs/evalml.git`
* install in edit mode with:
    ```bash
    cd evalml  # move to directory
    pip install -e . # install in edit mode
    ```


#### 2. Implement your Pull Request

* Implement your pull request. If needed, add new tests or update the documentation.
* Before submitting to GitHub, verify the tests run and the code lints properly
  ```bash
  # runs test
  make test

  # runs linting
  make lint

  # will fix some common linting issues automatically
  make lint-fix
  ```
* If you made changes to the documentation, build the documentation locally.
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

* Update the "Future Release" section of the changelog (`docs/source/changelog.rst`) to include your pull request and add your github username to the list of contributors.  Add a description of your PR to the subsection that most closely matches your contribution:
    * Enhancements: new features or additions to Featuretools.
    * Fixes: things like bugfixes or adding more descriptive error messages.
    * Changes: modifications to an existing part of Featuretools.
    * Documentation Changes
    * Testing Changes

