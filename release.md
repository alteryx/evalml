# Release Process

## 0. Pre-Release Checklist
Before starting the release process, verify the following:
* All work required for this release has been completed and the team is ready to release.
* [All CircleCI tests are green on main](https://app.circleci.com/pipelines/github/FeatureLabs/evalml?branch=main).
* The [ReadtheDocs build](https://readthedocs.com/projects/feature-labs-inc-evalml/builds/) for "latest" is marked as passed. To avoid mysterious errors, best practice is to empty your browser cache when reading new versions of the docs!
* The [public documentation for the "latest" branch](https://evalml.featurelabs.com/en/latest/) looks correct, and the [release notes](https://evalml.featurelabs.com/en/latest/release_notes.html) includes the last change which was made on main.
* The [performance tests](https://github.com/FeatureLabs/evalml_looking_glass) have passed on latest main, and the team has reviewed the results.
* Get agreement on the version number to use for the release.

#### Version Numbering

EvalML uses [semantic versioning](https://semver.org/). Every release has a major, minor and patch version number, and are displayed like so: `<majorVersion>.<minorVersion>.<patchVersion>`.

If you'd like to create a development release, which won't be deployed to pypi and conda and marked as a generally-available production release, please add a "dev" prefix to the patch version, i.e. `X.X.devX`. Note this claims the patch number--if the previous release was `0.12.0`, a subsequent dev release would be `0.12.dev1`, and the following release would be `0.12.2`, *not* `0.12.1`. Development releases deploy to [test.pypi.org](https://test.pypi.org/project/evalml/) instead of to [pypi.org](https://pypi.org/project/evalml).

## 1. Test conda version before releasing on PyPI
Conda releases of evalml rely on PyPI's hosted evalml packages. Once a version is uploaded to PyPI we cannot update it, so it is important that the version we upload to PyPI will work for conda.  We can test if an evalml release will run on conda by uploading a test release to PyPI's test server and building a conda version of evalml using the test release.

#### Upload evalml release to PyPI's test server
We need to upload an evalml package to test with the conda recipe
1. Make a new development release branch on evalml (in this example we'll be testing the 0.12.2 release)
    ```bash
    git checkout -b release_v0.12.2.dev
    ```
2. Update version number in `setup.py` and `evalml/__init__.py` to bump `__version__` to the new version. to v0.12.2.dev0 and push branch to repo.
3. Publish a new release of evalml on Github.
    1. Go to the [releases page](https://github.com/FeatureLabs/evalml/releases) on Github
    2. Click "Draft a new release"
    3. For the target, choose the new branch (v0.12.2.dev)
    4. For the tag, use the new version number (v0.12.2.dev0)
    5. For the release title, use the new version number (v0.12.2.dev0)
    6. For the release description, write "Development release for testing purposes"
    7. Check the "This is a pre-release" box
    8. Publish the release
4. The new release will be uploaded to TestPyPI automatically

#### Set up fork of our conda-forge repo
Branches on the conda-forge evalml repo are automatically built and the package uploaded to conda-forge, so to test a release without uploading to conda-forge we need to fork the repo and develop on the fork.
1. Fork conda-forge/evalml-core-feedstock: visit https://github.com/conda-forge/evalml-core-feedstock and click fork
2. Clone forked repo locally
3. Add conda-forge repo as the 'upstream' repository
    ```bash
    git remote add upstream https://github.com/conda-forge/evalml-core-feedstock.git 
    ```
4. If you made the fork previously and its master branch is missing commits, update it with any changes from upstream
    ```bash
    git fetch upstream
    git checkout master
    git merge upstream/master
    git push origin master
    ```
5. Make a branch with the version you want to release
    ```bash
    git checkout -b new-evalml-version
    ```

#### Update conda recipe to use TestPyPI release of evalml 
Fields to update in `recipe/meta.yaml` of feedstock repo:
* Always update:
    * Set the new release number (e.g. v0.12.2.dev0)
        ```
        {% set version = "0.12.2.dev0" %}
        ```
    * Source fields
        * url - visit https://test.pypi.org/project/evalml/, find correct release, go to download files page, and copy link location of the tar.gz file
        * sha256 - from the download files page, click the view hashes button for the tar.gz file and copy the sha256 digest
        ```
        source:
          url: <fill-this-in> 
          sha256: <fill-this-in> 
       ```
* Update if dependencies have changed.
    * Update the run requirements section for evalml-core if core dependencies have changed.
        ```
        requirements:
          host:
            - python >=3.6
            - pip
          run:
            - numpy >=1.16.4
            - pandas >=0.25.0
        ```
    * Update the run requirements section for evalml if non-core dependencies have changed.
    * Update the test requirements section for evalml-core and evalml if test dependencies have changed.
        ```
        test:
          imports:
            - evalml
          requires:
            - pytest >= 4.4.*
            - nbval ==0.9.3
            - python-graphviz >=0.8.4
            - category_encoders >=2.0.0
        ```

#### Test with conda-forge CI
1. Install conda
    1. If using pyenv, `pyenv install miniconda3-latest`
    2. Otherwise follow instructions in [conda docs](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Install conda-smithy (conda-forge tool to update boilerplate in repo)
    ```bash
    conda install -n root -c conda-forge conda-smithy
    ```
3. Run conda-smithy on feedstock
    ```bash
    cd /path/to/feedstock/repo
    conda-smithy rerender --commit auto
    ```
4. Push updated branch to the forked feedstock repo
3. Make a PR on conda-forge/evalml-core-feedstock from the forked repo and let CI tests run - add "[DO NOT MERGE]" to the PR name to indicate this is PR should not be merged in
4. After the tests pass, close the PR without merging

## 2. Create release PR to update version and release notes
Please use the following pattern for the release PR branch name: "release_vX.X.X". Doing so will bypass our release notes checkin test which requires all other PRs to add a release note entry.

Create a release PR with the following changes:
* Update `setup.py` and `evalml/__init__.py` to bump `__version__` to the new version.
* Move all entries in `docs/source/release_notes.rst` currently listed under `**Future Releases**` to be under a new heading with the version number and release date.
* Make sure `**Future Releases**` is empty except for the sub-headings, so its ready for new entries.
* Populate the release PR body with a copy of this release's release notes, reformatted to [GitHub markdown](https://guides.github.com/features/mastering-markdown/). You'll reuse this text in step 2. This is currently done by hand and can be done faster with some clever text editor features.
* Confirm that all release items are in the release notes under the correct header, and that no extra items are listed. You may have to do an "empty cache and hard reset" in your browser to see updates.

An example can be found here: https://github.com/FeatureLabs/evalml/pull/163

Checklist before merging:
* PR has been reviewed and approved.
* All tests are currently green on checkin and on main.
* The ReadtheDocs build for the release PR branch has passed, and the resulting docs contain the expected release notes.
* Confirm with the team that `main` will be frozen until step 3 (github release) is complete.

After merging, verify again that ReadtheDocs "latest" is correct.

## 3. Create GitHub Release
After the release pull request has been merged into the main branch, it is time to draft the GitHub release. [Here's GitHub's documentation](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository#creating-a-release) on how to do that. Include the following when creating the release:
* The target should be the main branch, which is the default value.
* The tag should be the version number with a "v" prefix (e.g. "vX.X.X").
* The release title is the same as the tag, "vX.X.X"
* The release description should be the full release notes updates for the release, reformatted as GitHub markdown (from the release PR body in step 1).

Note that by targeting `main`, there must be no new merges to `main` from the moment we merge the release PR to when we publish the new GitHub release. Otherwise, the release will point at the wrong commit on `main`!

Save the release as a draft and make sure it looks correct. You could start the draft while waiting for the release PR to be ready to merge.

When it's ready to go, hit "Publish release." This will create a "vX.X.X" tag for the release, which tells ReadtheDocs to build and update the "stable" version. This will also deploy the release [to pypi](https://pypi.org/project/evalml/), making it publicly accessible!

## 4. Make Documentation Public for Release Version
Creating the GitHub release should have updated the default `stable` docs branch to point at the new version. You'll now need to activate the new release version on ReadtheDocs so its publicly visible in the list of versions. This is important so users can view old documentation versions which match their installed version.

Please do the following:
* Log in to our ReadtheDocs account and go [here](https://readthedocs.com/projects/feature-labs-inc-evalml/versions/) to view the version list.
* Find "vX.X.X" in the version list, and click "Edit" on the right.
* Check the "Active" checkbox and set privacy level to "Public", then click "Save"
* Verify "vX.X.X" is now visible as a version on our ReadtheDocs page. You may have to do an "empty cache and hard reset" in your browser to see updates.
* Verify "stable" corresponds with the new version, which should've been done in step 2.

## 5. Verify the release package has been deployed
Now that the release has been made in the repo, to pypi and in our documentation, the final step is making sure the new release is publicly pip-installable via pypi.

In a fresh virtualenv, install evalml via pip and ensure it installs successfully:
```shell
# should come back empty
pip freeze | grep evalml

pip install evalml
python --version
# should now list the correct version
python -c "import evalml; print(evalml.__version__)"
pip freeze | grep evalml
```

Note: make sure when you do this that you're in a virtualenv, your current working directory isn't in the evalml repo, and that you haven't added your repo to the `PYTHONPATH`, because in both cases python could pick up the repo instead, even in a virtualenv.
