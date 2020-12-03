# Release Process

## 0. Pre-Release Checklist
Before starting the release process, verify the following:
* All work required for this release has been completed and the team is ready to release.
* [All CircleCI tests are green on main](https://app.circleci.com/pipelines/github/alteryx/evalml?branch=main).
* The [ReadtheDocs build](https://readthedocs.com/projects/feature-labs-inc-evalml/builds/) for "latest" is marked as passed. To avoid mysterious errors, best practice is to empty your browser cache when reading new versions of the docs!
* The [public documentation for the "latest" branch](https://evalml.featurelabs.com/en/latest/) looks correct, and the [release notes](https://evalml.featurelabs.com/en/latest/release_notes.html) includes the last change which was made on main.
* The [performance tests](https://github.com/FeatureLabs/evalml_looking_glass) have passed on latest main, and the team has reviewed the results.
* Get agreement on the version number to use for the release.

#### Version Numbering

EvalML uses [semantic versioning](https://semver.org/). Every release has a major, minor and patch version number, and are displayed like so: `<majorVersion>.<minorVersion>.<patchVersion>`.

If you'd like to create a development release, which won't be deployed to pypi and conda and marked as a generally-available production release, please add a "dev" prefix to the patch version, i.e. `X.X.devX`. Note this claims the patch number--if the previous release was `0.12.0`, a subsequent dev release would be `0.12.dev1`, and the following release would be `0.12.2`, *not* `0.12.1`. Development releases deploy to [test.pypi.org](https://test.pypi.org/project/evalml/) instead of to [pypi.org](https://pypi.org/project/evalml).

## 1. Create release PR to update version and release notes
Please use the following pattern for the release PR branch name: "release_vX.X.X". Doing so will bypass our release notes checkin test which requires all other PRs to add a release note entry.

Create a release PR with the following changes:
* Update `setup.py` and `evalml/__init__.py` to bump `__version__` to the new version.
* Move all entries in `docs/source/release_notes.rst` currently listed under `**Future Releases**` to be under a new heading with the version number and release date.
* Make sure `**Future Releases**` is empty except for the sub-headings, so its ready for new entries.
* Populate the release PR body with a copy of this release's release notes, reformatted to [GitHub markdown](https://guides.github.com/features/mastering-markdown/). You'll reuse this text in step 2. You can generate the markdown by running `tools/format_release_notes.sh` locally.
* Confirm that all release items are in the release notes under the correct header, and that no extra items are listed. You may have to do an "empty cache and hard reset" in your browser to see updates.

An example can be found here: https://github.com/alteryx/evalml/pull/163

Checklist before merging:
* PR has been reviewed and approved.
* All tests are currently green on checkin and on main.
* The ReadtheDocs build for the release PR branch has passed, and the resulting docs contain the expected release notes.
* Confirm with the team that `main` will be frozen until step 3 (github release) is complete.

After merging, verify again that ReadtheDocs "latest" is correct.

## 2. Create GitHub Release
After the release pull request has been merged into the main branch, it is time to draft the GitHub release. [Here's GitHub's documentation](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository#creating-a-release) on how to do that. Include the following when creating the release:
* The target should be the main branch, which is the default value.
* The tag should be the version number with a "v" prefix (e.g. "vX.X.X").
* The release title is the same as the tag, "vX.X.X"
* The release description should be the full release notes updates for the release, reformatted as GitHub markdown (from the release PR body in step 1).

Note that by targeting `main`, there must be no new merges to `main` from the moment we merge the release PR to when we publish the new GitHub release. Otherwise, the release will point at the wrong commit on `main`!

Save the release as a draft and make sure it looks correct. You could start the draft while waiting for the release PR to be ready to merge.

When it's ready to go, hit "Publish release." This will create a "vX.X.X" tag for the release, which tells ReadtheDocs to build and update the "stable" version. This will also deploy the release [to pypi](https://pypi.org/project/evalml/), making it publicly accessible!

## 3. Make Documentation Public for Release Version
Creating the GitHub release should have updated the default `stable` docs branch to point at the new version. You'll now need to activate the new release version on ReadtheDocs so its publicly visible in the list of versions. This is important so users can view old documentation versions which match their installed version.

Please do the following:
* Log in to our ReadtheDocs account and go [here](https://readthedocs.com/projects/feature-labs-inc-evalml/versions/) to view the version list.
* Find "vX.X.X" in the version list, and click "Edit" on the right.
* Check the "Active" checkbox and set privacy level to "Public", then click "Save"
* Verify "vX.X.X" is now visible as a version on our ReadtheDocs page. You may have to do an "empty cache and hard reset" in your browser to see updates.
* Verify "stable" corresponds with the new version, which should've been done in step 2.

## 4. Verify the release package has been deployed
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

## 5. Publish Our New Conda Package

A couple of hours after you publish the GitHub release, a bot will open a PR to our [feedstock](https://github.com/conda-forge/evalml-core-feedstock) that automatically
bumps the recipe to use the latest version of the package.
In order to publish our latest conda package, we need to make some changes to the bot's PR and merge it.

The bot's PR will will remove the quotes around the version tag in the recipe.
Removing these quotes will break our `build_conda_pkg` CI job so add them back in and push your changes to the bot's PR. 
For example, lines 3-5 of the [recipe](https://github.com/conda-forge/evalml-core-feedstock/blob/master/recipe/meta.yaml) should look like the following:
```yaml
package:
  name: evalml-core
  version: '{{ version }}'
```
For help on how to push changes to the bot's PR please read this [document.](https://conda-forge.org/docs/maintainer/updating_pkgs.html#pushing-to-regro-cf-autotick-bot-branch)
If the `build_conda_pkg` job passed on the `main` branch before the release, then no further changes are needed and you can merge the bot's PR once the tests pass.

After you merge the PR, our latest package will be deployed to conda-forge! To verify, run this in a fresh conda environment:

```shell
conda install -c conda-forge evalml
```

Verify the latest version of `evalml` got installed by running 

```shell
python -c "import evalml; print(evalml.__version__)"
``` 