# Release Process

## 0. Pre-release Checklist
Before starting the release process, verify the following:
* All work required for this release has been completed and the team is ready to release.
* [All CircleCI tests are green on master](https://app.circleci.com/pipelines/github/FeatureLabs/evalml?branch=master).
* The [Read the Docs build](https://readthedocs.com/projects/feature-labs-inc-evalml/builds/) for "latest" is marked as passed.
* The [public documentation for the "latest" branch](https://evalml.featurelabs.com/en/latest/) looks correct, and the [changelog](https://evalml.featurelabs.com/en/latest/changelog.html) includes the last change which was made on master.
* The [performance tests](https://github.com/FeatureLabs/evalml-performance-tests) have passed on latest master, and the team has reviewed the results.

## 1. Create release PR to update version and changelog
A release PR should have the version number as the title (i.e. "vX.X.X") and the changelog updates as the PR body text. The PR should be based off master.

Please use the following pattern for the branch name: "release_vX.X.X". Doing so will bypass our changelog checkin test which requires all other PRs to add a changelog entry.

Make the following changes in the release PR:
* Update `setup.py` and `evalml/__init__.py` to bump `__version__` to the new version.
* Reformat the changelog to [GitHub markdown](https://guides.github.com/features/mastering-markdown/). This is currently done by hand and can be done faster with some clever text editor features.
* Move changelog items from `Future Releases` into the correct version number.
* Confirm that all release items are in the changelog under the correct header, and that no extra items are listed.

An example can be found here: https://github.com/FeatureLabs/evalml/pull/163

**Note**: get the PR reviewed and approved before merging. Also, verify again that all tests are currently green on master, that all checkin tests are passing, that the Read the Docs build for the release PR branch has passed and that the resulting docs contain the expected changelog. After merging, verify Read the Docs "latest" is correct.

## 2. Create GitHub Release
After the release pull request has been merged into the master branch, it is time to draft the GitHub release. [Here's GitHub's documentation](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository#creating-a-release) on how to do that.
* The target should be the master branch.
* The tag should be the version number with a "v" prefix (e.g. "vX.X.X").
* Release title is the same as the tag.
* Release description should be the full changelog updates for the release, reformatted as GitHub markdown.

Save the draft and review it. You could start the draft while waiting for the release PR to be ready to merge. When it's ready to go, hit "Publish release."

## 3. Update Public Documentation
After creating the GitHub release, we need to activate the release version on Read the Docs [here](https://readthedocs.com/projects/feature-labs-inc-evalml/versions/).

Please do the following:
* Find "vX.X.X" in the version list, and click "Edit" on the right.
* Check the "Active" checkbox and set privacy level to "Public", then click "Save"
* Verify "vX.X.X" is now visible as a branch on our Read the Docs page.
* Verify "stable" has been updated to correspond with the new version.

## 4. Release using Release-tools
Now that the release has been made in the repo and in our documentation, the final step is deploying the code to make it pip-installable.

First, make sure you have [release-tools](https://github.com/FeatureLabs/release-tools) installed.

Make sure you're in the top directory in the evalml repo:
```shell
cd {your_workspace}/evalml
```

If necessary, add a folder called "licenses" and create an `admin.json` file in that folder:
```json
{
    "email": "admin@featurelabs.com"
}
```

Run
```shell
flrelease upload-package --url install.featurelabs.com --license licenses/admin.json
```

You can also run the following to release to specific users.
* `flrelease build-package`
* `flrelease upload-package <--url or --license> <install.featurelabs.com or licenses/user.json>`

More details can be found in the [release-tools repo](https://github.com/FeatureLabs/release-tools).
