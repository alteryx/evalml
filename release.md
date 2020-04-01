# Release Process

## 0. Pre-release Checklist
Before starting the release process, verify the following:
* All work required for this release has been completed and the team is ready to release.
* [All circleci tests are green on master](https://app.circleci.com/pipelines/github/FeatureLabs/evalml?branch=master).
* The [readthedocs build](https://readthedocs.com/projects/feature-labs-inc-evalml/builds/) for "latest" is marked as passed.
* The [public documentation for the "latest" branch](https://evalml.featurelabs.com/en/latest/) looks correct, and the [changelog](https://evalml.featurelabs.com/en/latest/changelog.html) includes the last change which was made on master.
* The [performance tests](https://evalml.featurelabs.com/en/latest/changelog.html) have passed on latest master, and the team has reviewed the results.

## 1. Create release PR to update version and changelog
A release PR should have the version number as the title (i.e. "vX.X.X") and the changelog updates as the PR body text. The PR should be based off master. Please use a branch name which is different than "vX.X.X" because we create tags with that format in step 2.

Make the following changes in the release PR:
* Update `setup.py` and `evalml/__init__.py` to bump `__version__` to the new version.
* Reformat the changelog to github markdown. This is currently done by hand and can be done faster with some clever text editor features.
* Move changelog items from `Future Releases` into the correct version number.
* Confirm that all release items are in the changelog under the correct header, and that no extra items are listed.

An example can be found here: https://github.com/FeatureLabs/evalml/pull/163

**Note**: get the PR reviewed and approved before merging. Also, verify again that all tests are currently green on master, that all checkin tests are passing, that the readthedocs build for the release PR branch has passed and that the resulting docs contain the expected changelog. And after merging, verify readthedocs "latest" is correct.

## 2. Create GitHub Release
After the release pull request has been merged into the master branch, it is time to draft the github release. [Here's github's documentation](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository#creating-a-release) on how to do that.
* The target should be the master branch.
* The tag should be the version number with a "v" prefix (e.g. "vX.X.X").
* Release title is the same as the tag.
* Release description should be the full changelog updates for the release, reformatted as github markdown.

Save the draft and review it. When it's ready to go, hit "Publish release."

## 3. Update Public Documentation
After creating the GitHub release, activate the release version on ReadTheDocs [here](https://readthedocs.com/projects/feature-labs-inc-evalml/versions/).

Please do the following:
* Find `vX.X.X` in the version list, and click "Edit" on the right.
* Check the `Active` checkbox and set privacy level to `Public`, then click "Save"

Readthedocs will kick off fresh builds of the new version `vX.X.X` and will use that for the default `stable` docs branch.

## 4. Release using Release-tools

Do the following assuming you have [release-tools](https://github.com/FeatureLabs/release-tools) installed to upload to our internal license key (connected to admin@featurelabs.com):

Navigate to the root directory
```shell
cd evalml
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
