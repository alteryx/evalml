# Release Process

## 0. Pre-release Checklist
Before starting the release process, verify the following:
* All work required for this release has been completed and the team is ready to release.
* [All CircleCI tests are green on master](https://app.circleci.com/pipelines/github/FeatureLabs/evalml?branch=master).
* The [ReadtheDocs build](https://readthedocs.com/projects/feature-labs-inc-evalml/builds/) for "latest" is marked as passed. To avoid mysterious errors, best practice is to empty your browser cache when reading new versions of the docs!
* The [public documentation for the "latest" branch](https://evalml.featurelabs.com/en/latest/) looks correct, and the [changelog](https://evalml.featurelabs.com/en/latest/changelog.html) includes the last change which was made on master.
* The [performance tests](https://github.com/FeatureLabs/evalml-performance-tests) have passed on latest master, and the team has reviewed the results.

## 1. Create release PR to update version and changelog
Please use the following pattern for the release PR branch name: "release_vX.X.X". Doing so will bypass our changelog checkin test which requires all other PRs to add a changelog entry.

Create a release PR with the following changes:
* Update `setup.py` and `evalml/__init__.py` to bump `__version__` to the new version.
* Move all entries in `docs/source/changelog.rst` currently listed under `**Future Releases**` to be under a new heading with the version number and release date.
* Make sure `**Future Releases**` is empty except for the sub-headings, so its ready for new entries.
* Populate the release PR body with a copy of this release's changelog, reformatted to [GitHub markdown](https://guides.github.com/features/mastering-markdown/). You'll reuse this text in step 2. This is currently done by hand and can be done faster with some clever text editor features.
* Confirm that all release items are in the changelog under the correct header, and that no extra items are listed. You may have to do an "empty cache and hard reset" in your browser to see updates.

An example can be found here: https://github.com/FeatureLabs/evalml/pull/163

Checklist before merging:
* PR has been reviewed and approved.
* All tests are currently green on checkin and on master.
* The ReadtheDocs build for the release PR branch has passed, and the resulting docs contain the expected changelog.
* Confirm with the team that `master` will be frozen until step 3 (github release) is complete.

After merging, verify again that ReadtheDocs "latest" is correct.

## 2. Create GitHub Release
After the release pull request has been merged into the master branch, it is time to draft the GitHub release. [Here's GitHub's documentation](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository#creating-a-release) on how to do that. Include the following when creating the release:
* The target should be the master branch, which is the default value.
* The tag should be the version number with a "v" prefix (e.g. "vX.X.X").
* The release title is the same as the tag, "vX.X.X"
* The release description should be the full changelog updates for the release, reformatted as GitHub markdown (from the release PR body in step 1).

Note that by targeting `master`, there must be no new merges to `master` from the moment we merge the release PR to when we publish the new GitHub release. Otherwise, the release will point at the wrong commit on `master`!

Save the release as a draft and make sure it looks correct. You could start the draft while waiting for the release PR to be ready to merge.

When it's ready to go, hit "Publish release." This will create a "vX.X.X" tag for the release, which tells ReadtheDocs to build and update the "stable" version.

## 3. Update Public Documentation
Creating the GitHub release should have updated the default `stable` docs branch to point at the new version. You'll now need to activate the new release version on ReadtheDocs so its publicly visible in the list of versions.

Please do the following:
* Log in to our ReadtheDocs account and go [here](https://readthedocs.com/projects/feature-labs-inc-evalml/versions/) to view the version list.
* Find "vX.X.X" in the version list, and click "Edit" on the right.
* Check the "Active" checkbox and set privacy level to "Public", then click "Save"
* Verify "vX.X.X" is now visible as a version on our ReadtheDocs page. You may have to do an "empty cache and hard reset" in your browser to see updates.
* Verify "stable" corresponds with the new version, which should've been done in step 2.

## 4. Release using Release-tools
Now that the release has been made in the repo and in our documentation, the final step is deploying the code to make it pip-installable.

First, make sure you have [release-tools](https://github.com/FeatureLabs/release-tools) installed.

Open a terminal and navigate to the top directory in the evalml repo:
```shell
cd {your_workspace}/evalml
```

Next, you'll need to configure a licenses file to include any emails you'd like to make the new release available to.

Create a folder called `licenses`, and create a file called `admin.json` file in that folder which contains the following:
```json
{"email": "admin@featurelabs.com"}
```

Run the following command to build a release tarball and make it pip-installable to the specified licenses:
```shell
flrelease upload-package --url install.featurelabs.com --license licenses/admin.json
```

This will print out the license hash associated with the email in `licenses/admin.json`.

The final step is to verify that the release was successful.
Log into the S3 console and navigate to [the install.featurelabs.com bucket](https://s3.console.aws.amazon.com/s3/buckets/install.featurelabs.com/?region=us-east-1). Find and open the folder corresponding to the license hash from the `flrelease` command. The `evalml` folder should contain a release tarball with the appropriate version, i.e. `evalml-X.X.X.tar.gz`, and an `index.html` package index.

Open the `index.html` file and verify it lists the new version.

In a fresh virtualenv, install evalml via pip and ensure it installs successfully:
```shell
pip install evalml --index-url https://install.featurelabs.com/<key>
python --version
python -c "import evalml; print(evalml.__version__)"
```
