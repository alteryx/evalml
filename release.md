# Release Process

## Run Integration Tests
Before creating a release EvalML integration tests should be run. Instructions can be found here:
https://github.com/FeatureLabs/evalml-integration-tests

## Create Release PR
A release PR should have the version number as the title and the changelog updates as the PR body text.
The PR should *merge off master* and include the following:
* Changes to `setup.py` and `evalml/__init__.py` to bump `__version__` to the correct version
* Move changelog items from `Future Releases` into the correct version number
* Confirm that all release items are in the changelog

An example can be found here: https://github.com/FeatureLabs/evalml/pull/163

## Create GitHub Release
After the release pull request has been merged into the master branch, it is time to draft the github release.
* The target should be the master branch
* The tag should be the version number with a `v` prefix (e.g. v0.1.2)
* Release title is the same as the tag
* Release description should be the full changelog updates for the release.

## Documentation
After creating the GitHub release, use the GitHub tag and activate the current version on ReadTheDocs [here](https://readthedocs.com/projects/feature-labs-inc-evalml/versions/).

Please do the following:
* Activate `vX.X.X`
* Check the `Active` checkbox
* Set privacy level to `Public`

Documentation automatically updates using readthedocs so no need to manually compile.

## Release using Release-tools

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
