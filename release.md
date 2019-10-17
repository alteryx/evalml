# Release Process

## Run Integration Tests
Before creating a release EvalML integration tests should be run. Instructions can be found here:
https://github.com/FeatureLabs/evalml-integration-tests

## Create Release PR
A release PR should have the version number as the title and the changelog updates as the PR body text. 
The PR should include the following:
    * Changes to `setup.py` and `evalml/__init__.py` to bump `__version__` to the correct version
    * Move change log items from `Future Releases` into the correct version number
    * Confirm that all release items are in the changelog

## Create GitHub Release
After the release pull request has been merged into the master branch, it is time draft the github release.
    * The target should be the master branch
    * The tag should be the version number with a v prefix (e.g. v0.1.2)
    * Release title is the same as the tag
    * Release description should be the full changelog updates for the release.

## Release using Release-tools
Run the following assuming you have release-tools installed:
    * flrelease build-package
    * flrelease upload-package <--url or --license> <install.featurelabs.com or licenses/user.json>

More details can be found here: https://github.com/FeatureLabs/release-tools