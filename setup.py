from setuptools import find_packages, setup

setup(
    name='evalml',
    version='0.2.1',
    author='Feature Labs, Inc.',
    author_email='support@featurelabs.com',
    url='http://www.featurelabs.com/',
    install_requires=open('requirements.txt').readlines(),
    tests_require=open('test-requirements.txt').readlines(),
    packages=find_packages(),
    include_package_data=True,
)
