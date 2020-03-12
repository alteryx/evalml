from setuptools import find_packages, setup

setup(
    name='evalml',
    version='0.7.0',
    author='Feature Labs, Inc.',
    author_email='support@featurelabs.com',
    url='http://www.featurelabs.com/',
    install_requires=open('requirements.txt').readlines(),
    tests_require=open('test-requirements.txt').readlines(),
    packages=find_packages(),
    include_package_data=True,
    data_files=[('evalml/demos/data', ['evalml/demos/data/fraud_transactions.csv.tar.gz'])]
)
