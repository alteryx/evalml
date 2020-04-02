from setuptools import find_packages, setup

setup(
    name='evalml',
    version='0.8.0',
    author='Feature Labs, Inc.',
    author_email='support@featurelabs.com',
    url='http://www.featurelabs.com/',
    install_requires=open('core-requirements.txt').readlines() + open('requirements.txt').readlines()[1:],
    tests_require=open('test-requirements.txt').readlines(),
    packages=find_packages(),
    include_package_data=True,
    data_files=[('evalml/demos/data', ['evalml/demos/data/fraud_transactions.csv.tar.gz'])]
)
