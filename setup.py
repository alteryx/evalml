from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='evalml',
    version='0.24.1',
    author='Alteryx, Inc.',
    author_email='support@featurelabs.com',
    description='EvalML is an AutoML library that builds, optimizes, and evaluates machine learning pipelines using domain-specific objective functions.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/alteryx/evalml/',
    install_requires=open('core-requirements.txt').readlines() + open('requirements.txt').readlines()[1:],
    tests_require=open('test-requirements.txt').readlines(),
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
          'evalml = evalml.__main__:cli'
        ]
    },
    data_files=[('evalml/demos/data', ['evalml/demos/data/fraud_transactions.csv.gz', 'evalml/demos/data/churn.csv']),
                ('evalml/tests/data', ['evalml/tests/data/tips.csv', 'evalml/tests/data/titanic.csv'])],
)
