from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

extras_require = {
    'update_checker': ['alteryx-open-src-update-checker >= 2.0.0'],
    'prophet': ['cmdstan-builder == 0.0.4']
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(
    name='evalml',
    version='0.34.1rc1',
    author='Alteryx, Inc.',
    author_email='support@featurelabs.com',
    description='EvalML is an AutoML library that builds, optimizes, and evaluates machine learning pipelines using domain-specific objective functions.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/alteryx/evalml/',
    install_requires=open('core-requirements.txt').readlines() + open('requirements.txt').readlines()[1:],
    extras_require=extras_require,
    tests_require=open('test-requirements.txt').readlines(),
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
          'evalml = evalml.__main__:cli'
        ]
    },
    data_files=[('evalml/tests/data', ['evalml/tests/data/tips.csv',
                                       'evalml/tests/data/titanic.csv',
                                       'evalml/tests/data/churn.csv',
                                       'evalml/tests/data/fraud_transactions.csv.gz']),
                ('evalml/demos/data', ['evalml/demos/data/daily-min-temperatures.csv'])],
)
