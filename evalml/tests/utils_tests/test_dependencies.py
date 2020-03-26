import pathlib
from importlib import import_module

import requirements


def test_has_minimal_deps(has_minimal_dependencies):
    reqs_path = pathlib.Path(__file__).absolute().parents[3].joinpath('requirements.txt')
    lines = open(reqs_path, 'r').readlines()
    lines = [line for line in lines if '-r ' not in line]
    reqs = requirements.parse(''.join(lines))
    extra_deps = [req.name for req in reqs]
    extra_deps += ['plotly.graph_objects']
    for module in extra_deps:
        try:
            import_module(module)
            # an extra dep was imported. if the tests were configured with --has-minimal-deps, that's an error.
            assert not has_minimal_dependencies, ("The test environment includes extra dependency '{}', " +
                                                  "but tests were configured with " +
                                                  "'--has-minimal-dependencies'. Please either uninstall " +
                                                  "all extra dependencies as listed in requirements.txt, " +
                                                  "or rerun the tests without " +
                                                  "'--has-minimal-dependencies'.").format(module)
        except ImportError:
            # an extra dep failed to import. if the tests were configured with --has-minimal-deps, that's
            # expected. otherwise, it's an error.
            assert has_minimal_dependencies, ("The test environment is missing expected extra dependency '{}'. " +
                                              "Please either install all requirements in requirements.txt, " +
                                              "or rerun the tests with " +
                                              "'--has-minimal-dependencies'.").format(module)
