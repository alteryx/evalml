import argparse
import json
import re
import subprocess
import requirements
from pip._vendor import pkg_resources
from pkg_resources import packaging

def get_all_package_versions(package_name: str) -> list:
    # grabs all package versions using pip index
    args = ['pip', 'index', 'versions', package_name]
    version_substring = "Available versions: "

    subprocess.run(args, capture_output=True)
    try:
        sub_output = subprocess.check_output(args).decode('utf-8')
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    # this gets the string versions available from the `pip index` command
    string_versions = [output_split[len(version_substring):].split(", ") 
                        for output_split in sub_output.split("\n") if version_substring in output_split][0]
    # we filter out versions that might have letters in it (ie 22.2.post1, etc) to only compare versions that we can convert to float
    float_versions = list(set([packaging.version.parse(vers_string).base_version for vers_string in string_versions]))
    # sorting for ease later on
    return sorted(float_versions)

def get_version_logic(vers_equality: list) -> list:
    # handles grabbing all the logic for version control
    # expects list of tuples for [(equality, version)]
    version_logic = []
    for ve in vers_equality:
        value = ve[1]
        # replace the '~=' equality with '>=' for simplicity
        # since we are finding the minimum requirement that satisfies the equality
        # this logic should be about the same
        equality = ve[0].replace("~", ">")
        version_logic.append("{} packaging.version.parse('{}')".format(equality, value))
    return version_logic

def get_min_version_by_logic(available_versions: list, version_logic: list) -> list:
    # the list of available versions is sorted in descending order, which is how `pip index` returns
    if not len(version_logic):
        return available_versions[0]
    for index in range(len(available_versions)):
        # grab the 0 index to get the float value of the version
        # creates the expression that we evaluate to determine if the version satisfies the logic
        eval_string = "packaging.version.parse('{}')".format(available_versions[index]) + \
                      f" and packaging.version.parse('{available_versions[index]}')".join(version_logic)
        if eval(eval_string):
            return available_versions[index]

def add_versions_to_dict(package_to_version_dict: dict, package: str, version: tuple):
    # handles the logic for adding a version to the version dictionary we track
    if package in package_to_version_dict:
        if packaging.version.parse(version) > packaging.version.parse(package_to_version_dict[package]):
            # minimal dependency for this package is greater than another, so we take the greater
            package_to_version_dict.update({package: version})
    else:
        package_to_version_dict.update({package: version})

def get_min_version_string(package_to_version_dict: dict, delim: str, write: bool, output_name: str) -> str:
    # returns the min versions of all packages as a string delimited by the delim value.
    return_string = ""
    for package, version in package_to_version_dict.items():
        return_string += f"{package}=={version}"
        return_string += delim
    if (write):
        with open(output_name, "w+") as f:
            f.write(return_string)
        f.close()
    return return_string

def install_min_deps():
    # handles the install of all min dependencies
    print("Installing the minimum requirements generated")
    process = ['pip', 'install']
    process.extend(min_reqs.split(delim)[:-1])
    process = [x for x in process if ('pip==' not in x)]
    subprocess.run(process, capture_output=False)
    print("Done!")

def get_min_test_and_core_requirements(core_requirements: str, test_requirements: str) -> tuple:
    # input arguments should be the path to the core and test requirements.txt files
    min_reqs = ''
    min_core_reqs, min_test_reqs = [], []
    with open(test_requirements, "r") as f:
        min_reqs = f.read()
        min_test_reqs = min_reqs.split("\n")[:-1]

    with open(core_requirements, "r") as f:
        s = f.read()
        min_reqs += s
        min_core_reqs = s.split("\n")[:-1]

    return (min_reqs, min_core_reqs, min_test_reqs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies of min dependencies")
    parser.add_argument('--json-filename', default='mindep.json', required=False)
    parser.add_argument('--write-txt', default="False", required=False)
    parser.add_argument('--output-name', default='min_min_dep.txt', required=False)
    parser.add_argument('--delimiter', default='\n', required=False)
    parser.add_argument('--install', default="True", required=False)
    parser.add_argument('--req-file-path', default='evalml/tests/dependency_update_check/', required=False)
    args = parser.parse_args()

    # get the arguments from the parser
    json_name = args.json_filename
    delim = args.delimiter
    write = bool(args.write_txt=='True')
    output_name = args.output_name
    install = bool(args.install=='True')
    path = args.req_file_path
    
    # get the minimum requirements and install the min core requirements
    min_reqs, min_core_reqs, min_test_reqs = get_min_test_and_core_requirements(path + "minimum_core_requirements.txt", path + "minimum_test_requirements.txt")
    install_min_deps()

    package_to_version_dict = {}
    all_requirements = []

    # find all packages that the core requirements rely on
    for package in min_core_reqs:
        pack = tuple(requirements.parse(package))[0]
        _package_name = pack.name
        _package = pkg_resources.working_set.by_key[_package_name]
        all_requirements.append(package)
        # reliance will consist of requirements like ["scipy>=0.17.0", "pandas>=X.x", "moto"]
        reliance = [str(r) for r in _package.requires()]
        all_requirements.extend(reliance)  # retrieve deps from setup.py

    # for any requirements in our test-requirements that apply to the core requirement packages, add that in as well
    for package in min_test_reqs:
        pack = tuple(requirements.parse(package))[0]
        _package_name = pack.name
        if any([_package_name in s for s in all_requirements]):
            all_requirements.append(package)
    all_requirements = list(set(all_requirements))

    # iterate through each dependency to determine the minimum version allowed
    for package_value in all_requirements:
        req = tuple(requirements.parse(package_value))
        try:
            package_name = req[0].name
            version_logic = get_version_logic(req[0].specs)
        except IndexError:
            continue
        all_versions = get_all_package_versions(package_name)
        min_version = get_min_version_by_logic(all_versions, version_logic)
        add_versions_to_dict(package_to_version_dict, package_name, min_version)

    # min_reqs will represent the string version of all min requirements for the package
    # this does not include testing requirements, which we will need to install prior
    min_reqs = get_min_version_string(package_to_version_dict, delim, write, output_name)
    if install:
        install_min_deps()
