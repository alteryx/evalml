import yaml, argparse, os, pathlib


class CondaDumper(yaml.Dumper):

    def increase_indent(self, flow=False, indentless=False):
        return super(CondaDumper, self).increase_indent(flow, False)


def write_conda_recipe(version):
    """
    Writes the EvalML recipe to build a conda package based on the version
    Args:
        version: The version of EvalML we are building with this feedstock

    Returns:
        None: Side effect of overwriting the existing meta.yaml in the feedstock
    """

    recipe_file_path = os.path.join(os.path.join(os.path.join(os.getcwd(), 'evalml-core-feedstock'), 'recipe'), 'meta'
                                                                                                                '.yaml')
    with open(recipe_file_path, 'rb') as config_file:
        # Toss out the first line that declares the version since its not supported YAML syntax
        next(config_file)
        config = yaml.safe_load(config_file)
        # Path to the evalml repository on the docker container.  Since we are doing a local build this is
        # the target we are copying this directory to.
        recipe_path = str(pathlib.Path('..', 'feedstock_root', 'evalml'))
        config['source'] = {'path': recipe_path}
        config['package']['version'] = version

    with open(recipe_file_path, 'w') as recipe:
        yaml.dump(config, recipe, default_flow_style=False, sort_keys=False, Dumper=CondaDumper)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure conda for local build. Run from the feedstock root")
    parser.add_argument('version', help='The version of EvalML being built')
    args = parser.parse_args()
    write_conda_recipe(args.version)
