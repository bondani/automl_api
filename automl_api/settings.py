import pathlib
import yaml

def load_config(path):
    with open(path) as f:
        try:
            conf = yaml.safe_load(f)
        except yaml.YAMLError:
            print('Config Error')
            conf = {}

    return conf