# import glob
import os
# import re
# import shutil
# import dill
# import pickle5
# from shutil import copyfile


def save_yaml(data, filepath):
    import yaml
    with open(filepath, 'w') as f:
        yaml.dump(data, f)


def load_yaml(filepath, filename=None):
    import yaml

    if filename is not None:
        full_path = os.path.join(filepath, filename)
    else:
        full_path = filepath

    with open(full_path) as f:
        data = yaml.safe_load(f)

    return data
