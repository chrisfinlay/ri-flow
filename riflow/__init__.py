import yaml
import re
import os
from importlib.resources import files


loader = yaml.SafeLoader
loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def yaml_load(path):
    config = yaml.load(open(path), Loader=loader)
    return config


class Tee(object):
    """https://stackoverflow.com/questions/17866724/python-logging-print-statements-while-having-them-print-to-stdout"""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        pass


def deep_update(d: dict, u: dict) -> dict:
    """Recursively update a dictionary which includes subdictionaries.

    Parameters
    ----------
    d : dict
        Base dictionary to update.
    u : dict
        Update dictionary.

    Returns
    -------
    dict
        Updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_config(path: str, config_type: str = "extract") -> dict:
    """Load a configuration file and populate default parameters where needed.

    Parameters
    ----------
    path : str
        Path to the yaml config file.
    config_type : str, optional
        Type of configuration file, by default "extract". Options are {"extract"}.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    config_dir = files("riflow").joinpath("data").__str__()

    extract_base_config_path = os.path.join(config_dir, "extract_config_base.yaml")

    config = yaml_load(path)
    if config_type == "extract":
        base_config = yaml_load(extract_base_config_path)
    # elif config_type == "pow_spec":
    #     base_config = yaml_load(pow_spec_base_config_path)
    else:
        raise ValueError("A config type must be specified. Options are {extract}.")

    return deep_update(base_config, config)
