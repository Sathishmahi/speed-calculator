# import yaml
# import os
# from const import CONFIG_YAML_PATH

# def read_yaml(yaml_file_path:str =  CONFIG_YAML_PATH)->dict[str:dict]:

#     if not os.path.exists(yaml_file_path):
#         raise FileNotFoundError(f"{yaml_file_path} not found")

#     with open(yaml_file_path) as f:
#         return yaml.safe_load(f)

import yaml
import os
from const import CONFIG_YAML_PATH

def read_yaml(yaml_file_path: str = CONFIG_YAML_PATH) -> dict:
    """
    Read a YAML file and parse its content into a dictionary.

    Parameters:
    - yaml_file_path (str): Path to the YAML file to be read.

    Returns:
    - config_dict (dict): Dictionary containing the parsed YAML content.

    Raises:
    - FileNotFoundError: If the specified YAML file does not exist.
    
    Example usage:
    >>> config = read_yaml("config.yaml")
    >>> print(config)
    {'key1': 'value1', 'key2': 'value2', ...}
    """
    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(f"{yaml_file_path} not found")

    with open(yaml_file_path) as f:
        config_dict = yaml.safe_load(f)  # Parse the YAML content into a dictionary

    return config_dict




