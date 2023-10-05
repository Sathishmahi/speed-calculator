import yaml
import os
from const import CONFIG_YAML_PATH

def read_yaml(yaml_file_path:str =  CONFIG_YAML_PATH)->dict[str:dict]:

    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(f"{yaml_file_path} not found")

    with open(yaml_file_path) as f:
        return yaml.safe_load(f)