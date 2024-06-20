import os
from pathlib import Path

import yaml


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_config_file_path():
    current_path = Path(__file__)
    return str(current_path.parents[2] / 'conf.d' / 'config.yaml')


def get_config():
    config_path = get_config_file_path()
    return load_config(config_path)


def get_path():
    return Path(__file__).parents[1]


def get_api_key():
    config = get_config()
    os.environ['OPENAI_API_KEY'] = config['openai']['key']


def load_prompt(name):
    path = get_path()
    prompt_path = path / 'eval' / 'prompts' / name
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt
