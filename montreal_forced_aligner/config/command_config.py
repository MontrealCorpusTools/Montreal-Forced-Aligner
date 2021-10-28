from __future__ import annotations
from typing import Any
import yaml
import os


class CommandConfig(object):
    def __init__(self, data: dict):
        self.data = data

    def __getitem__(self, item: str) -> Any:
        if item not in self.data:
            return None
        return self.data[item]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def update(self, new_data: dict) -> None:
        self.data.update(new_data)

    def save(self, conf_path: str) -> None:
            with open(conf_path, 'w') as f:
                yaml.dump(self.data, f)


def load_command_configuration(conf_path: str, default: dict) -> CommandConfig:
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
        config = CommandConfig(conf)
    else:
        config = CommandConfig(default)
    return config