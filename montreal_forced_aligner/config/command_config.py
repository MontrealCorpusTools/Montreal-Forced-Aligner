import yaml
import os


class CommandConfig(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        if item not in self.data:
            return None
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def update(self, new_data):
        self.data.update(new_data)

    def save(self, conf_path):
            with open(conf_path, 'w') as f:
                yaml.dump(self.data, f)


def load_command_configuration(conf_path, default):
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
        config = CommandConfig(conf)
    else:
        config = CommandConfig(default)
    return config