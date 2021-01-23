import yaml
from ..exceptions import ConfigError


class BaseConfig(object):
    def update(self, data):
        for k, v in data.items():
            if not hasattr(self, k):
                raise ConfigError('No field found for key {}'.format(k))
            setattr(self, k, v)

    def update_from_args(self, args):
        for i, a in enumerate(args):
            if not a.startswith('--'):
                continue
            name = a.replace('--', '')
            try:
                original_value = getattr(self, name)
            except AttributeError:
                continue
            if not isinstance(original_value, (bool, int, float, str)):
                continue
            try:
                val = type(original_value)(args[i+1])
            except (IndexError, ValueError):
                continue
            setattr(self, name, val)

def save_config(config, path):
    with open(path, 'w', encoding='utf8') as f:
        yaml.dump(config.params(), f)
