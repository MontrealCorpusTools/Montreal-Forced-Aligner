import yaml
from ..exceptions import ConfigError


class BaseConfig(object):
    def update(self, data):
        for k, v in data.items():
            if not hasattr(self, k):
                raise ConfigError('No field found for key {}'.format(k))
            setattr(self, k, v)

def save_config(config, path):
    with open(path, 'w', encoding='utf8') as f:
        yaml.dump(config.params(), f)
