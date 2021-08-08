import yaml
from ..exceptions import ConfigError

DEFAULT_PUNCTUATION = r'、。।，@<>"(),.:;¿?¡!\\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=‘'

DEFAULT_CLITIC_MARKERS = "'’"
DEFAULT_COMPOUND_MARKERS = "-/"
DEFAULT_STRIP_DIACRITICS = ['ː', 'ˑ', '̩', '̆', '̑', '̯', '͡', '‿', '͜']
DEFAULT_DIGRAPHS = ['[dt][szʒʃʐʑʂɕç]', '[aoɔe][ʊɪ]']


PARSING_KEYS = ['punctuation', 'clitic_markers', 'compound_markers', 'multilingual_ipa', 'strip_diacritics', 'digraphs']

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
                if isinstance(original_value, bool):
                    if args[i+1].lower() == 'true':
                        val = True
                    elif args[i+1].lower() == 'false':
                        val = False
                    elif not original_value:
                        val = True
                    else:
                        continue
                else:
                    val = type(original_value)(args[i+1])
            except (ValueError):
                continue
            except (IndexError):
                if isinstance(original_value, bool):
                    if not original_value:
                        val = True
                    else:
                        continue
                else:
                    continue
            setattr(self, name, val)


def save_config(config, path):
    with open(path, 'w', encoding='utf8') as f:
        yaml.dump(config.params(), f)
