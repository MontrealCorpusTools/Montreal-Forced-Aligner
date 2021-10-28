from __future__ import annotations
import os
import shutil
import sys
import logging
import yaml
import textwrap
from colorama import Fore, Style
from typing import List
from .models import MODEL_TYPES
from .exceptions import ThirdpartyError, KaldiProcessingError



def thirdparty_binary(binary_name: str) -> str:
    bin_path = shutil.which(binary_name)
    if bin_path is None:
        if binary_name in ['fstcompile', 'fstarcsort', 'fstconvert'] and sys.platform != 'win32':
            raise ThirdpartyError(binary_name, open_fst=True)
        else:
            raise ThirdpartyError(binary_name)
    return bin_path


def parse_logs(log_directory: str) -> None:
    error_logs = []
    for name in os.listdir(log_directory):
        log_path = os.path.join(log_directory, name)
        with open(log_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if 'error while loading shared libraries: libopenblas.so.0' in line:
                    raise ThirdpartyError('libopenblas.so.0', open_blas=True)
                for libc_version in ['GLIBC_2.27', 'GLIBCXX_3.4.20']:
                    if libc_version in line:
                        raise ThirdpartyError(libc_version, libc=True)
                if 'sox FAIL formats' in line:
                    f = line.split(' ')[-1]
                    raise ThirdpartyError(f, sox=True)
                if line.startswith('ERROR') or line.startswith('ASSERTION_FAILED'):
                    error_logs.append(log_path)
                    break
    if error_logs:
        raise KaldiProcessingError(error_logs)


def log_kaldi_errors(error_logs: List[str], logger: logging.Logger) -> None:
    logger.debug('There were {} kaldi processing files that had errors:'.format(len(error_logs)))
    for path in error_logs:
        logger.debug('')
        logger.debug(path)
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                logger.debug('\t' + line.strip())


def get_available_models(model_type: str) -> List[str]:
    from .config import TEMP_DIR
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', model_type)
    os.makedirs(pretrained_dir, exist_ok=True)
    available = []
    model_class = MODEL_TYPES[model_type]
    for f in os.listdir(pretrained_dir):
        if model_class is None or model_class.valid_extension(f):
            available.append(os.path.splitext(f)[0])
    return available


def guess_model_type(path: str) -> List[str]:
    ext = os.path.splitext(path)[1]
    if not ext:
        return []
    possible = []
    for m, mc in MODEL_TYPES.items():
        if ext in mc.extensions:
            possible.append(m)
    return possible


def get_available_acoustic_languages() -> List[str]:
    return get_available_models('acoustic')


def get_available_g2p_languages() -> List[str]:
    return get_available_models('g2p')


def get_available_ivector_languages() -> List[str]:
    return get_available_models('ivector')


def get_available_lm_languages() -> List[str]:
    return get_available_models('language_model')


def get_available_dict_languages() -> List[str]:
    return get_available_models('dictionary')


def get_pretrained_path(model_type: str, name: str) -> str:
    from .config import TEMP_DIR
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', model_type)
    model_class = MODEL_TYPES[model_type]
    return model_class.generate_path(pretrained_dir, name)


def get_pretrained_acoustic_path(name: str) -> str:
    return get_pretrained_path('acoustic', name)


def get_pretrained_ivector_path(name: str) -> str:
    return get_pretrained_path('ivector', name)


def get_pretrained_language_model_path(name: str) -> str:
    return get_pretrained_path('language_model', name)


def get_pretrained_g2p_path(name: str) -> str:
    return get_pretrained_path('g2p', name)


def get_dictionary_path(name: str) -> str:
    return get_pretrained_path('dictionary', name)


class CustomFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .config import load_global_config
        config = load_global_config()
        self.width = config['terminal_width']
        use_colors = config.get('terminal_colors', True)
        red = ''
        green = ''
        yellow = ''
        blue = ''
        reset = ''
        if use_colors:
            red = Fore.RED
            green = Fore.GREEN
            yellow = Fore.YELLOW
            blue = Fore.CYAN
            reset = Style.RESET_ALL

        self.FORMATS = {
            logging.DEBUG: (f'{blue}DEBUG{reset} - ', '%(message)s'),
            logging.INFO: (f'{green}INFO{reset} - ', '%(message)s'),
            logging.WARNING: (f'{yellow}WARNING{reset} - ', '%(message)s'),
            logging.ERROR: (f'{red}ERROR{reset} - ', '%(message)s'),
            logging.CRITICAL: (f'{red}CRITICAL{reset} - ', '%(message)s')
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        return textwrap.fill(record.getMessage(), initial_indent=log_fmt[0], subsequent_indent=' '* len(log_fmt[0]), width=self.width)


def setup_logger(identifier, output_directory, console_level='info'):
    os.makedirs(output_directory, exist_ok=True)
    log_path = os.path.join(output_directory, identifier + '.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logging.getLogger(identifier)
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_path, encoding='utf8')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, console_level.upper()))
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)

    return logger


def log_config(logger, config):
    stream = yaml.dump(config)
    logger.debug(stream)