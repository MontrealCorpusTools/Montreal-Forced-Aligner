import os
import sys
from collections import defaultdict

def thirdparty_binary(binary_name):
    if getattr(sys, 'frozen', False) and sys.platform == 'win32':
        base_dir = os.path.dirname(sys.executable)
        thirdparty_dir = os.path.join(base_dir, 'thirdparty')
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        thirdparty_dir = os.path.join(base_dir, 'thirdparty')
    bin_path = os.path.join(thirdparty_dir, 'bin', binary_name)
    if sys.platform == 'win32':
        bin_path += '.exe'
    if not os.path.exists(bin_path):
        return binary_name
    return bin_path

def make_path_safe(path):
    return '"{}"'.format(path)

def load_text(path):
    with open(path, 'r', encoding = 'utf8') as f:
        text = f.read().strip().lower()
    return text

def make_safe(element):
    if isinstance(element, list):
        return ' '.join(map(make_safe, element))
    return str(element)
