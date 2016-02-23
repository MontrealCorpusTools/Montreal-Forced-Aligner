import os

def thirdparty_binary(binary_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    thirdparty_dir = os.path.join(base_dir, 'thirdparty')
    bin_path = os.path.join(thirdparty_dir, binary_name)
    if not os.path.exists(bin_path):
        return binary_name
    return bin_path
