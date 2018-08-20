import os
import re

base_dir = os.path.dirname(os.path.abspath(__file__))

report_path = os.path.join(base_dir, 'report.txt')

used_binaries = set()

with open(report_path, 'r') as f:
    for line in f:
        m = re.search('thirdparty_binary\(\'([-\w]*?)\'\)', line)
        if not m:
            continue
        used_binaries.add(m.groups()[0])

print(sorted(used_binaries))
print(len(used_binaries))