import sys
import shutil, os
import argparse
import subprocess
import re

if sys.platform == 'win32':
    exe_ext = '.exe'
    lib_ext = '.dll'
elif sys.platform == 'darwin':
    exe_ext = ''
    lib_ext = ['.dylib']
else:
    exe_ext = ''
    lib_ext = ['.so', '.so.1']

included_filenames = ['compute-mfcc-feats', 'copy-feats', 'gmm-acc-stats-ali',
                      'compile-train-graphs', 'compile-train-graphs-fsts', 'align-equal-compiled', 'gmm-acc-stats-ali',
                      'gmm-align-compiled', 'gmm-boost-silence', 'linear-to-nbest',
                      'lattice-align-words', 'nbest-to-ctm', 'lattice-to-phone-lattice',
                      'acc-tree-stats', 'sum-tree-stats', 'convert-ali',
                      'weight-silence-post', 'gmm-est-fmllr', 'compose-transforms',
                      'transform-feats', 'gmm-est', 'gmm-sum-accs', 'gmm-init-mono',
                      'cluster-phones', 'compile-questions', 'build-tree', 'gmm-init-model',
                      'gmm-mixup', 'gmm-info', 'fstcompile', 'fstarcsort', 'fstcopy', 'dot', 'compute-cmvn-stats',
                      'apply-cmvn', 'add-deltas', 'feat-to-dim', 'subset-feats',
                      'extract-segments', 'openblas', 'openfst64', 'gmm-latgen-faster',
                      'draw-tree', 'fstdraw', 'show-transitions', 'ali-to-post', 'farcompilestrings',
                      'acc-lda', 'est-lda', 'gmm-acc-mllt', 'est-mllt',
                      'gmm-transform-means', 'ali-to-phones', 'matrix-sum',
                      'apply-cvmn-online', 'gmm-global-init-from-feats', 'gmm-gselect',
                      'gmm-global-acc-stats', 'gmm-global-est', 'splice-feats',
                      'ivector-extractor-init', 'gmm-global-get-post', 'scale-post',
                      'ivector-extractor-sum-accs', 'ivector-extractor-est',
                      'ivector-extract-online-2', 'sum-lda-accs,', 'nnet-get-feature-transform',
                      'paste-feats', 'copy-int-vector', 'nnet-get-egs', 'nnet-subset-egs',
                      'nnet-copy-egs', 'nnet-shuffle-egs', 'nnet-train-transitions',
                      'post-to-tacc', 'nnet-am-info', 'nnet-init', 'nnet-compute-from-egs',
                      'matrix-sum-rows', 'vector-sum', 'nnet-adjust-priors', 'nnet-align-compiled',
                      'nnet-relabel-egs', 'nnet-compute-prob', 'nnet-show-progress',
                      'nnet-am-copy', 'nnet-train', 'nnet-am-average', 'nnet-am-fix',
                      'nnet-combine-fast', 'online2-wav-nnet2-latgen-threaded',
                      'online2-wav-ivector-config-latgen-faster', 'apply-cmvn-online',
                      'subsample-feats', 'gmm-global-sum-accs', 'gmm-global-to-fgmm',
                      'ivector-extractor-acc-stats', 'tree-info', 'am-info', 'ali-to-pdf',
                      'nnet-am-init', 'sum-lda-accs', 'matrix-dim', 'nnet-info',
                      'nnet-train-parallel', 'nnet-insert', 'ivector-extract-online2',
                      'ivector-randomize', 'feat-to-len']

linux_libraries = ['libfst.so.7', 'libfstfar.so.7', 'libngram.so.2',
                   'libfstscript.so.7', 'libfstfarscript.so.7',
                   'libkaldi-hmm.so', 'libkaldi-util.so', 'libkaldi-thread.so',
                   'libkaldi-base.so', 'libkaldi-tree.so', 'libkaldi-matrix.so',
                   'libkaldi-feat.so', 'libkaldi-transform.so','libkaldi-lm.so',
                   'libkaldi-gmm.so', 'libkaldi-lat.so', 'libkaldi-decoder.so',
                   'libkaldi-fstext.so']
included_libraries = {'linux': linux_libraries,
                      'win32': ['openfst64.dll', 'libopenblas.dll'],
                      'darwin': ['libfst.7.dylib', 'libfstfarscript.7.dylib', 'libfstscript.7.dylib',
                                 'libfstfar.7.dylib', 'libfstngram.7.dylib',
                                 'libkaldi-hmm.dylib', 'libkaldi-util.dylib', 'libkaldi-thread.dylib',
                                 'libkaldi-base.dylib', 'libkaldi-tree.dylib', 'libkaldi-matrix.dylib',
                                 'libkaldi-feat.dylib', 'libkaldi-transform.dylib', 'libkaldi-lm.dylib',
                                 'libkaldi-gmm.dylib', 'libkaldi-lat.dylib', 'libkaldi-decoder.dylib',
                                 'libkaldi-fstext.dylib']}

dylib_pattern = re.compile(r'\s*(.*)\s+\(')


def CollectBinaries(directory):
    outdirectory = os.path.dirname(os.path.realpath(__file__))
    bin_out = os.path.join(outdirectory, 'bin')
    os.makedirs(bin_out, exist_ok=True)
    tools_dir = os.path.join(directory, 'tools')
    for root, dirs, files in os.walk(tools_dir):
        for name in files:
            if os.path.islink(os.path.join(root, name)):
                continue
            ext = os.path.splitext(name)
            (key, value) = ext
            if value == exe_ext and key in included_filenames:
                bin_name = os.path.join(bin_out, name)
                shutil.copy(os.path.join(root, name), bin_out)
                if sys.platform == 'darwin':
                    p = subprocess.Popen(['otool', '-L', bin_name], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
                    output, err = p.communicate()
                    rc = p.returncode
                    output = output.decode()
                    libs = dylib_pattern.findall(output)
                    for l in libs:
                        if l.startswith('/usr') and not l.startswith('/usr/local'):
                            continue
                        lib = os.path.basename(l)
                        subprocess.call(['install_name_tool', '-change', l, '@loader_path/' + lib, bin_name])
            elif sys.platform == 'win32' and name in included_libraries[sys.platform]:
                shutil.copy(os.path.join(root, name), bin_out)
            elif sys.platform != 'win32':
                c = False
                for l in included_libraries[sys.platform]:
                    if name.startswith(l):
                        c = True
                        new_name = l
                if c:
                    bin_name = os.path.join(bin_out, new_name)
                    shutil.copyfile(os.path.join(root, name), bin_name)
                    if sys.platform == 'darwin':
                        p = subprocess.Popen(['otool', '-L', bin_name], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
                        output, err = p.communicate()
                        rc = p.returncode
                        output = output.decode()
                        libs = dylib_pattern.findall(output)
                        for l in libs:
                            if l.startswith('/usr') and not l.startswith('/usr/local'):
                                continue
                            lib = os.path.basename(l)
                            subprocess.call(['install_name_tool', '-change', l, '@loader_path/' + lib, bin_name])

    if sys.platform == 'win32':
        src_dir = directory
    else:
        src_dir = os.path.join(directory, 'src')
    for root, dirs, files in os.walk(src_dir, followlinks=True):
        cur_dir = os.path.basename(root)
        for name in files:
            if os.path.islink(os.path.join(root, name)):
                continue
            ext = os.path.splitext(name)
            (key, value) = ext
            bin_name = os.path.join(bin_out, name)
            if key == 'libkaldi-hmm':
                print(name, value == lib_ext)
            if value == exe_ext:
                if key not in included_filenames:
                    continue
                shutil.copy(os.path.join(root, name), bin_out)
            elif name in included_libraries[sys.platform]:
                shutil.copy(os.path.join(root, name), bin_out)
            else:
                continue
            if sys.platform == 'darwin':
                p = subprocess.Popen(['otool', '-L', bin_name], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                output, err = p.communicate()
                rc = p.returncode
                output = output.decode()
                libs = dylib_pattern.findall(output)
                for l in libs:
                    if (l.startswith('/usr') and not l.startswith('/usr/local')) or l.startswith('/System'):
                        continue
                    lib = os.path.basename(l)
                    subprocess.call(['install_name_tool', '-change', l, '@loader_path/' + lib, bin_name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()
    directory = os.path.expanduser(args.dir)
    CollectBinaries(directory)
