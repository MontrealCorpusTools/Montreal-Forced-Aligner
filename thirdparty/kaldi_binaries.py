import sys
import shutil, os, stat
import argparse
import subprocess
import re

if sys.platform == 'win32':
    exe_ext = '.exe'
elif sys.platform == 'darwin':
    exe_ext = ''
    open_blas_library = 'libopenblas.dylib'
else:
    exe_ext = ''
    open_blas_library = 'libopenblas.so.0'

included_filenames = ['acc-lda', 'acc-tree-stats', 'add-deltas', 'ali-to-pdf', 'ali-to-post', 'align-equal-compiled',
                      'append-vector-to-feats', 'apply-cmvn', 'build-tree', 'cluster-phones', 'compile-questions',
                      'compile-train-graphs', 'compile-train-graphs-fsts', 'compose-transforms', 'compute-cmvn-stats',
                      'compute-mfcc-feats', 'convert-ali', 'copy-feats', 'dot', 'est-lda', 'est-mllt',
                      'extract-segments', 'farcompilestrings', 'feat-to-dim', 'feat-to-len', 'fstarcsort', 'fstcompile',
                      'fstcopy', 'fstdraw', 'gmm-acc-mllt', 'gmm-acc-stats-ali', 'gmm-align-compiled',
                      'gmm-boost-silence', 'gmm-est', 'gmm-est-fmllr', 'gmm-global-acc-stats', 'gmm-global-est',
                      'gmm-global-get-post', 'gmm-global-init-from-feats', 'gmm-global-sum-accs', 'gmm-global-to-fgmm',
                      'gmm-gselect', 'gmm-info', 'gmm-init-model', 'gmm-init-mono', 'gmm-latgen-faster', 'gmm-mixup',
                      'gmm-sum-accs', 'gmm-transform-means', 'ivector-extract', 'ivector-extractor-acc-stats',
                      'ivector-extractor-est', 'ivector-extractor-init', 'ivector-extractor-sum-accs',
                      'lattice-align-words', 'lattice-oracle', 'lattice-to-phone-lattice', 'linear-to-nbest',
                      'matrix-sum-rows', 'nbest-to-ctm',
                      'nnet-adjust-priors', 'nnet-align-compiled', 'nnet-am-average', 'nnet-am-copy', 'nnet-am-info',
                      'nnet-am-init', 'nnet-am-mixup', 'nnet-compute-from-egs', 'nnet-compute-prob', 'nnet-copy-egs',
                      'nnet-get-egs', 'nnet-get-feature-transform', 'nnet-init', 'nnet-insert', 'nnet-relabel-egs',
                      'nnet-shuffle-egs', 'nnet-subset-egs', 'nnet-to-raw-nnet', 'nnet-train-parallel',
                      'nnet-train-transitions', 'paste-feats', 'post-to-weights', 'scale-post', 'select-feats',
                      'show-transitions',
                      'splice-feats', 'subsample-feats', 'sum-lda-accs', 'sum-tree-stats', 'transform-feats',
                      'tree-info', 'vector-sum', 'weight-silence-post']

linux_libraries = ['libfst.so.13', 'libfstfar.so.13', 'libngram.so.13',
                   'libfstscript.so.13', 'libfstfarscript.so.13',
                   'libkaldi-hmm.so', 'libkaldi-util.so', 'libkaldi-thread.so',
                   'libkaldi-base.so', 'libkaldi-tree.so', 'libkaldi-matrix.so',
                   'libkaldi-feat.so', 'libkaldi-transform.so', 'libkaldi-lm.so',
                   'libkaldi-gmm.so', 'libkaldi-lat.so', 'libkaldi-decoder.so',
                   'libkaldi-fstext.so', 'libkaldi-ivector.so', 'libkaldi-nnet2.so', 'libkaldi-cudamatrix.so']
included_libraries = {'linux': linux_libraries,
                      'win32': ['openfst64.dll', 'libgcc_s_seh-1.dll', 'libgfortran-3.dll',
                                'libquadmath-0.dll', 'libopenblas.dll'],
                      'darwin': ['libfst.13.dylib', 'libfstfarscript.13.dylib', 'libfstscript.13.dylib',
                                 'libfstfar.13.dylib', 'libfstngram.13.dylib', 'libkaldi-nnet.dylib',
                                 'libkaldi-hmm.dylib', 'libkaldi-util.dylib', 'libkaldi-thread.dylib',
                                 'libkaldi-base.dylib', 'libkaldi-tree.dylib', 'libkaldi-matrix.dylib',
                                 'libkaldi-feat.dylib', 'libkaldi-transform.dylib', 'libkaldi-lm.dylib',
                                 'libkaldi-gmm.dylib', 'libkaldi-lat.dylib', 'libkaldi-decoder.dylib',
                                 'libkaldi-fstext.dylib', 'libkaldi-rnnlm.dylib', 'libkaldi-nnet2.dylib',
                                 'libkaldi-nnet3.dylib', 'libkaldi-cudamatrix.dylib', 'libkaldi-chain.dylib',
                                 'libkaldi-ivector.dylib']}

dylib_pattern = re.compile(r'\s*(.*)\s+\(')


def collect_linux_tools_binaries(directory):
    outdirectory = os.path.dirname(os.path.realpath(__file__))
    bin_out = os.path.join(outdirectory, 'bin')
    os.makedirs(bin_out, exist_ok=True)
    tools_dir = os.path.join(directory, 'tools')
    openfst_dir = os.path.join(tools_dir, 'openfst')
    bin_dir = os.path.join(openfst_dir, 'bin')
    lib_dir = os.path.join(openfst_dir, 'lib')
    for name in os.listdir(bin_dir):
        if os.path.islink(os.path.join(bin_dir, name)):
            continue
        ext = os.path.splitext(name)
        (key, value) = ext
        if value == exe_ext and key in included_filenames:
            out_path = os.path.join(bin_out, name)
            in_path = os.path.join(bin_dir, name)
            if os.path.exists(out_path) and os.path.getsize(in_path) > os.path.getsize(out_path):
                continue  # Get the smallest file size when multiples exist
            shutil.copyfile(in_path, out_path)
            st = os.stat(out_path)
            os.chmod(out_path, st.st_mode | stat.S_IEXEC)
            if sys.platform == 'darwin':
                p = subprocess.Popen(['otool', '-L', out_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                output, err = p.communicate()
                rc = p.returncode
                output = output.decode()
                libs = dylib_pattern.findall(output)
                for l in libs:
                    if l.startswith('/usr') and not l.startswith('/usr/local'):
                        continue
                    lib = os.path.basename(l)
                    subprocess.call(['install_name_tool', '-change', l, '@loader_path/' + lib, out_path])
    for name in os.listdir(lib_dir):
        if name in included_libraries[sys.platform]:
            if sys.platform == 'win32':
                shutil.copy(os.path.join(lib_dir, name), bin_out)
            else:
                actual_lib = os.path.join(lib_dir, name)
                while os.path.islink(actual_lib):
                    linkto = os.readlink(actual_lib)
                    actual_lib = os.path.join(lib_dir, linkto)

                bin_name = os.path.join(bin_out, name)
                shutil.copyfile(actual_lib, bin_name)
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
    openblas_dir = os.path.join(tools_dir, 'OpenBLAS', 'install', 'lib')
    lib_file = os.path.join(openblas_dir, open_blas_library)
    out_lib = os.path.join(bin_out, open_blas_library)
    if os.path.islink(lib_file):
        linkto = os.readlink(lib_file)
        actual_lib = os.path.join(openblas_dir, linkto)
        shutil.copyfile(actual_lib, out_lib)
    else:
        shutil.copyfile(lib_file, out_lib)


def collect_kaldi_binaries(directory):
    outdirectory = os.path.dirname(os.path.realpath(__file__))
    bin_out = os.path.join(outdirectory, 'bin')
    os.makedirs(bin_out, exist_ok=True)
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
            if value == exe_ext:
                if key not in included_filenames:
                    continue
                shutil.copy(os.path.join(root, name), bin_out)
                st = os.stat(bin_out)
                os.chmod(bin_out, st.st_mode | stat.S_IEXEC)
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
    if sys.platform in ['linux', 'darwin']:
        collect_linux_tools_binaries(directory)
    collect_kaldi_binaries(directory)
