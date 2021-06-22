import sys
import shutil
import os
import stat
import subprocess
import re

from ..config import TEMP_DIR

if sys.platform == 'win32':
    exe_ext = '.exe'
elif sys.platform == 'darwin':
    exe_ext = ''
    open_blas_library = 'libopenblas.dylib'
else:
    exe_ext = ''
    open_blas_library = 'libopenblas.so.0'

alignment_filenames = ['acc-lda', 'acc-tree-stats', 'add-deltas', 'ali-to-pdf', 'ali-to-post', 'align-equal-compiled',
                       'append-vector-to-feats', 'apply-cmvn', 'build-tree', 'cluster-phones', 'compile-questions',
                       'compile-train-graphs', 'compile-train-graphs-fsts', 'compose-transforms', 'compute-cmvn-stats',
                       'compute-mfcc-feats', 'convert-ali', 'copy-feats', 'est-lda', 'est-mllt',
                       'extract-segments', 'feat-to-dim', 'feat-to-len', 'gmm-acc-mllt', 'gmm-acc-stats-ali',
                       'gmm-align-compiled',
                       'gmm-boost-silence', 'gmm-est', 'gmm-est-fmllr', 'gmm-info', 'gmm-init-model', 'gmm-init-mono',
                       'gmm-latgen-faster', 'gmm-mixup',
                       'gmm-sum-accs', 'gmm-transform-means',
                       'lattice-align-words', 'lattice-oracle', 'lattice-to-phone-lattice', 'linear-to-nbest',
                       'nbest-to-ctm', 'paste-feats', 'post-to-weights', 'select-feats',
                       'show-transitions',
                       'splice-feats', 'sum-lda-accs', 'sum-tree-stats', 'transform-feats',
                       'tree-info', 'weight-silence-post', 'subset-feats']

train_dict_filenames = ['nbest-to-prons']

transcribe_filenames = [
    'fstaddselfloops', 'arpa2fst', 'fsttablecompose', 'fstdeterminizestar', 'fstminimizeencoded',
    'fstpushspecial', 'fstcomposecontext', 'make-h-transducer', 'fstrmsymbols', 'fstrmepslocal',
    'add-self-loops', 'lattice-scale', 'lattice-add-penalty', 'lattice-best-path', 'lattice-to-post',
    'gmm-post-to-gpost', 'gmm-est-fmllr-gpost', 'lattice-determinize-pruned', 'gmm-rescore-lattice',
    'gmm-latgen-faster-parallel', 'lattice-determinize-pruned-parallel', 'arpa-to-const-arpa',
    'lattice-lmrescore', 'lattice-lmrescore-const-arpa', 'lattice-compose', 'lattice-determinize',
    'lattice-add-trans-probs', 'lattice-lmrescore-pruned'
]

speaker_diarization_filenames = [
    'compute-vad', 'apply-cmvn-sliding',
    'subsample-feats', 'scale-post',
    'ivector-extract', 'ivector-extractor-acc-stats', 'ivector-extractor-est', 'ivector-extractor-init',
    'ivector-extractor-sum-accs', 'gmm-global-acc-stats', 'gmm-global-est',
    'gmm-global-get-post', 'gmm-global-init-from-feats', 'gmm-global-sum-accs', 'gmm-global-to-fgmm',
    'gmm-gselect', 'select-voiced-frames',
]

included_filenames = alignment_filenames + train_dict_filenames + transcribe_filenames + speaker_diarization_filenames
if sys.platform == 'win32':
    included_filenames += ['fstcompile', 'fstarcsort', 'fstconvert', 'fstproject']

linux_libraries = [#'libfst.so.13', 'libfstfar.so.13',
                   #'libfstscript.so.13', 'libfstfarscript.so.13',
                   #'libkaldi-hmm.so', 'libkaldi-util.so',
                   #'libkaldi-base.so', 'libkaldi-tree.so',
                   #'libkaldi-feat.so', 'libkaldi-transform.so', 'libkaldi-lm.so',
                   #'libkaldi-gmm.so', 'libkaldi-lat.so', 'libkaldi-decoder.so',
                   #'libkaldi-fstext.so', 'libkaldi-ivector.so'
]
included_libraries = {'linux': linux_libraries,
                      'win32': [  # 'openfst64.dll',
                          'libgcc_s_seh-1.dll', 'libgfortran-3.dll',
                          'libquadmath-0.dll', 'libopenblas.dll'],
                      'darwin': [#'libfst.13.dylib', 'libfstfarscript.13.dylib', 'libfstscript.13.dylib',
                                 #'libfstfar.13.dylib', 'libfstngram.13.dylib',
                                 'libkaldi-hmm.dylib', 'libkaldi-util.dylib', 'libkaldi-thread.dylib',
                                 'libkaldi-base.dylib', 'libkaldi-tree.dylib', 'libkaldi-matrix.dylib',
                                 'libkaldi-feat.dylib', 'libkaldi-transform.dylib', 'libkaldi-lm.dylib',
                                 'libkaldi-gmm.dylib', 'libkaldi-lat.dylib', 'libkaldi-decoder.dylib',
                                 'libkaldi-fstext.dylib',
                                 'libkaldi-chain.dylib',
                                 'libkaldi-ivector.dylib']}

dylib_pattern = re.compile(r'\s*(.*)\s+\(')


def collect_kaldi_binaries(directory):
    bin_out = os.path.join(TEMP_DIR, 'thirdparty', 'bin')
    os.makedirs(bin_out, exist_ok=True)

    for root, dirs, files in os.walk(directory, followlinks=True):
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
                try:
                    shutil.copy(os.path.join(root, name), bin_out)
                    st = os.stat(bin_out)
                    os.chmod(bin_out, st.st_mode | stat.S_IEXEC)
                except OSError:
                    pass
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


def validate_binaries(file_list):
    bin_out = os.path.join(TEMP_DIR, 'thirdparty', 'bin')
    if not os.path.exists(bin_out):
        print('The folder {} does not exist'.format(bin_out))
        return False
    bin_files = os.listdir(bin_out)
    not_found = []
    erroring = []
    # for lib_file in included_libraries[plat]:
    #    if lib_file not in bin_files:
    #        not_found.append(lib_file)
    for bin_file in file_list:
        bin_file += exe_ext
        if bin_file not in bin_files:
            not_found.append(bin_file)
            try:
                pipes = subprocess.Popen([os.path.join(bin_out, bin_file), '--help'], stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True)
                std_out, std_err = pipes.communicate()
                if std_err:
                    print(std_out)
                    print(std_err)
                    erroring.append(bin_file)
            except:
                pass
    if not_found:
        print('The following kaldi binaries were not found in {}: {}'.format(bin_out, ', '.join(sorted(not_found))))
        return False
    if erroring:
        print('The following kaldi binaries had errors in running: {}'.format(', '.join(sorted(erroring))))
        return False
    print('All required kaldi binaries were found!')
    return True


def validate_alignment_binaries():
    return validate_binaries(alignment_filenames)


def validate_transcribe_binaries():
    return validate_binaries(transcribe_filenames)


def validate_train_dictionary_binaries():
    return validate_binaries(train_dict_filenames)


def validate_kaldi_binaries():
    return validate_binaries(included_filenames)
