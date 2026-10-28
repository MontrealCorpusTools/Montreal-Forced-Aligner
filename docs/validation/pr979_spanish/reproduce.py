"""Reproduce the Spanish comparison inside an installed MFA environment.

Example:
  python reproduce.py --repo /path/to/Montreal-Forced-Aligner \
    --audio-dir /path/to/selected/wavs --dictionary spanish_mfa.dict \
    --acoustic-model spanish_mfa.zip --output /path/to/new/results

The input directory must contain the two <id>.wav files in samples.json.
The repository must contain the base and PR commits recorded below.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = '93411fc0df3bc770dfd34c4282dc136fdf6e8dcb'
PR = '8545fd38f47b81f8a1743c8c9f8ab7b323e853eb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--dictionary', type=Path, required=True)
    parser.add_argument('--acoustic-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    samples = json.loads(Path(__file__).with_name('samples.json').read_text(encoding='utf-8'))
    (output / 'inputs.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')
    versions = {}
    for version, commit in [('base', BASE), ('pr', PR)]:
        source = subprocess.check_output([
            'git', '-C', str(args.repo.resolve()), 'show',
            commit + ':montreal_forced_aligner/textgrid.py'], text=True)
        path = output / (version + '_textgrid.py')
        path.write_text(source, encoding='utf-8')
        versions[version] = path
    mfa = shutil.which('mfa')
    if not mfa:
        raise RuntimeError('Run this script in an environment with MFA installed.')
    # Shared wrapper replaces every imported alias and traces actual calls.
    wrapper = Path(__file__).resolve().parents[1] / 'pr979_public_corpora' / 'reproduce.py'
    for transcript in ['original', 'corrected']:
        corpus = output / 'corpus' / transcript
        for row in samples:
            folder = corpus / row['speaker']
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.audio_dir / (row['id'] + '.wav'), folder / (row['id'] + '.wav'))
            (folder / (row['id'] + '.lab')).write_text(row[transcript] + '\n', encoding='utf-8')
        for version, source in versions.items():
            job = output / 'runs' / (transcript + '_' + version)
            job.mkdir(parents=True)
            env = dict(os.environ, MFA_ROOT_DIR=str(job / 'mfa_root'),
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            for key in ['PYTHONPATH', 'MFA_PATCH_TEXTGRID_EXPORT', 'MFA_PATCH_TEXTGRID_EXPORT_WITH_UNK', 'MFA_SYNTHESIZE_UNK']:
                env.pop(key, None)
            command = [sys.executable, str(wrapper), '_cli', str(source), str(job / 'export_trace.jsonl'), mfa, 'align', str(corpus),
                       str(args.dictionary.resolve()), str(args.acoustic_model.resolve()),
                       str(job / 'textgrids'), '--num_jobs', '1', '--no_use_mp', '--temporary_directory',
                       str(job / 'scratch'), '--no_use_postgres', '--no_final_clean',
                       '--output_format', 'long_textgrid', '--include_original_text', '--clean']
            with (job / 'run.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Finished four MFA runs:', output)


if __name__ == '__main__':
    main()
