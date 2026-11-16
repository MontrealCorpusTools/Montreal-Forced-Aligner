"""Prepare public inputs, run traced CLI comparisons, or export one read-only DB.

Run this script in the same Python environment as MFA (validated with 3.4.1).
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sqlite3
import subprocess
import sys
import traceback
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = '93411fc0df3bc770dfd34c4282dc136fdf6e8dcb'
PR = '8545fd38f47b81f8a1743c8c9f8ab7b323e853eb'


def selected_function(path):
    from itertools import groupby
    from montreal_forced_aligner import textgrid

    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'construct_textgrid_output')
    namespace = dict(vars(textgrid), groupby=groupby)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['construct_textgrid_output'], node


def traced_cli():
    sys.argv.pop(1)
    source, trace, cli = (Path(sys.argv.pop(1)) for _ in range(3))
    chosen, node = selected_function(source)
    from montreal_forced_aligner import textgrid

    old = textgrid.construct_textgrid_output

    def traced(*args, **kwargs):
        with trace.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'pid': os.getpid(), 'source': source.name,
                               'function_ast_sha256': hashlib.sha256(
                                   ast.dump(node, include_attributes=False).encode()).hexdigest()}) + '\n')
        yield from chosen(*args, **kwargs)

    aliases = []
    for name, module in list(sys.modules.items()):
        if name.startswith('montreal_forced_aligner') and getattr(
                module, 'construct_textgrid_output', None) is old:
            setattr(module, 'construct_textgrid_output', traced)
            aliases.append(name)
    assert 'montreal_forced_aligner.alignment.base' in aliases
    assert 'montreal_forced_aligner.alignment.multiprocessing' in aliases
    print('VERIFIED_EXPORT_ALIASES', json.dumps(aliases), flush=True)
    runpy.run_path(str(cli), run_name='__main__')


def fetch(url):
    return urlopen(Request(url, headers={'User-Agent': 'MFA-PR979-validation'}), timeout=60)


def prepare(args):
    manifest = json.loads(Path(__file__).with_name('inputs.json').read_text(encoding='utf-8'))[args.case]
    current = json.load(fetch('https://huggingface.co/api/datasets/' + manifest['dataset']))
    if current['sha'] != manifest['revision']:
        raise RuntimeError('Dataset revision changed. Obtain the recorded revision and source file paths from inputs.json; do not assume row indices are unchanged.')
    args.output.mkdir(parents=True, exist_ok=False)
    for row in manifest['rows']:
        query = {k: manifest[k] for k in ['dataset', 'config', 'split']}
        query.update(offset=row['row_idx'], length=1)
        record = json.load(fetch('https://datasets-server.huggingface.co/rows?' + urlencode(query)))['rows'][0]['row']
        assert record[manifest['text_field']].strip() == row['text']
        assert record['file_path'] == row['file_path']
        audio = record['audio']
        url = audio[0]['src'] if isinstance(audio, list) else audio['src']
        data = fetch(url).read()
        assert hashlib.sha256(data).hexdigest() == row['download_sha256']
        folder = args.output / row['speaker']
        folder.mkdir(exist_ok=True)
        downloaded = folder / (row['file'] + '.download')
        downloaded.write_bytes(data)
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                        '-i', str(downloaded), '-ac', '1', '-ar', '16000', '-c:a',
                        'pcm_s16le', str(folder / (row['file'] + '.wav'))], check=True)
        downloaded.unlink()
        (folder / (row['file'] + '.lab')).write_text(row['text'] + '\n', encoding='utf-8')


def run(args):
    cli = shutil.which('mfa')
    if not cli:
        raise RuntimeError('Activate the MFA environment first.')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for version, ref in [('base', BASE), ('pr', PR)]:
        source = args.output / (version + '_textgrid.py')
        source.write_text(subprocess.check_output(
            ['git', '-C', str(args.repo), 'show', ref + ':montreal_forced_aligner/textgrid.py'],
            text=True), encoding='utf-8')
        job = args.output / version
        job.mkdir()
        env = dict(os.environ, MFA_ROOT_DIR=str(job / 'mfa_root'),
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for key in ['PYTHONPATH', 'MFA_PATCH_TEXTGRID_EXPORT',
                    'MFA_PATCH_TEXTGRID_EXPORT_WITH_UNK', 'MFA_SYNTHESIZE_UNK']:
            env.pop(key, None)
        command = [sys.executable, str(Path(__file__).resolve()), '_cli', str(source),
                   str(job / 'export_trace.jsonl'), cli, 'align', str(args.corpus),
                   str(args.dictionary), str(args.acoustic_model), str(job / 'textgrids'),
                   '--num_jobs', '1', '--no_use_mp', '--temporary_directory', str(job / 'scratch'),
                   '--no_use_postgres', '--no_final_clean', '--clean', '--output_format',
                   'long_textgrid', '--uses_speaker_adaptation', 'false', '--no_tokenization']
        with (job / 'run.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        results.append({'version': version, 'exit_code': result.returncode,
                        'textgrids': len(list((job / 'textgrids').rglob('*.TextGrid')))})
    (args.output / 'cli_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


def export(args):
    import sqlalchemy
    from sqlalchemy.orm import Session

    digest = hashlib.sha256(args.database.read_bytes()).hexdigest()
    connection = sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True)
    batch = {i: (name, Path(relative or '.'), duration, Path(text))
             for i, name, relative, duration, text in connection.execute(
                 'select f.id,f.name,f.relative_path,s.duration,t.text_file_path '
                 'from file f join sound_file s on s.file_id=f.id '
                 'join text_file t on t.file_id=f.id order by f.id')}
    engine = sqlalchemy.create_engine('sqlite://', creator=lambda: connection,
                                      poolclass=sqlalchemy.pool.StaticPool)
    chosen, _ = selected_function(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    error = None
    try:
        with Session(engine) as session:
            list(chosen(session, batch, not args.no_cleanup, "'", args.output,
                        0.01, 'long_textgrid', args.include_original_text))
    except Exception as exc:
        error = type(exc).__name__
        (args.output / 'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
    assert hashlib.sha256(args.database.read_bytes()).hexdigest() == digest
    result = {'error': error, 'textgrids': len(list(args.output.rglob('*.TextGrid'))),
              'database_unchanged': True}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    engine.dispose()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_cli':
        return traced_cli()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Download the recorded public inputs using inputs.json')
    p.add_argument('--case', required=True, choices=['coraa_mupe_train', 'coraa_v11', 'nurc_sp', 'mupe_validation'])
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(action=prepare)
    p = sub.add_parser('run', help='Run both exact export functions through the full CLI')
    for name in ['repo', 'corpus', 'dictionary', 'acoustic-model', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.set_defaults(action=run)
    p = sub.add_parser('export', help='Export one existing read-only SQLite database')
    for name in ['database', 'source', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--no-cleanup', action='store_true')
    p.add_argument('--include-original-text', action='store_true')
    p.set_defaults(action=export)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.action(args)


if __name__ == '__main__':
    main()
