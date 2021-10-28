from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from ..dictionary import Dictionary
    from argparse import Namespace
import os
import shutil
from montreal_forced_aligner.g2p.trainer import PyniniTrainer as Trainer
from montreal_forced_aligner.dictionary import Dictionary
from montreal_forced_aligner.config import TEMP_DIR
from montreal_forced_aligner.config.train_g2p_config import train_g2p_yaml_to_config, load_basic_train_g2p_config
from montreal_forced_aligner.command_line.utils import validate_model_arg


def train_g2p(args: Namespace, unknown_args: Optional[list]=None) -> None:
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    if args.clean:
        shutil.rmtree(os.path.join(temp_dir, 'G2P'), ignore_errors=True)
        shutil.rmtree(os.path.join(temp_dir, 'models', 'G2P'), ignore_errors=True)
    if args.config_path:
        train_config = train_g2p_yaml_to_config(args.config_path)
    else:
        train_config = load_basic_train_g2p_config()
    train_config.use_mp = not args.disable_mp
    if unknown_args:
        train_config.update_from_unknown_args(unknown_args)
    dictionary = Dictionary(args.dictionary_path, '')
    t = Trainer(dictionary, args.output_model_path, temp_directory=temp_dir, train_config=train_config, num_jobs=args.num_jobs,
                verbose=args.verbose)
    if args.validate:
        t.validate()
    else:
        t.train()


def validate(args: Namespace) -> None:
    args.dictionary_path = validate_model_arg(args.dictionary_path, 'dictionary')


def run_train_g2p(args: Namespace, unknown: Optional[list]=None) -> None:
    validate(args)
    train_g2p(args, unknown)

