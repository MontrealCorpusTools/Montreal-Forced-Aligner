"""Command line functions for training new acoustic models"""
from __future__ import annotations

import typing
from pathlib import Path

import rich_click as click

from montreal_forced_aligner.acoustic_modeling import TrainableAligner
from montreal_forced_aligner.command_line.utils import (
    common_options,
    initialize_configuration,
    validate_corpus_directory,
    validate_dictionary,
    validate_g2p_model,
)
from montreal_forced_aligner.data import Language
from montreal_forced_aligner.models import DictionaryModel, MfaAlignmentModel

__all__ = ["train_acoustic_model_cli"]


@click.command(
    name="train",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Train a new acoustic model",
)
@click.argument("corpus_directory", type=click.UNPROCESSED, callback=validate_corpus_directory)
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.argument(
    "output_model_path", type=click.Path(file_okay=True, dir_okay=True, path_type=Path)
)
@click.option(
    "--output_directory",
    help="Path to save alignments.",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for training. See "
    "https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic for examples.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--speaker_characters",
    "-s",
    help="Number of characters of file names to use for determining speaker, "
    "default is to use directory names.",
    type=str,
    default="0",
)
@click.option(
    "--audio_directory",
    "-a",
    help="Audio directory root to use for finding audio files.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--phone_set",
    "phone_set_type",
    help="DEPRECATED, please use --phone_groups_path to specify phone groups instead.",
    default="UNKNOWN",
    type=click.Choice(["UNKNOWN", "AUTO", "MFA", "IPA", "ARPA", "PINYIN"]),
)
@click.option(
    "--phone_groups_path",
    "phone_groups_path",
    help="Path to yaml file defining phone groups. See "
    "https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic/phone_groups for examples.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--rules_path",
    "rules_path",
    help="Path to yaml file defining phonological rules. See "
    "https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic/rules for examples.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--topology_path",
    "topology_path",
    help="Path to yaml file defining topologies. See "
    "https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic/topologies for examples.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output_format",
    help="Format for aligned output files (default is long_textgrid).",
    default="long_textgrid",
    type=click.Choice(["long_textgrid", "short_textgrid", "json", "csv"]),
)
@click.option(
    "--include_original_text",
    is_flag=True,
    help="Flag to include original utterance text in the output.",
    default=False,
)
@click.option(
    "--language",
    "language",
    help="Language to use for spacy tokenizers and other preprocessing of language data.",
    default=Language.unknown.name,
    type=click.Choice([x.name for x in Language]),
)
@click.option(
    "--g2p_model_path",
    "g2p_model_path",
    help="Path to G2P model to use for OOV items.",
    type=click.UNPROCESSED,
)
@click.option(
    "--metadata_path",
    help="Path to additional metadata json file to use for creating the model card.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--ignore_oovs",
    is_flag=True,
    help="Ignore any file that contains OOV tokens by marking its utterances as ignored.",
    default=False,
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def train_acoustic_model_cli(context, **kwargs) -> None:
    """
    Train a new acoustic model on a corpus and optionally export alignments
    """
    initialize_configuration(context)
    config_path = kwargs.get("config_path", None)
    metadata_path = kwargs.get("metadata_path", None)
    output_model_path = kwargs.get("output_model_path", None)
    hf_train = False
    if output_model_path is not None:
        output_model_path = Path(output_model_path)
        if not output_model_path.suffix:
            hf_train = True

    output_directory = kwargs.get("output_directory", None)
    corpus_directory = kwargs["corpus_directory"].absolute()
    dictionary = kwargs["dictionary_path"]
    g2p_model_path: typing.Optional[Path] = kwargs.get("g2p_model_path", None)
    if kwargs.get("phone_set_type", "UNKNOWN") != "UNKNOWN":
        import warnings

        warnings.warn(
            "The flag `--phone_set` is deprecated, please use a yaml file for phone groups passed to "
            "`--phone_groups_path`.  See "
            "https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic/phone_groups "
            "for example phone group configurations that have been used in training MFA models."
        )
    g2p_model = None
    if g2p_model_path:
        g2p_model = validate_g2p_model(context, kwargs, g2p_model_path)
    extra_kwargs = TrainableAligner.parse_parameters(config_path, context.params, context.args)
    extra_kwargs["ignore_oovs"] = kwargs.get("ignore_oovs", False)
    trainer = TrainableAligner(
        corpus_directory=corpus_directory,
        dictionary=dictionary,
        g2p_model=g2p_model,
        **extra_kwargs,
    )
    try:
        trainer.train()
        if output_model_path is not None:
            trainer.export_model(output_model_path)
            if hf_train:
                model = MfaAlignmentModel(output_model_path.name, output_model_path)
                model.generate_model_card(trainer, metadata_path)
                if trainer.g2p_model is None:
                    from montreal_forced_aligner.g2p.phonetisaurus_trainer import (
                        PhonetisaurusTrainer,
                    )

                    for d in trainer.dictionary_model.load_dictionary_paths():
                        if not isinstance(d, DictionaryModel):
                            continue
                        name = Path(d.path).stem
                        if model.g2p_model_directory.joinpath(name).exists():
                            continue
                        g2p_trainer = PhonetisaurusTrainer(
                            dictionary=d,
                            evaluation_mode=True,
                            **PhonetisaurusTrainer.parse_parameters(
                                config_path, context.params, context.args
                            ),
                        )
                        g2p_trainer.setup()
                        g2p_trainer.train()
                        model.add_g2p_model(g2p_trainer.working_directory, g2p_trainer.meta, name)
                else:
                    if isinstance(trainer.g2p_model, dict):
                        for k, v in trainer.g2p_model.items():
                            name = Path(k).stem
                            model.add_g2p_model(v.directory, v.meta, name)
                    else:
                        name = Path(g2p_model_path).stem
                        model.add_g2p_model(
                            trainer.g2p_model.directory, trainer.g2p_model.meta, name
                        )

        if output_directory is not None:
            trainer.export_files(
                output_directory,
                kwargs["output_format"],
                include_original_text=kwargs["include_original_text"],
            )
    except Exception:
        trainer.dirty = True
        raise
    finally:
        trainer.cleanup()
