"""Command line functions for transcribing corpora"""
from __future__ import annotations

import sys
from pathlib import Path

import rich_click as click
from kalpy.data import Segment

from montreal_forced_aligner import config
from montreal_forced_aligner.command_line.utils import (
    common_options,
    validate_acoustic_model,
    validate_dictionary,
    validate_language_model,
)
from montreal_forced_aligner.data import Language
from montreal_forced_aligner.online.transcription import transcribe_utterance_online_whisper
from montreal_forced_aligner.transcription.transcriber import (
    SpeechbrainTranscriber,
    Transcriber,
    WhisperTranscriber,
)
from montreal_forced_aligner.utils import mfa_open

__all__ = ["transcribe_corpus_cli", "transcribe_speechbrain_cli", "transcribe_whisper_cli"]


@click.command(
    name="transcribe",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Transcribe audio files",
)
@click.argument(
    "corpus_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.argument("acoustic_model_path", type=click.UNPROCESSED, callback=validate_acoustic_model)
@click.argument("language_model_path", type=click.UNPROCESSED, callback=validate_language_model)
@click.argument(
    "output_directory", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for transcription.",
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
    "--output_type",
    help="Flag for outputting transcription text or alignments.",
    default="transcription",
    type=click.Choice(["transcription", "alignment"]),
)
@click.option(
    "--output_format",
    help="Format for aligned output files (default is long_textgrid).",
    default="long_textgrid",
    type=click.Choice(["long_textgrid", "short_textgrid", "json", "csv"]),
)
@click.option(
    "--evaluate",
    "evaluation_mode",
    is_flag=True,
    help="Evaluate the transcription against golden texts.",
    default=False,
)
@click.option(
    "--include_original_text",
    is_flag=True,
    help="Flag to include original utterance text in the output.",
    default=False,
)
@click.option(
    "--language_model_weight",
    help="Specific language model weight to use in evaluating transcriptions, defaults to 16.",
    type=int,
    default=16,
)
@click.option(
    "--word_insertion_penalty",
    help="Specific word insertion penalty between 0.0 and 1.0 to use in evaluating transcription, defaults to 1.0.",
    type=float,
    default=1.0,
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def transcribe_corpus_cli(context, **kwargs) -> None:
    """
    Transcribe utterances using an acoustic model, language model, and pronunciation dictionary.
    """
    if kwargs.get("profile", None) is not None:
        config.profile = kwargs.pop("profile")
    config.update_configuration(kwargs)

    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"].absolute()
    acoustic_model_path = kwargs["acoustic_model_path"]
    language_model_path = kwargs["language_model_path"]
    dictionary_path = kwargs["dictionary_path"]
    output_directory = kwargs["output_directory"]
    output_format = kwargs["output_format"]
    include_original_text = kwargs["include_original_text"]
    transcriber = Transcriber(
        corpus_directory=corpus_directory,
        dictionary_path=dictionary_path,
        acoustic_model_path=acoustic_model_path,
        language_model_path=language_model_path,
        **Transcriber.parse_parameters(config_path, context.params, context.args),
    )
    try:
        transcriber.setup()
        transcriber.transcribe()
        transcriber.export_files(
            output_directory,
            output_format=output_format,
            include_original_text=include_original_text,
        )
    except Exception:
        transcriber.dirty = True
        raise
    finally:
        transcriber.cleanup()


@click.command(
    name="transcribe_speechbrain",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Transcribe utterances using an ASR model trained by SpeechBrain",
)
@click.argument(
    "corpus_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "language",
    type=click.Choice(
        sorted(
            [
                "arabic",
                "german",
                "english",
                "spanish",
                "french",
                "italian",
                "kinyarwanda",
                "portuguese",
                "mandarin",
            ]
        )
    ),
)
@click.argument(
    "output_directory", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--architecture",
    help="ASR model architecture",
    default=SpeechbrainTranscriber.ARCHITECTURES[0],
    type=click.Choice(SpeechbrainTranscriber.ARCHITECTURES),
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for transcription.",
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
    "--cuda/--no_cuda",
    "cuda",
    help="Flag for using CUDA for Whisper's model",
    default=False,
)
@click.option(
    "--evaluate",
    "evaluation_mode",
    is_flag=True,
    help="Evaluate the transcription against golden texts.",
    default=False,
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def transcribe_speechbrain_cli(context, **kwargs) -> None:
    """
    Transcribe utterances using an ASR model trained by SpeechBrain.
    """
    if kwargs.get("profile", None) is not None:
        config.profile = kwargs.pop("profile")
    config.update_configuration(kwargs)

    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"].absolute()
    output_directory = kwargs["output_directory"]
    transcriber = SpeechbrainTranscriber(
        corpus_directory=corpus_directory,
        **SpeechbrainTranscriber.parse_parameters(config_path, context.params, context.args),
    )
    try:
        transcriber.setup()
        transcriber.transcribe()
        transcriber.export_files(output_directory)
    except Exception:
        transcriber.dirty = True
        raise
    finally:
        transcriber.cleanup()


@click.command(
    name="transcribe_whisper",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Transcribe utterances using a Whisper ASR model via faster-whisper",
)
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
)
@click.argument("output_path", type=click.Path(file_okay=True, dir_okay=True, path_type=Path))
@click.option(
    "--architecture",
    help="Model size to use",
    default=WhisperTranscriber.ARCHITECTURES[0],
    type=click.Choice(WhisperTranscriber.ARCHITECTURES),
)
@click.option(
    "--language",
    help="Language to use for transcription.",
    default=Language.unknown.name,
    type=click.Choice([x.name for x in Language]),
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for transcription.",
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
    "--cuda/--no_cuda",
    "cuda",
    help="Flag for using CUDA for Whisper's model",
    default=False,
)
@click.option(
    "--evaluate",
    "evaluation_mode",
    is_flag=True,
    help="Evaluate the transcription against golden texts.",
    default=False,
)
@click.option(
    "--vad",
    is_flag=True,
    help="Use VAD to split utterances.",
    default=False,
)
@click.option(
    "--incremental",
    is_flag=True,
    help="Save outputs immediately and use previous progress.",
    default=False,
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def transcribe_whisper_cli(context, **kwargs) -> None:
    """
    Transcribe utterances using a Whisper ASR model via faster-whisper.
    """
    if kwargs.get("profile", None) is not None:
        config.profile = kwargs.pop("profile")
    config.update_configuration(kwargs)

    config_path = kwargs.get("config_path", None)
    incremental = kwargs.get("incremental", False)
    input_path: Path = kwargs["input_path"].absolute()
    output_path: Path = kwargs["output_path"]
    corpus_root = input_path
    if not corpus_root.is_dir():
        corpus_root = corpus_root.parent

    transcriber = WhisperTranscriber(
        corpus_directory=corpus_root,
        export_directory=output_path if incremental else None,
        **WhisperTranscriber.parse_parameters(config_path, context.params, context.args),
    )
    try:
        if not input_path.is_dir():
            segment = Segment(input_path)
            transcriber.setup_model(online=True)

            text = transcribe_utterance_online_whisper(
                transcriber.model,
                segment,
            )
            if str(output_path) == "-":
                print(text)  # noqa
                sys.exit(0)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with mfa_open(output_path, "w") as f:
                f.write(text)
            del transcriber.model
        elif input_path.is_dir():
            transcriber.setup()
            transcriber.transcribe()
            if not incremental:
                transcriber.export_files(output_path)
    except Exception:
        transcriber.dirty = True
        raise
    finally:
        transcriber.cleanup()
