"""Model classes for Transcription"""
from __future__ import annotations

import re
import typing
import warnings

import numpy as np

from montreal_forced_aligner.data import Language

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import faster_whisper
        from whisperx import asr
        from whisperx.asr import FasterWhisperPipeline

    FOUND_WHISPERX = True

except (ImportError, OSError):
    FasterWhisperPipeline = object
    FOUND_WHISPERX = False

if typing.TYPE_CHECKING:
    import torch


class MfaFasterWhisperPipeline(FasterWhisperPipeline):
    def __init__(
        self,
        model,
        vad,
        vad_params: dict,
        options: typing.NamedTuple,
        tokenizer=None,
        device: typing.Union[int, str, torch.device] = -1,
        framework: str = "pt",
        language: typing.Optional[str] = None,
        suppress_numerals: bool = False,
        **kwargs,
    ):
        self.preset_language = None
        self.tokenizer = None
        super().__init__(
            model,
            vad,
            vad_params,
            options,
            tokenizer,
            device,
            framework,
            language,
            suppress_numerals,
            **kwargs,
        )
        self.base_suppress_tokens = self.options.suppress_tokens
        if self.preset_language is not None:
            self.load_tokenizer(language=self.preset_language)

    def set_language(self, language: typing.Union[str, Language]):
        if isinstance(language, str):
            language = Language[language]
        language = language.value
        if self.preset_language != language:
            self.preset_language = language
            self.tokenizer = None
            if self.preset_language is not None:
                self.load_tokenizer(language=self.preset_language)

    def get_suppressed_tokens(
        self,
    ) -> typing.List[int]:
        suppressed = []
        import unicodedata

        alpha_pattern = re.compile(r"\w", flags=re.UNICODE)
        roman_numeral_pattern = re.compile(r"^(x+(vi+|i+|i?v|x+))$", flags=re.IGNORECASE)
        case_roman_numeral_pattern = re.compile(r"(^[IXV]{2,}$|^[xv]+i{2,}$|^x{2,}iv$|\d)")
        abbreviations_pattern = re.compile(
            r"^(sr|sra|mr|dr|mrs|vds|vd|etc)\.?$", flags=re.IGNORECASE
        )

        def _should_suppress(t):
            if t.startswith("<|"):
                return False
            if any(unicodedata.category(c) in {"Mn", "Mc"} for c in t):
                return False
            if (
                roman_numeral_pattern.search(t)
                or case_roman_numeral_pattern.search(t)
                or abbreviations_pattern.match(t)
                or re.match(r"^[XV]$", t)
                or not alpha_pattern.search(t)
            ):
                return True
            return False

        for token_id in range(self.tokenizer.eot):
            token = self.tokenizer.decode([token_id]).strip()
            if not token:
                continue
            if _should_suppress(token):
                suppressed.append(token_id)
        return suppressed

    def load_tokenizer(self, language):
        self.tokenizer = faster_whisper.tokenizer.Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language=language,
        )
        if self.suppress_numerals:
            numeral_symbol_tokens = self.get_suppressed_tokens()
            new_suppressed_tokens = numeral_symbol_tokens + self.base_suppress_tokens
            new_suppressed_tokens = sorted(set(new_suppressed_tokens))
            self.options = self.options._replace(suppress_tokens=new_suppressed_tokens)

    def transcribe(
        self,
        audio_batch: typing.List[typing.Dict[str, typing.Union[np.ndarray, float]]],
        utterance_ids,
        batch_size=None,
        num_workers=0,
    ):
        if self.preset_language is None:
            max_len = 0
            audio = None
            for a in audio_batch:
                if a["inputs"].shape[-1] > max_len:
                    max_len = a["inputs"].shape[-1]
                    audio = a["inputs"]
            language = self.detect_language(audio)
            self.load_tokenizer(language=language)

        utterances = {}
        batch_size = batch_size or self._batch_size
        for idx, out in enumerate(
            self.__call__(audio_batch, batch_size=batch_size, num_workers=num_workers)
        ):
            text = out["text"]
            utterance_id = utterance_ids[idx]
            if utterance_id not in utterances:
                utterances[utterance_id] = []
            if batch_size in [0, 1, None]:
                text = text[0]
            utterances[utterance_id].append(
                {
                    "text": text,
                    "start": round(audio_batch[idx]["start"], 3),
                    "end": round(audio_batch[idx]["end"], 3),
                }
            )
        if self.preset_language is None:
            self.tokenizer = None
        return utterances


def load_model(
    whisper_arch,
    device,
    device_index=0,
    compute_type="float16",
    asr_options=None,
    language: typing.Optional[str] = None,
    vad_model=None,
    vad_options=None,
    model: typing.Optional[asr.WhisperModel] = None,
    download_root=None,
    threads=4,
):
    """Load a Whisper model for inference.
    Args:
        whisper_arch: str - The name of the Whisper model to load.
        device: str - The device to load the model on.
        compute_type: str - The compute type to use for the model.
        options: dict - A dictionary of options to use for the model.
        language: str - The language of the model. (use English for now)
        vad_model_fp: str - File path to the VAD model to use
        model: Optional[WhisperModel] - The WhisperModel instance to use.
        download_root: Optional[str] - The root directory to download the model to.
        threads: int - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    """

    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or asr.WhisperModel(
        whisper_arch,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        download_root=download_root,
        cpu_threads=threads,
    )

    default_asr_options = {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "suppress_numerals": True,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = faster_whisper.transcribe.TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "apply_energy_VAD": False,
        "double_check": False,
        "activation_th": 0.5,
        "deactivation_th": 0.25,
        "en_activation_th": 0.5,
        "en_deactivation_th": 0.4,
        "speech_th": 0.5,
        "close_th": 0.333,
        "len_th": 0.333,
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    return MfaFasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
    )
