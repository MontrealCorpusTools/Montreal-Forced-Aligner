"""Class definitions for base configuration"""
from __future__ import annotations

from typing import TYPE_CHECKING, Collection

import yaml

if TYPE_CHECKING:
    from argparse import Namespace


PARSING_KEYS = [
    "punctuation",
    "clitic_markers",
    "compound_markers",
    "multilingual_ipa",
    "strip_diacritics",
    "digraphs",
]

__all__ = ["BaseConfig"]


class BaseConfig:
    """
    Base configuration class
    """

    def update(self, data: dict) -> None:
        """Update configuration parameters"""
        for k, v in data.items():
            if not hasattr(self, k):
                continue
            setattr(self, k, v)

    def update_from_args(self, args: Namespace) -> None:
        """Update from command line arguments"""
        if args is not None:
            try:
                self.use_mp = not args.disable_mp
            except AttributeError:
                pass
            try:
                self.debug = args.debug
            except AttributeError:
                pass
            try:
                self.overwrite = args.overwrite
            except AttributeError:
                pass
            try:
                self.cleanup_textgrids = not args.disable_textgrid_cleanup
            except AttributeError:
                pass

    def params(self) -> dict:
        """Configuration parameters"""
        return {}

    def update_from_unknown_args(self, args: Collection[str]) -> None:
        """Update from unknown command line arguments"""
        for i, a in enumerate(args):
            if not a.startswith("--"):
                continue
            name = a.replace("--", "")
            try:
                original_value = getattr(self, name)
            except AttributeError:
                continue
            if not isinstance(original_value, (bool, int, float, str)):
                continue
            try:
                if isinstance(original_value, bool):
                    if args[i + 1].lower() == "true":
                        val = True
                    elif args[i + 1].lower() == "false":
                        val = False
                    elif not original_value:
                        val = True
                    else:
                        continue
                else:
                    val = type(original_value)(args[i + 1])
            except (ValueError):
                continue
            except (IndexError):
                if isinstance(original_value, bool):
                    if not original_value:
                        val = True
                    else:
                        continue
                else:
                    continue
            setattr(self, name, val)

    def save(self, path: str) -> None:
        """
        Dump configuration to path

        Parameters
        ----------
        path: str
            Path to export to
        """
        with open(path, "w", encoding="utf8") as f:
            yaml.dump(self.params(), f)
