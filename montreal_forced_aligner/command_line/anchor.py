"""Command line functions for launching anchor annotation"""
from __future__ import annotations

import sys
import warnings

__all__ = ["run_anchor"]


def run_anchor() -> None:  # pragma: no cover
    """
    Wrapper function for launching Anchor Annotator
    """
    try:
        from anchor import Application, MainWindow
    except ImportError:
        print(
            "Anchor annotator utility is not installed, please install it via pip install anchor-annotator."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app = Application(sys.argv)
        main = MainWindow()

        app.setActiveWindow(main)
        main.show()
        sys.exit(app.exec_())
