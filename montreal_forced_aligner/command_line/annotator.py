import sys
from montreal_forced_aligner.gui import MainWindow, Application, QtWidgets
import warnings


def run_annotator(args):  # pragma: no cover
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app = Application(sys.argv)
        main = MainWindow()

        app.setActiveWindow(main)
        main.show()
        sys.exit(app.exec_())


if __name__ == '__main__':  # pragma: no cover
    from montreal_forced_aligner.command_line.mfa import fix_path, unfix_path
    fix_path()
    run_annotator(args=None)
    unfix_path()
