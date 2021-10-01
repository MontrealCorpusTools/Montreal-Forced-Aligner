import sys
import warnings


def run_anchor(args):  # pragma: no cover
    try:
        from anchor import MainWindow, Application, QtWidgets
    except ImportError:
        print('Anchor annotator utility is not installed, please install it via pip install anchor-annotator.')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app = Application(sys.argv)
        main = MainWindow()

        app.setActiveWindow(main)
        main.show()
        sys.exit(app.exec_())