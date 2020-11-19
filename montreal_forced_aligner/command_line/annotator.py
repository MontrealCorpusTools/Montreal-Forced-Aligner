import sys
from montreal_forced_aligner.gui import MainWindow, QtWidgets


def run_annotator(args):
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()

    app.setActiveWindow(main)
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    from montreal_forced_aligner.command_line.mfa import fix_path, unfix_path
    fix_path()
    run_annotator(args=None)
    unfix_path()
