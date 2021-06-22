import os
import time
from PyQt5 import QtCore

from ..corpus.align_corpus import AlignableCorpus

class FunctionWorker(QtCore.QThread):
    updateProgress = QtCore.pyqtSignal(object)
    updateMaximum = QtCore.pyqtSignal(object)
    updateProgressText = QtCore.pyqtSignal(str)
    errorEncountered = QtCore.pyqtSignal(object)
    finishedCancelling = QtCore.pyqtSignal()
    actionCompleted= QtCore.pyqtSignal(object)

    dataReady = QtCore.pyqtSignal(object)

    def __init__(self):
        super(FunctionWorker, self).__init__()
        self.stopped = False
        self.finished = True

    def setParams(self, kwargs):
        self.kwargs = kwargs
        self.kwargs['call_back'] = self.emitProgress
        self.kwargs['stop_check'] = self.stopCheck
        self.stopped = False
        self.total = None

    def stop(self):
        self.stopped = True

    def stopCheck(self):
        return self.stopped

    def emitProgress(self, *args):
        if isinstance(args[0],str):
            self.updateProgressText.emit(args[0])
        elif isinstance(args[0],dict):
            self.updateProgressText.emit(args[0]['status'])
        else:
            progress = args[0]
            if len(args) > 1:
                self.updateMaximum.emit(args[1])
            self.updateProgress.emit(progress)


class ImportCorpusWorker(FunctionWorker):
    def __init__(self, logger):
        super(FunctionWorker, self).__init__()
        self.directory = None
        self.corpus_name = None
        self.corpus_temp_dir = None
        self.logger = logger
        self.stopped = False
        self.finished = True

    def setParams(self, directory, temp_directory):
        self.corpus_name = os.path.basename(directory)
        self.directory = directory
        self.corpus_temp_dir = os.path.join(temp_directory, self.corpus_name)

    def run(self):
        time.sleep(0.1)
        if not self.directory:
            return
        print('hello?')
        corpus = AlignableCorpus(self.directory, self.corpus_temp_dir, logger=self.logger)
        #if not corpus.loaded_from_temp:
        #    corpus.initialize_corpus(None)
        self.dataReady.emit(corpus)