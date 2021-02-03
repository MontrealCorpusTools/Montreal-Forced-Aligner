from ..g2p.generator import PyniniDictionaryGenerator as Generator, G2P_DISABLED

from PyQt5 import QtGui, QtCore, QtWidgets, QtMultimedia

import pyqtgraph as pg

import librosa
import numpy as np


class DetailedMessageBox(QtWidgets.QMessageBox):  # pragma: no cover
    # Adapted from http://stackoverflow.com/questions/2655354/how-to-allow-resizing-of-qmessagebox-in-pyqt4
    def __init__(self, *args, **kwargs):
        super(DetailedMessageBox, self).__init__(*args, **kwargs)
        self.setWindowTitle('Error encountered')
        self.setIcon(QtWidgets.QMessageBox.Critical)
        self.setStandardButtons(QtWidgets.QMessageBox.Close)
        self.setText("Something went wrong!")
        self.setInformativeText("Please copy the text below and send to Michael.")

        self.setMinimumWidth(200)

    def resizeEvent(self, event):
        result = super(DetailedMessageBox, self).resizeEvent(event)
        details_box = self.findChild(QtWidgets.QTextEdit)
        if details_box is not None:
            details_box.setFixedHeight(details_box.sizeHint().height())
        return result


class MediaPlayer(QtMultimedia.QMediaPlayer):  # pragma: no cover
    def __init__(self):
        super(MediaPlayer, self).__init__()
        self.max_time = None
        self.min_time = None
        self.setNotifyInterval(1)
        self.positionChanged.connect(self.checkStop)

    def setMaxTime(self, max_time):
        self.max_time = max_time * 1000

    def setMinTime(self, min_time):
        self.min_time = min_time * 1000

    def checkStop(self, position):
        if self.state() == QtMultimedia.QMediaPlayer.PlayingState:
            if self.min_time is not None:
                if position < self.min_time:
                    self.setPosition(self.min_time)
            if self.max_time is not None:
                if position > self.max_time:
                    self.stop()


class UtteranceListWidget(QtWidgets.QWidget):  # pragma: no cover
    utteranceChanged = QtCore.pyqtSignal(object)

    def __init__(self, parent):
        super(UtteranceListWidget, self).__init__(parent=parent)
        self.table_widget = QtWidgets.QTableWidget()
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setColumnCount(2)
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table_widget.setHorizontalHeaderLabels(['Utterance', 'OOV found'])
        self.table_widget.setSortingEnabled(True)
        self.table_widget.currentItemChanged.connect(self.update_utterance)
        self.table_widget.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table_widget)
        self.setLayout(layout)
        self.corpus = None
        self.dictionary = None
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.table_widget.setFocusPolicy(QtCore.Qt.NoFocus)

    def update_utterance(self, cell):
        utterance = self.table_widget.item(cell.row(), 0).text()
        self.utteranceChanged.emit(utterance)

    def update_corpus(self, corpus):
        self.corpus = corpus
        self.refresh_list()

    def update_dictionary(self, dictionary):
        self.dictionary = dictionary
        self.refresh_list()

    def refresh_list(self):
        self.table_widget.setRowCount(len(self.corpus.text_mapping))
        if self.corpus is not None:
            for i, (u, t) in enumerate(self.corpus.text_mapping.items()):
                self.table_widget.setItem(i, 0, QtWidgets.QTableWidgetItem(u))
                oov_found = False
                if self.dictionary is not None:
                    words = t.split(' ')
                    for w in words:
                        if not self.dictionary.check_word(w):
                            oov_found = True
                if oov_found:
                    t = QtWidgets.QTableWidgetItem('yes')
                    t.setBackground(QtCore.Qt.red)
                else:
                    t = QtWidgets.QTableWidgetItem('no')
                    t.setBackground(QtCore.Qt.green)
                self.table_widget.setItem(i, 1, t)


class TranscriptionWidget(QtWidgets.QTextEdit):  # pragma: no cover
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Tab:
            event.ignore()
            return
        super(TranscriptionWidget, self).keyPressEvent(event)


class UtteranceDetailWidget(QtWidgets.QWidget):  # pragma: no cover
    lookUpWord = QtCore.pyqtSignal(object)
    createWord = QtCore.pyqtSignal(object)
    saveUtterance = QtCore.pyqtSignal(object, object)

    def __init__(self, parent):
        super(UtteranceDetailWidget, self).__init__(parent=parent)
        self.corpus = None
        self.dictionary = None
        self.utterance = None
        self.audio = None
        self.sr = None
        self.current_time = 0
        self.min_time = 0
        self.max_time = None
        self.m_audioOutput = MediaPlayer()
        #self.m_audioOutput.error.connect(self.showError)
        self.m_audioOutput.positionChanged.connect(self.notified)
        self.m_audioOutput.stateChanged.connect(self.handleAudioState)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.ax = pg.PlotWidget()
        self.ax.setFocusPolicy(QtCore.Qt.NoFocus)
        self.line = pg.InfiniteLine(
            pos=-20,
            pen=pg.mkPen('r', width=3),
            movable=False  # We have our own code to handle dragless moving.
        )
        self.ax.getPlotItem().hideAxis('left')
        self.ax.getPlotItem().getAxis('bottom').setScale(1/1000)
        self.ax.getPlotItem().setMouseEnabled(False, False)
        self.ax.addItem(self.line)
        self.ax.getPlotItem().setMenuEnabled(False)
        self.ax.scene().sigMouseClicked.connect(self.update_current_time)
        layout = QtWidgets.QVBoxLayout()

        button_layout = QtWidgets.QVBoxLayout()
        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.clicked.connect(self.play_audio)
        self.reset_button = QtWidgets.QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_text)
        self.save_button = QtWidgets.QPushButton('Save')
        self.save_button.clicked.connect(self.save_text)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.save_button)
        self.text_widget = TranscriptionWidget()
        self.text_widget.setFontPointSize(20)
        self.text_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.text_widget.customContextMenuRequested.connect(self.generate_context_menu)
        text_layout = QtWidgets.QHBoxLayout()
        text_layout.addWidget(self.text_widget)
        text_layout.addLayout(button_layout)
        layout.addWidget(self.ax)
        layout.addLayout(text_layout)
        self.setLayout(layout)

    def update_plot_scale(self):
        self.p2.setGeometry(self.p1.vb.sceneBoundingRect())

    def update_corpus(self, corpus):
        self.corpus = corpus
        if self.utterance:
            self.reset_text()

    def update_dictionary(self, dictionary):
        self.dictionary = dictionary

    def generate_context_menu(self, location):

        menu = self.text_widget.createStandardContextMenu()
        cursor = self.text_widget.cursorForPosition(location)
        cursor.select(QtGui.QTextCursor.WordUnderCursor)
        word = cursor.selectedText()
        # add extra items to the menu
        lookUpAction = QtWidgets.QAction("Look up '{}' in dictionary".format(word), self)
        createAction = QtWidgets.QAction("Add pronunciation for '{}'".format(word), self)
        lookUpAction.triggered.connect(lambda : self.lookUpWord.emit(word))
        createAction.triggered.connect(lambda : self.createWord.emit(word))
        menu.addAction(lookUpAction)
        menu.addAction(createAction)
        # show the menu
        menu.exec_(self.text_widget.mapToGlobal(location))

    def update_current_time(self, ev):
        point = self.ax.getPlotItem().vb.mapSceneToView(ev.scenePos())
        x = point.x()
        self.current_time = x / self.sr
        if self.current_time < self.min_time:
            self.current_time = self.min_time
            x = self.current_time * self.sr
        if self.current_time > self.max_time:
            self.current_time = self.max_time
            x = self.current_time * self.sr

        self.line.setPos(x)
        self.m_audioOutput.setMinTime(self.current_time)
        self.m_audioOutput.setPosition(self.m_audioOutput.min_time)

    def update_utterance(self, utterance):
        if utterance is None:
            return
        self.utterance = utterance
        self.reset_text()
        if self.utterance in self.corpus.segments:
            segment = self.corpus.segments[self.utterance]
            file_name, begin, end = segment.split(' ')
            begin = float(begin)
            end = float(end)
            wav_path = self.corpus.utt_wav_mapping[file_name]
            duration = end - begin
            y, sr = librosa.load(wav_path, offset=begin, duration=duration, sr=1000)
            begin_samp = int(begin * sr)
            x = np.arange(start=begin_samp, stop=begin_samp + y.shape[0])
            self.min_time = begin
            self.max_time = end
        else:
            wav_path = self.corpus.utt_wav_mapping[self.utterance]

            y, sr = librosa.load(wav_path, sr=1000)
            x = np.arange(y.shape[0])
            self.min_time = 0
            self.max_time = y.shape[0] / sr
        p = QtCore.QUrl.fromLocalFile(wav_path)
        self.m_audioOutput.setMedia(QtMultimedia.QMediaContent(p))
        self.m_audioOutput.setMinTime(self.min_time)
        self.m_audioOutput.setMaxTime(self.max_time)
        self.sr = sr
        self.ax.getPlotItem().clear()
        self.ax.plot(x, y, pen=pg.mkPen('w', width=3))

    def reset_text(self):
        if self.utterance not in self.corpus.text_mapping:
            self.utterance = None
            self.audio = None
            self.sr = None
            self.text_widget.setText('')
            return
        text = self.corpus.text_mapping[self.utterance]
        words = text.split(' ')
        mod_words = []
        if self.dictionary is not None:
            for w in words:
                if not self.dictionary.check_word(w):
                    w = '<span style=\" font-size: 20pt; font-weight:600; color:#ff0000;\" >{} </span>'.format(w)
                else:
                    w = '<span style=\" font-size: 20pt\" >{} </span>'.format(w)
                mod_words.append(w)

        else:
            mod_words = words
        self.text_widget.setText(''.join(mod_words))

    def save_text(self):
        text = self.text_widget.toPlainText()
        self.saveUtterance.emit(self.utterance, text)

    def showError(self, e):
        reply = DetailedMessageBox()
        reply.setDetailedText(str(e))
        ret = reply.exec_()

    def play_audio(self):
        if self.m_audioOutput.state() == QtMultimedia.QMediaPlayer.StoppedState or \
                self.m_audioOutput.state() == QtMultimedia.QMediaPlayer.PausedState:
            if self.m_audioOutput.state() == QtMultimedia.QMediaPlayer.StoppedState:
                self.m_audioOutput.play()
        elif self.m_audioOutput.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.m_audioOutput.pause()
        elif self.m_audioOutput.state() == QtMultimedia.QMediaPlayer.PausedState:
            self.m_audioOutput.play()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Tab:
            print('PLAYING')
            if self.utterance is None:
                return
            self.play_audio()

    def updatePlayTime(self, time):
        if self.sr:
            pos = int(time * self.sr)
            self.line.setPos(pos)

    def notified(self, position):
        time = position / 1000
        self.updatePlayTime(time)

    def handleAudioState(self, state):
        if state == QtMultimedia.QAudio.StoppedState:
            if self.min_selected_time is None:
                min_time = self.view_begin
            else:
                min_time = self.min_selected_time
            self.updatePlayTime(min_time)
            self.m_audioOutput.setPosition(0)


class InformationWidget(QtWidgets.QWidget):  # pragma: no cover
    resetDictionary = QtCore.pyqtSignal()
    saveDictionary = QtCore.pyqtSignal(object)

    def __init__(self, parent):
        super(InformationWidget, self).__init__(parent=parent)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.dictionary = None

        layout = QtWidgets.QVBoxLayout()
        self.dictionary_widget = QtWidgets.QTableWidget()
        self.dictionary_widget.verticalHeader().setVisible(False)
        self.dictionary_widget.setColumnCount(2)
        self.dictionary_widget.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.dictionary_widget.setHorizontalHeaderLabels(['Word', 'Pronunciation'])
        layout.addWidget(self.dictionary_widget)

        button_layout = QtWidgets.QHBoxLayout()
        self.reset_button = QtWidgets.QPushButton('Reset dictionary')
        self.save_button = QtWidgets.QPushButton('Save dictionary')

        self.reset_button.clicked.connect(self.resetDictionary.emit)
        self.save_button.clicked.connect(self.create_dictionary_for_save)

        button_layout.addWidget(self.reset_button)

        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def update_dictionary(self, dictionary):
        self.dictionary = dictionary
        self.dictionary_widget.setRowCount(len(self.dictionary))
        cur_index = 0
        for word, prons in sorted(self.dictionary.words.items()):
            for p in prons:
                pronunciation = ' '.join(p['pronunciation'])
                self.dictionary_widget.setItem(cur_index, 0, QtWidgets.QTableWidgetItem(word))
                self.dictionary_widget.setItem(cur_index, 1, QtWidgets.QTableWidgetItem(pronunciation))
                cur_index += 1

    def update_g2p(self, g2p_model):
        self.g2p_model = g2p_model

    def create_dictionary_for_save(self):
        from collections import defaultdict
        words = defaultdict(list)
        phones = set()
        for i in range(self.dictionary_widget.rowCount()):
            word = self.dictionary_widget.item(i, 0).text()
            pronunciation = self.dictionary_widget.item(i, 1).text()
            pronunciation = tuple(pronunciation.split(' '))
            phones.update(pronunciation)
            words[word].append((pronunciation, None))
        new_phones = phones - self.dictionary.phones
        if new_phones:
            print("ERROR can't save because of new phones: {}".format(new_phones))
            return
        self.saveDictionary.emit(words)

    def create_pronunciation(self, word):
        if self.dictionary is None:
            return
        if not word:
            return
        pronunciation = None
        if self.g2p_model is not None and not G2P_DISABLED:
            gen = Generator(self.g2p_model, [word])
            results = gen.generate()
            pronunciation = results[word]
            self.dictionary.words[word].append((tuple(pronunciation.split(' ')), 1))
        for i in range(self.dictionary_widget.rowCount()):
            row_text = self.dictionary_widget.item(i, 0).text()
            if not row_text:
                continue
            if row_text < word:
                continue
            self.dictionary_widget.insertRow(i)
            self.dictionary_widget.setItem(i, 0, QtWidgets.QTableWidgetItem(word))
            if pronunciation is not None:
                self.dictionary_widget.setItem(i, 1, QtWidgets.QTableWidgetItem(pronunciation))
            self.dictionary_widget.scrollToItem(self.dictionary_widget.item(i, 0))
            self.dictionary_widget.selectRow(i)
            break

    def look_up_word(self, word):
        if self.dictionary is None:
            return
        if not word:
            return
        for i in range(self.dictionary_widget.rowCount()):
            if self.dictionary_widget.item(i, 0).text() == word:
                self.dictionary_widget.scrollToItem(self.dictionary_widget.item(i, 0))
                self.dictionary_widget.selectRow(i)
                break
