from ..g2p.generator import PyniniDictionaryGenerator as Generator, G2P_DISABLED

from ..corpus.base import get_wav_info

from PyQt5 import QtGui, QtCore, QtWidgets, QtMultimedia

import pyqtgraph as pg
import librosa
import numpy as np

pg.setConfigOptions(antialias=True)
pg.setConfigOptions(useOpenGL=True)


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
    timeChanged = QtCore.pyqtSignal(object)
    def __init__(self):
        super(MediaPlayer, self).__init__()
        self.max_time = None
        self.min_time = None
        self.start_time = 0
        self.sr = None
        self.setNotifyInterval(1)
        self.positionChanged.connect(self.checkStop)
        self.buf = QtCore.QBuffer()

    def currentTime(self):
        pos = self.position()
        return pos / 1000

    def setMaxTime(self, max_time):
        self.max_time = max_time * 1000

    def setMinTime(self, min_time):  # Positions for MediaPlayer are in milliseconds, no SR required
        self.min_time = min_time * 1000
        if self.start_time < self.min_time:
            self.start_time = self.min_time

    def setStartTime(self, start_time):  # Positions for MediaPlayer are in milliseconds, no SR required
        self.start_time = start_time * 1000

    def setCurrentTime(self, time):
        if self.state() == QtMultimedia.QMediaPlayer.PlayingState:
            return
        pos = time * 1000
        self.setPosition(pos)

    def checkStop(self, position):
        if self.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.timeChanged.emit(self.currentTime())
            if self.max_time is not None:
                if position > self.max_time + 3:
                    self.stop()


class UtteranceListWidget(QtWidgets.QWidget):  # pragma: no cover
    utteranceChanged = QtCore.pyqtSignal(object, object)
    fileChanged = QtCore.pyqtSignal(object)
    utteranceMerged = QtCore.pyqtSignal()
    utteranceDeleted = QtCore.pyqtSignal()
    updateView = QtCore.pyqtSignal(object, object)
    saveFile = QtCore.pyqtSignal(object)

    def __init__(self, parent):
        super(UtteranceListWidget, self).__init__(parent=parent)
        self.setMaximumWidth(400)
        self.file_dropdown = QtWidgets.QComboBox()
        self.file_dropdown.currentTextChanged.connect(self.file_changed)
        self.table_widget = QtWidgets.QTableWidget()
        self.table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(['Utterance', 'Speaker', 'OOV found'])
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table_widget.setSortingEnabled(True)
        self.table_widget.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.deleted_utts = []
        layout = QtWidgets.QVBoxLayout()
        button_layout = QtWidgets.QHBoxLayout()
        self.saveButton = QtWidgets.QPushButton('Save current file')
        self.saveButton.setDisabled(True)
        self.saveButton.clicked.connect(self.save_file)
        self.restoreButton = QtWidgets.QPushButton('Restore deleted')
        self.restoreButton.clicked.connect(self.restore_deleted_utts)
        button_layout.addWidget(self.saveButton)
        button_layout.addWidget(self.restoreButton)
        layout.addWidget(self.file_dropdown)
        layout.addWidget(self.table_widget)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.corpus = None
        self.dictionary = None
        self.current_file = None
        # self.setFocusPolicy(QtCore.Qt.NoFocus)
        # self.table_widget.setFocusPolicy(QtCore.Qt.NoFocus)

    def setFileSaveable(self, value=True):
        self.saveButton.setEnabled(value)

    def file_changed(self, file_name):
        self.current_file = file_name
        self.fileChanged.emit(file_name)
        self.refresh_list()
        self.setFileSaveable(False)

    def save_file(self):
        self.saveFile.emit(self.current_file)

    def update_utterance(self, cell):
        if not cell:
            return
        utterance = self.table_widget.item(cell.row(), 0).text()
        if self.corpus.segments:
            self.current_file = self.corpus.segments[utterance]['file_name']
        else:
            self.current_file = utterance
        self.utteranceChanged.emit(utterance, False)

    def update_and_zoom_utterance(self, cell):
        if not cell:
            return
        utterance = self.table_widget.item(cell.row(), 0).text()
        if self.corpus.segments:
            self.current_file = self.corpus.segments[utterance]['file_name']
        else:
            self.current_file = utterance
        self.utteranceChanged.emit(utterance, True)

    def update_utterance_text(self, utterance):
        t = self.corpus.text_mapping[utterance]
        for r in range(self.table_widget.rowCount()):
            if self.table_widget.item(r, 0).text() == utterance:
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
                if self.table_widget.columnCount() == 3:
                    ind = 2
                else:
                    ind = 4
                self.table_widget.setItem(r, ind, t)
                break

    def get_current_selection(self):
        utts = []
        for i in self.table_widget.selectedItems():
            if i.column() == 0:
                utts.append(i.text())
        return utts

    def restore_deleted_utts(self):
        for utt_data in self.deleted_utts:
            self.corpus.add_utterance(**utt_data)

        self.refresh_list()
        self.table_widget.clearSelection()
        self.utteranceDeleted.emit()
        self.deleted_utts = []

    def delete_utterances(self):
        utts = self.get_current_selection()
        if len(utts) < 1:
            return
        for old_utt in utts:
            if old_utt in self.corpus.segments:
                seg = self.corpus.segments[old_utt]
                speaker = self.corpus.utt_speak_mapping[old_utt]
                file = self.corpus.utt_file_mapping[old_utt]
                text = self.corpus.text_mapping[old_utt]
                utt_data = {'utterance': old_utt, 'seg': seg, 'speaker': speaker, 'file': file,
                            'text': text}
                self.deleted_utts.append(utt_data)
                self.corpus.delete_utterance(old_utt)
        self.refresh_list()
        self.table_widget.clearSelection()
        self.utteranceDeleted.emit()
        self.setFileSaveable(True)

    def split_utterances(self):
        utts = self.get_current_selection()
        if len(utts) != 1:
            return
        old_utt = utts[0]
        if old_utt in self.corpus.segments:
            seg = self.corpus.segments[old_utt]
            speaker = self.corpus.utt_speak_mapping[old_utt]
            file = self.corpus.utt_file_mapping[old_utt]
            filename = seg['file_name']
            beg = seg['begin']
            end = seg['end']
            channel = seg['channel']
            utt_text = self.corpus.text_mapping[old_utt]
            duration = end - beg
            split_time = beg + (duration / 2)
        else:
            return

        first_utt = '{}_{}_{}_{}'.format(speaker, filename, beg, split_time).replace('.', '_')
        first_seg = {'file_name': filename, 'begin': beg, 'end': split_time, 'channel': channel}
        self.corpus.add_utterance(first_utt, speaker, file, utt_text, seg=first_seg)

        second_utt_text = ''
        if utt_text == 'speech':  # Check for segmentation
            second_utt_text = 'speech'

        second_utt = '{}_{}_{}_{}'.format(speaker, filename, split_time, end).replace('.', '_')
        second_seg = {'file_name': filename, 'begin': split_time, 'end': end, 'channel': channel}
        self.corpus.add_utterance(second_utt, speaker, file, second_utt_text, seg=second_seg)

        self.corpus.delete_utterance(old_utt)

        self.refresh_list()
        self.utteranceMerged.emit()
        self.table_widget.clearSelection()
        self.select_utterance(first_utt)
        self.utteranceChanged.emit(first_utt, False)
        self.updateView.emit(None, None)
        self.setFileSaveable(True)

    def merge_utterances(self):
        print('MERGING')
        utts = {}
        rows = []
        for i in self.table_widget.selectedItems():
            if i.column() == 0:
                utts[i.row()] = i.text()
                rows.append(i.row())
        if len(rows) < 2:
            return
        row = None
        for r in sorted(rows):
            if row is not None:
                if r - row != 1:
                    return
            row = r
        min_begin = 1000000000
        max_end = 0
        text = ''
        speaker = None
        file = None
        for r, old_utt in sorted(utts.items(), key=lambda x: x[0]):
            if old_utt in self.corpus.segments:
                seg = self.corpus.segments[old_utt]
                if speaker is None:
                    speaker = self.corpus.utt_speak_mapping[old_utt]
                    file = self.corpus.utt_file_mapping[old_utt]
                filename = seg['file_name']
                beg = seg['begin']
                end = seg['end']
                channel = seg['channel']
                if beg < min_begin:
                    min_begin = beg
                if end > max_end:
                    max_end = end
                utt_text = self.corpus.text_mapping[old_utt]
                if utt_text == 'speech' and text.strip() == 'speech':
                    continue
                text += utt_text + ' '
            else:
                return
        text = text[:-1]
        new_utt = '{}_{}_{}_{}'.format(speaker, filename, min_begin, max_end).replace('.', '_')
        new_seg = {'file_name': filename, 'begin': min_begin, 'end': max_end, 'channel': channel}
        self.corpus.add_utterance(new_utt, speaker, file, text, seg=new_seg)

        for r, old_utt in sorted(utts.items(), key=lambda x: x[0]):
            self.corpus.delete_utterance(old_utt)
        self.refresh_list()
        self.table_widget.clearSelection()
        self.utteranceMerged.emit()
        self.select_utterance(new_utt)
        self.utteranceChanged.emit(new_utt, False)
        self.setFileSaveable(True)

    def create_utterance(self, speaker, begin, end, channel):
        begin = round(begin, 4)
        end = round(end, 4)
        text = ''
        file = self.current_file
        new_utt = '{}_{}_{}_{}'.format(speaker, file, begin, end).replace('.', '_')
        new_seg = {'file_name': file, 'begin': begin, 'end': end, 'channel': channel}
        self.corpus.add_utterance(new_utt, speaker, file, text, seg=new_seg)
        self.refresh_list()
        self.table_widget.clearSelection()
        self.utteranceMerged.emit()
        self.select_utterance(new_utt)
        self.utteranceChanged.emit(new_utt, False)
        self.setFileSaveable(True)

    def select_utterance(self, utt, zoom=False):
        self.table_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        if utt is None:
            self.table_widget.clearSelection()
        else:
            for r in range(self.table_widget.rowCount()):
                if self.table_widget.item(r, 0).text() == utt:
                    self.table_widget.selectRow(r)
                    break
        self.table_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        cur = self.get_current_selection()
        if not cur:
            return
        if zoom:
            if self.corpus.segments:
                min_time = 100000
                max_time = 0
                for u in cur:
                    beg = self.corpus.segments[u]['begin']
                    end = self.corpus.segments[u]['end']
                    if beg < min_time:
                        min_time = beg
                    if end > max_time:
                        max_time = end
                min_time -= 1
                max_time += 1
        else:
            min_time = None
            max_time = None
        self.updateView.emit(min_time, max_time)

    def refresh_corpus(self, utt=None):
        self.refresh_list()
        self.table_widget.clearSelection()
        self.select_utterance(utt)

    def update_corpus(self, corpus):
        self.corpus = corpus
        if self.corpus:
            if not self.corpus.segments:
                self.file_dropdown.clear()
                self.file_dropdown.hide()
                self.current_file = None
                self.table_widget.setColumnCount(3)
                self.table_widget.setHorizontalHeaderLabels(['Utterance', 'Speaker', 'OOV found'])
            else:
                self.file_dropdown.show()
                self.refresh_file_dropdown()
                self.table_widget.setColumnCount(5)
                self.table_widget.setHorizontalHeaderLabels(['Utterance', 'Speaker', 'Begin', 'End', 'OOV found'])

            self.table_widget.currentItemChanged.connect(self.update_utterance)
            self.table_widget.itemDoubleClicked.connect(self.update_and_zoom_utterance)
        else:
            self.file_dropdown.clear()
            self.file_dropdown.hide()
            try:
                self.table_widget.currentItemChanged.disconnect(self.update_utterance)
            except TypeError:
                pass
        self.refresh_list()

    def update_dictionary(self, dictionary):
        self.dictionary = dictionary
        self.refresh_list()

    def refresh_file_dropdown(self):
        self.file_dropdown.clear()
        for fn in sorted(self.corpus.file_utt_mapping):

            self.file_dropdown.addItem(fn)

    def refresh_list(self):
        self.table_widget.clearContents()
        if not self.corpus:
            self.table_widget.setRowCount(0)
            return

        if self.corpus.segments:
            file = self.file_dropdown.currentText()
            if not file:
                return
            self.table_widget.setRowCount(len(self.corpus.file_utt_mapping[file]))
            for i, u in enumerate(sorted(self.corpus.file_utt_mapping[file], key=lambda x: self.corpus.segments[x]['begin'])):
                t = self.corpus.text_mapping[u]
                self.table_widget.setItem(i, 0, QtWidgets.QTableWidgetItem(u))
                self.table_widget.setItem(i, 1, QtWidgets.QTableWidgetItem(self.corpus.utt_speak_mapping[u]))
                if u in self.corpus.segments:
                    item = QtWidgets.QTableWidgetItem()
                    item.setData(QtCore.Qt.EditRole, self.corpus.segments[u]['begin'])
                    self.table_widget.setItem(i, 2, item)
                    item = QtWidgets.QTableWidgetItem()
                    item.setData(QtCore.Qt.EditRole, self.corpus.segments[u]['end'])
                    self.table_widget.setItem(i, 3, item)
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
                self.table_widget.setItem(i, 4, t)
        else:
            self.table_widget.setRowCount(len(self.corpus.text_mapping))
            if self.corpus is not None:
                for i, (u, t) in enumerate(sorted(self.corpus.text_mapping.items())):
                    self.table_widget.setItem(i, 0, QtWidgets.QTableWidgetItem(u))
                    self.table_widget.setItem(i, 1, QtWidgets.QTableWidgetItem(self.corpus.utt_speak_mapping[u]))
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
                    self.table_widget.setItem(i, 2, t)


class TranscriptionWidget(QtWidgets.QTextEdit):  # pragma: no cover
    playAudio = QtCore.pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Tab:
            event.accept()
            self.playAudio.emit()
            return
        super(TranscriptionWidget, self).keyPressEvent(event)


class SelectedUtterance(pg.LinearRegionItem):
    dragFinished = QtCore.pyqtSignal(object)
    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return

        if ev.isFinish():
            pos = ev.pos()
            self.dragFinished.emit(pos)
            return

        ev.accept()


def construct_text_region(utt, view_min, view_max, point_min, point_max, sr, speaker_ind,
                          selected_range_color='blue', selected_line_color='g',
                          break_line_color=1.0, text_color=1.0, interval_background_color=0.25,
                          plot_text_font=12, plot_text_width=400):
    y_range = point_max - point_min
    b_s = utt['begin'] * sr
    e_s = utt['end'] * sr
    if not isinstance(selected_range_color, QtGui.QColor):
        selected_range_color = QtGui.QColor(selected_range_color)
    selected_range_color.setAlpha(50)
    reg_brush = pg.mkBrush(selected_range_color)

    reg = SelectedUtterance([b_s, e_s], pen=pg.mkPen(selected_line_color, width=3), brush=reg_brush)
    reg.movable = False
    mid_b = utt['begin']
    mid_e = utt['end']
    speaker_tier_range = (y_range / 2)

    if view_min > mid_b and mid_e - view_min > 1:
        b_s = view_min * sr
    if view_max < mid_e and view_max - mid_b > 1:
        e_s = view_max * sr
    text_dur = e_s - b_s
    mid_point = b_s + (text_dur * 0.5)  # * self.sr
    t = pg.TextItem(utt['text'], anchor=(0.5, 0.5), color=text_color)
    top_point = point_min - speaker_tier_range * (speaker_ind - 1)
    y_mid_point = top_point - (speaker_tier_range / 2)
    t.setPos(mid_point, y_mid_point)
    font = t.textItem.font()
    font.setPointSize(plot_text_font)
    t.setFont(font)
    if t.textItem.boundingRect().width() > plot_text_width:
        t.setTextWidth(plot_text_width)
    pen = pg.mkPen(break_line_color, width=3)
    pen.setCapStyle(QtCore.Qt.FlatCap)
    begin_line = pg.PlotCurveItem([b_s, b_s], [top_point, top_point - speaker_tier_range],
                                  pen=pg.mkPen(pen))
    end_line = pg.PlotCurveItem([e_s, e_s], [top_point, top_point - speaker_tier_range],
                                pen=pg.mkPen(pen))
    begin_line.setClickable(False)
    end_line.setClickable(False)
    fill_brush = pg.mkBrush(interval_background_color)
    fill_between = pg.FillBetweenItem(begin_line, end_line, brush=fill_brush)
    return t, reg, fill_between


def construct_text_box(utt, view_min, view_max, point_min, point_max, sr, speaker_ind,
                       break_line_color=1.0, text_color=1.0, interval_background_color=0.25,
                       plot_text_font=12, plot_text_width=400):
    y_range = point_max - point_min
    b_s = utt['begin'] * sr
    e_s = utt['end'] * sr
    pen = pg.mkPen(break_line_color, width=3)
    pen.setCapStyle(QtCore.Qt.FlatCap)
    mid_b = utt['begin']
    mid_e = utt['end']
    speaker_tier_range = (y_range / 2)
    b_s_for_text = b_s
    e_s_for_text = e_s
    if view_min > mid_b and mid_e - view_min > 1:
        b_s_for_text = view_min * sr
    if view_max < mid_e and view_max - mid_b > 1:
        e_s_for_text = view_max * sr
    text_dur = e_s_for_text - b_s_for_text
    mid_point = b_s_for_text + (text_dur * 0.5)  # * self.sr
    t = pg.TextItem(utt['text'], anchor=(0.5, 0.5), color=text_color)
    top_point = point_min - speaker_tier_range * (speaker_ind - 1)
    y_mid_point = top_point - (speaker_tier_range / 2)
    t.setPos(mid_point, y_mid_point)
    font = t.textItem.font()
    font.setPointSize(plot_text_font)
    t.setFont(font)
    if t.textItem.boundingRect().width() > plot_text_width:
        t.setTextWidth(plot_text_width)
    begin_line = pg.PlotCurveItem([b_s, b_s], [top_point, top_point - speaker_tier_range],
                                  pen=pg.mkPen(pen))
    end_line = pg.PlotCurveItem([e_s, e_s], [top_point, top_point - speaker_tier_range],
                                pen=pg.mkPen(pen))
    begin_line.setClickable(False)
    end_line.setClickable(False)
    fill_brush = pg.mkBrush(interval_background_color)
    fill_between = pg.FillBetweenItem(begin_line, end_line, brush=fill_brush)
    return t, begin_line, end_line, fill_between


class UtteranceDetailWidget(QtWidgets.QWidget):  # pragma: no cover
    lookUpWord = QtCore.pyqtSignal(object)
    createWord = QtCore.pyqtSignal(object)
    saveUtterance = QtCore.pyqtSignal(object, object)
    selectUtterance = QtCore.pyqtSignal(object, object)
    createUtterance = QtCore.pyqtSignal(object, object, object, object)
    refreshCorpus = QtCore.pyqtSignal(object)
    updateSpeaker = QtCore.pyqtSignal(object, object)
    utteranceUpdated = QtCore.pyqtSignal(object)
    utteranceChanged = QtCore.pyqtSignal(object)

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
        self.selected_min = None
        self.selected_max = None
        self.background_color = '#000000'
        self.m_audioOutput = MediaPlayer()
        # self.m_audioOutput.error.connect(self.showError)
        self.m_audioOutput.timeChanged.connect(self.notified)
        self.m_audioOutput.stateChanged.connect(self.handleAudioState)
        # self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.ax = pg.PlotWidget()
        # self.ax.setFocusPolicy(QtCore.Qt.NoFocus)
        self.line = pg.InfiniteLine(
            pos=-20,
            pen=pg.mkPen('r', width=1),
            movable=False  # We have our own code to handle dragless moving.
        )
        self.ax.getPlotItem().hideAxis('left')
        self.ax.getPlotItem().setMouseEnabled(False, False)
        # self.ax.getPlotItem().setFocusPolicy(QtCore.Qt.NoFocus)
        self.ax.addItem(self.line)
        self.ax.getPlotItem().setMenuEnabled(False)
        self.ax.scene().sigMouseClicked.connect(self.update_current_time)
        layout = QtWidgets.QVBoxLayout()

        self.scroll_bar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.scroll_bar.valueChanged.connect(self.update_from_slider)

        button_layout = QtWidgets.QVBoxLayout()
        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.clicked.connect(self.play_audio)
        self.show_all_speaker_checkbox = QtWidgets.QCheckBox('Show all speakers')
        self.show_all_speakers = False
        self.show_all_speaker_checkbox.stateChanged.connect(self.update_show_speakers)
        # self.play_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.speaker_dropdown = QtWidgets.QComboBox()
        self.speaker_dropdown.currentIndexChanged.connect(self.update_speaker)
        self.speaker_dropdown.hide()
        # self.speaker_dropdown.setFocusPolicy(QtCore.Qt.NoFocus)
        volume_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel()
        label.setPixmap(self.style().standardPixmap(QtWidgets.QStyle.SP_MediaVolume))
        label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        volume_layout.addWidget(label)
        self.volume_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.volume_slider.setTickInterval(1)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.setSliderPosition(self.m_audioOutput.volume())
        self.volume_slider.valueChanged.connect(self.m_audioOutput.setVolume)
        volume_layout.addWidget(self.volume_slider)

        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.show_all_speaker_checkbox)
        button_layout.addLayout(volume_layout)
        button_layout.addWidget(self.speaker_dropdown)
        self.text_widget = TranscriptionWidget()
        self.text_widget.setMaximumHeight(100)
        self.text_widget.playAudio.connect(self.play_audio)
        self.text_widget.textChanged.connect(self.update_utterance_text)
        self.text_widget.setFontPointSize(20)
        self.text_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # self.text_widget.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.text_widget.customContextMenuRequested.connect(self.generate_context_menu)
        text_layout = QtWidgets.QHBoxLayout()
        text_layout.addWidget(self.text_widget)
        text_layout.addLayout(button_layout)
        layout.addWidget(self.ax)
        layout.addWidget(self.scroll_bar)
        layout.addLayout(text_layout)
        self.setLayout(layout)
        self.wav_path = None
        self.channels = 1
        self.wave_data = None
        self.long_file = None
        self.sr = None
        self.file_utts = []
        self.selected_utterance = None

    def update_show_speakers(self, state):
        self.show_all_speakers = state > 0
        self.update_plot(self.min_time, self.max_time)

    def update_config(self, config):
        self.background_color = config['background_color']
        self.play_line_color = config['play_line_color']
        self.selected_range_color = config['selected_range_color']
        self.selected_line_color = config['selected_line_color']
        self.break_line_color = config['break_line_color']
        self.text_color = config['text_color']
        self.wave_line_color = config['wave_line_color']
        self.interval_background_color = config['interval_background_color']
        self.plot_text_font = config['plot_text_font']
        self.plot_text_width = config['plot_text_width']
        self.update_plot(self.min_time, self.max_time)

    def update_from_slider(self, value):
        if self.max_time is None:
            return
        cur_window = self.max_time - self.min_time
        self.update_plot(value, value + cur_window)

    def update_speaker(self):
        self.updateSpeaker.emit(self.utterance, self.speaker_dropdown.currentText())

    def update_utterance_text(self):
        if self.utterance is None:
            return
        new_text = self.text_widget.toPlainText().strip().lower()
        if new_text != self.corpus.text_mapping[self.utterance]:
            self.utteranceChanged.emit(True)

            self.corpus.text_mapping[self.utterance] = new_text

        for u in self.file_utts:
            if u['utt'] == self.utterance:
                u['text'] = new_text
                break
        self.update_plot(self.min_time, self.max_time)
        self.utteranceUpdated.emit(self.utterance)

    def update_plot_scale(self):
        self.p2.setGeometry(self.p1.vb.sceneBoundingRect())

    def refresh_speaker_dropdown(self):
        current_speaker = self.speaker_dropdown.currentText()
        self.speaker_dropdown.clear()
        if not self.corpus:
            return
        speakers = sorted(self.corpus.speak_utt_mapping.keys())
        for i, s in enumerate(speakers):
            if not s:
                continue
            self.speaker_dropdown.addItem(s)
            if current_speaker and current_speaker == s:
                self.speaker_dropdown.setCurrentIndex(i)

    def reset(self):
        self.utterance = None
        self.file_name = None
        self.wave_data = None
        self.wav_path = None
        try:
            self.scroll_bar.valueChanged.disconnect(self.update_from_slider)
        except TypeError:
            pass
        self.ax.getPlotItem().clear()
        self.reset_text()

    def update_corpus(self, corpus):
        self.wave_data = None
        self.corpus = corpus
        if corpus is None:
            self.reset()
        self.refresh_speaker_dropdown()
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
        lookUpAction.triggered.connect(lambda: self.lookUpWord.emit(word))
        createAction.triggered.connect(lambda: self.createWord.emit(word))
        menu.addAction(lookUpAction)
        menu.addAction(createAction)
        # show the menu
        menu.exec_(self.text_widget.mapToGlobal(location))

    def update_current_time(self, ev):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        point = self.ax.getPlotItem().vb.mapSceneToView(ev.scenePos())
        x = point.x()
        y = point.y()
        y_range = self.max_point - self.min_point
        speaker_tier_range = (y_range / 2)
        move_line = False
        time = x / self.sr
        if y < self.min_point:
            speaker = None
            for k, s_id in self.speaker_mapping.items():
                top_pos = self.min_point - speaker_tier_range * (s_id - 1)
                bottom_pos = top_pos - speaker_tier_range
                if bottom_pos < y < top_pos:
                    speaker = k
                    break
            utt = None
            for u in self.file_utts:
                if u['end'] < time:
                    continue
                if u['begin'] > time:
                    break
                if u['speaker'] != speaker:
                    continue
                utt = u
            if utt is not None:
                zoom = ev.double()
                if modifiers == QtCore.Qt.ControlModifier and utt is not None:
                    self.selectUtterance.emit(utt['utt'], zoom)
                else:
                    self.selectUtterance.emit(None, zoom)
                    if utt is not None:
                        self.selectUtterance.emit(utt['utt'], zoom)
                self.m_audioOutput.setMaxTime(self.selected_max)
                self.m_audioOutput.setCurrentTime(self.selected_min)
            elif ev.double():
                beg = time - 0.5
                end = time + 0.5
                channel = 0
                if self.channels > 1:
                    ind = self.corpus.speaker_ordering[self.file_name].index(speaker)
                    if ind >= len(self.corpus.speaker_ordering[self.file_name]) / 2:
                        channel = 1
                self.createUtterance.emit(speaker, beg, end, channel)
                return
            else:
                move_line = True
        elif ev.double():
            beg = time - 0.5
            end = time + 0.5
            channel = 0
            if self.channels > 1:
                if y < 2:
                    channel = 1
                if self.file_name not in self.corpus.speaker_ordering:
                    self.corpus.speaker_ordering[self.file_name] = ['speech']
                if channel == 0:
                    ind = 0
                else:
                    ind = int(round(len(self.corpus.speaker_ordering[self.file_name]) / 2))
                speaker = self.corpus.speaker_ordering[self.file_name][ind]
                self.createUtterance.emit(speaker, beg, end, channel)
                return

        else:
            move_line = True
        if move_line:
            self.current_time = x / self.sr
            if self.current_time < self.min_time:
                self.current_time = self.min_time
                x = self.current_time * self.sr
            if self.current_time > self.max_time:
                self.current_time = self.max_time
                x = self.current_time * self.sr

            self.line.setPos(x)
            self.m_audioOutput.setStartTime(self.current_time)
            self.m_audioOutput.setCurrentTime(self.current_time)
            self.m_audioOutput.setMaxTime(self.max_time)

    def refresh_view(self):
        self.refresh_utterances()
        self.update_plot(self.min_time, self.max_time)

    def refresh_utterances(self):
        self.file_utts = []
        if not self.file_name:
            return
        self.wav_path = self.corpus.utt_wav_mapping[self.file_name]
        self.wav_info = get_wav_info(self.wav_path)
        self.scroll_bar.setMaximum(self.wav_info['duration'])
        if self.file_name in self.corpus.file_utt_mapping:
            for u in self.corpus.file_utt_mapping[self.file_name]:
                begin = self.corpus.segments[u]['begin']
                end = self.corpus.segments[u]['end']
                self.file_utts.append({'utt': u, 'begin': begin,
                                       'end': end, 'text': self.corpus.text_mapping[u],
                                       'speaker': self.corpus.utt_speak_mapping[u]})
        else:
            u = self.file_name
            self.file_utts.append({'utt': u, 'begin': 0,
                                   'end': self.wav_info['duration'], 'text': self.corpus.text_mapping[u],
                                   'speaker': self.corpus.utt_speak_mapping[u]})
        self.file_utts.sort(key=lambda x: x['begin'])

    def update_file_name(self, file_name):
        if not self.corpus:
            self.file_name = None
            self.wave_data = None
            return
        if file_name == self.file_name:
            return
        self.file_name = file_name
        if file_name in self.corpus.utt_speak_mapping:
            self.long_file = False
            self.speaker_dropdown.hide()
        else:
            self.long_file = True
            self.speaker_dropdown.show()
        if self.wav_path != self.corpus.utt_wav_mapping[file_name]:
            self.refresh_utterances()
        self.wav_path = self.corpus.utt_wav_mapping[file_name]
        try:
            self.scroll_bar.valueChanged.disconnect(self.update_from_slider)
        except TypeError:
            pass
        self.scroll_bar.valueChanged.connect(self.update_from_slider)
        self.channels = self.wav_info['num_channels']
        end = min(10, self.wav_info['duration'])
        self.update_plot(0, end)
        p = QtCore.QUrl.fromLocalFile(self.wav_path)
        self.m_audioOutput.setMedia(QtMultimedia.QMediaContent(p))
        self.updatePlayTime(0)
        self.m_audioOutput.setMinTime(0)
        self.m_audioOutput.setStartTime(0)
        self.m_audioOutput.setMaxTime(end)
        self.m_audioOutput.setCurrentTime(0)

    def update_utterance(self, utterance, zoom=False):
        if utterance is None:
            return
        self.utterance = utterance
        self.reset_text()
        if self.utterance in self.corpus.segments:
            segment = self.corpus.segments[self.utterance]
            file_name = segment['file_name']
            begin = segment['begin']
            end = segment['end']
            begin = float(begin)
            end = float(end)
            self.update_file_name(file_name)
            self.selected_min = begin
            self.selected_max = end
            # if self.max_time is not None and self.min_time is not None:
            #    if self.min_time + 1 <= end <= self.max_time - 1:
            #        return
            #    if self.min_time + 1 <= begin <= self.max_time - 1:
            #        return
            self.long_file = True
            begin -= 1
            end += 1
        else:
            self.update_file_name(self.utterance)
            self.long_file = False
            self.wave_data = None
            begin = 0
            end = self.wav_info['duration']
        if not zoom:
            begin, end = None, None
        self.update_plot(begin, end)
        if self.long_file:
            self.m_audioOutput.setMaxTime(self.selected_max)
        self.updatePlayTime(self.selected_min)
        self.m_audioOutput.setStartTime(self.selected_min)
        self.m_audioOutput.setCurrentTime(self.selected_min)
        self.m_audioOutput.setMaxTime(self.selected_max)

    def update_selected_times(self, region):
        self.selected_min, self.selected_max = region.getRegion()
        self.selected_min /= self.sr
        self.selected_max /= self.sr
        self.updatePlayTime(self.selected_min)
        self.m_audioOutput.setStartTime(self.selected_min)
        self.m_audioOutput.setCurrentTime(self.selected_min)
        self.m_audioOutput.setMaxTime(self.selected_max)

    def update_selected_speaker(self, pos):
        pos = pos.y()
        if pos > self.min_point:
            return
        y_range = self.max_point - self.min_point
        speaker_tier_range = (y_range / 2)
        new_speaker = None
        for k, s_id in self.speaker_mapping.items():
            top_pos = self.min_point - speaker_tier_range * (s_id - 1)
            bottom_pos = top_pos - speaker_tier_range
            if top_pos > pos > bottom_pos:
                new_speaker = k
        if new_speaker != self.selected_utterance['speaker']:
            self.updateSpeaker.emit(self.utterance, new_speaker)


    def update_plot(self, begin, end):
        self.ax.getPlotItem().clear()
        self.ax.setBackground(self.background_color)
        if self.corpus is None:
            return
        if self.wav_path is None:
            return
        from functools import partial
        if begin is None and end is None:
            begin = self.min_time
            end = self.max_time
        if end is None:
            return
        if end <= 0:
            end = self.max_time
        if begin < 0:
            begin = 0
        if self.long_file:
            duration = end - begin
            self.wave_data, self.sr = librosa.load(self.wav_path, offset=begin, duration=duration + 2, sr=None, mono=False)

        elif self.wave_data is None:
            self.wave_data, self.sr = librosa.load(self.wav_path, sr=None)
            # Normalize y1 between 0 and 2
            self.wave_data /= np.max(np.abs(self.wave_data), axis=0)  # between -1 and 1
            self.wave_data += 1  # shift to 0 and 2
        #self.m_audioOutput.setData(self.wave_data, self.sr)
        begin_samp = int(begin * self.sr)
        end_samp = int(end * self.sr)
        window_size = end - begin
        try:
            self.scroll_bar.valueChanged.disconnect(self.update_from_slider)
        except TypeError:
            pass
        self.scroll_bar.setValue(begin)
        self.scroll_bar.setPageStep(window_size)
        self.scroll_bar.setMaximum(self.wav_info['duration'] - window_size)
        self.scroll_bar.valueChanged.connect(self.update_from_slider)
        self.min_time = begin
        self.max_time = end
        self.ax.addItem(self.line)
        self.updatePlayTime(self.min_time)
        wave_pen = pg.mkPen(self.wave_line_color, width=1)
        if len(self.wave_data.shape) > 1 and self.wave_data.shape[0] == 2:
            if not self.long_file:
                y0 = self.wave_data[0, begin_samp:end_samp]
                y1 = self.wave_data[1, begin_samp:end_samp]
                x = np.arange(start=begin_samp, stop=end_samp)
            else:
                y0 = self.wave_data[0, :]
                y1 = self.wave_data[1, :]
                x = np.arange(start=begin_samp, stop=begin_samp + y0.shape[0])

            # Normalize y0 between 2 and 4
            y0 /= np.max(np.abs(y0), axis=0)  # between -1 and 1
            y0[np.isnan(y0)] = 0
            y0 += 3  # shift to 2 and 4
            # Normalize y1 between 0 and 2
            y1 /= np.max(np.abs(y1), axis=0)  # between -1 and 1
            y1[np.isnan(y1)] = 0
            y1 += 1  # shift to 0 and 2
            pen = pg.mkPen(self.break_line_color, width=1)
            pen.setStyle(QtCore.Qt.DotLine)
            sub_break_line = pg.InfiniteLine(
                pos=2,
                angle=0,
                pen=pen,
                movable=False  # We have our own code to handle dragless moving.
            )
            self.ax.addItem(sub_break_line)

            self.ax.plot(x, y0, pen=wave_pen)
            self.ax.plot(x, y1, pen=wave_pen)
            self.min_point = 0
            self.max_point = 4

        else:
            if not self.long_file:
                y = self.wave_data[begin_samp:end_samp]
                x = np.arange(start=begin_samp, stop=begin_samp + y.shape[0])
            else:
                y = self.wave_data
                y /= np.max(np.abs(y), axis=0)  # between -1 and 1
                y += 1  # shift to 0 and 2
                x = np.arange(start=begin_samp, stop=begin_samp + y.shape[0])
            self.min_point = 0
            self.max_point = 2
            self.ax.plot(x, y, pen=wave_pen)

        if self.file_name in self.corpus.speaker_ordering:
            break_line = pg.InfiniteLine(
                pos=self.min_point,
                angle=0,
                pen=pg.mkPen(self.break_line_color, width=2),
                movable=False  # We have our own code to handle dragless moving.
            )
            self.ax.addItem(break_line)

            y_range = self.max_point - self.min_point
            speaker_tier_range = (y_range / 2)
            speaker_ind = 1
            self.speaker_mapping = {}
            # Figure out speaker mapping first
            speakers = set()
            if not self.show_all_speakers:
                for u in self.file_utts:
                    if u['end'] - self.min_time <= 0:
                        continue
                    if self.max_time - u['begin'] <= 0:
                        break
                    speakers.add(u['speaker'])
            for sp in self.corpus.speaker_ordering[self.file_name]:
                if not self.show_all_speakers and sp not in speakers:
                    continue
                if sp not in self.speaker_mapping:
                    self.speaker_mapping[sp] = speaker_ind
                    speaker_ind += 1

            for u in self.file_utts:
                if u['end'] - self.min_time <= 0:
                    continue
                if self.max_time - u['begin'] <= 0:
                    break
                s_id = self.speaker_mapping[u['speaker']]
                if u['utt'] == self.utterance:
                    self.selected_utterance = u
                    t, reg, fill = construct_text_region(u, self.min_time, self.max_time, self.min_point, self.max_point,
                                                         self.sr, s_id, selected_range_color=self.selected_range_color,
                                                         selected_line_color=self.selected_line_color,
                                                         break_line_color=self.break_line_color,
                                                         text_color=self.text_color, plot_text_font=self.plot_text_font,
                                                         interval_background_color=self.interval_background_color,
                                                         plot_text_width=self.plot_text_width)

                    self.ax.addItem(fill)
                    self.ax.addItem(reg)
                    func = partial(self.update_utt_times, u)
                    reg.sigRegionChangeFinished.connect(func)
                    reg.sigRegionChangeFinished.connect(self.update_selected_times)
                    reg.dragFinished.connect(self.update_selected_speaker)
                else:
                    t, bl, el, fill = construct_text_box(u, self.min_time, self.max_time, self.min_point, self.max_point,
                                                         self.sr, s_id, break_line_color=self.break_line_color,
                                                         text_color=self.text_color, plot_text_font=self.plot_text_font,
                                                         interval_background_color=self.interval_background_color,
                                                         plot_text_width=self.plot_text_width)
                    self.ax.addItem(bl)
                    self.ax.addItem(el)
                    self.ax.addItem(fill)
                if u['end'] - self.min_time <= 1:
                    continue
                if self.max_time < u['begin'] <= 1:
                    continue
                self.ax.addItem(t)
            num_speakers = speaker_ind - 1
            min_y = self.min_point - speaker_tier_range * num_speakers
            self.ax.setYRange(min_y, self.max_point)
            for k, s_id in self.speaker_mapping.items():
                t = pg.TextItem(k, anchor=(0, 0), color=self.text_color)
                font = t.textItem.font()
                font.setPointSize(12)
                t.setFont(font)
                top_pos = self.min_point - speaker_tier_range * (s_id - 1)
                bottom_pos = top_pos - speaker_tier_range
                mid_pos = ((top_pos - bottom_pos) / 2) + bottom_pos
                t.setPos(begin_samp, top_pos)
                self.ax.addItem(t)
                break_line = pg.InfiniteLine(
                    pos=bottom_pos,
                    angle=0,
                    pen=pg.mkPen(self.break_line_color, width=1),
                    movable=False  # We have our own code to handle dragless moving.
                )
                self.ax.addItem(break_line)
        else:
            self.ax.setYRange(self.min_point, self.max_point)
        self.ax.setXRange(begin_samp, end_samp)
        self.ax.getPlotItem().getAxis('bottom').setScale(1 / self.sr)
        self.m_audioOutput.setMinTime(self.min_time)
        self.m_audioOutput.setMaxTime(self.max_time)

    def update_utt_times(self, utt, x):
        beg, end = x.getRegion()
        new_begin = round(beg / self.sr, 4)
        new_end = round(end / self.sr, 4)
        if new_end - new_begin > 100:
            x.setRegion((utt['begin'] * self.sr, utt['end'] * self.sr))
            return
        utt['begin'] = new_begin
        utt['end'] = new_end
        x.setSpan(int(new_begin * self.sr), int(new_end * self.sr))
        old_utt = utt['utt']
        speaker = self.corpus.utt_speak_mapping[old_utt]
        file = self.corpus.utt_file_mapping[old_utt]
        text = self.corpus.text_mapping[old_utt]
        if old_utt in self.corpus.segments:
            seg = self.corpus.segments[old_utt]
            filename = seg['file_name']
            new_utt = '{}_{}_{}_{}'.format(speaker, filename, utt['begin'], utt['end']).replace('.', '_')
            new_seg = {'file_name': filename, 'begin': utt['begin'], 'end': utt['end'], 'channel': seg['channel']}
            utt['utt'] = new_utt
        else:
            new_seg = None

        self.corpus.delete_utterance(old_utt)
        self.corpus.add_utterance(new_utt, speaker, file, text, seg=new_seg)
        self.utterance = new_utt
        self.update_plot(self.min_time, self.max_time)
        self.refreshCorpus.emit(new_utt)

    def reset_text(self):
        if not self.corpus or self.utterance not in self.corpus.text_mapping:
            self.utterance = None
            self.audio = None
            self.sr = None
            self.text_widget.setText('')
            return
        text = self.corpus.text_mapping[self.utterance]
        words = text.split(' ')
        mod_words = []
        if self.dictionary is not None:
            for i, w in enumerate(words):
                if i != len(words) - 1:
                    space = ' '
                else:
                    space = ''
                if not self.dictionary.check_word(w):
                    w = '<span style=\" font-size: 20pt; font-weight:600; color:#ff0000;\" >{}{}</span>'.format(w,
                                                                                                                space)
                else:
                    w = '<span style=\" font-size: 20pt\" >{}{}</span>'.format(w, space)
                mod_words.append(w)
            self.text_widget.setText(''.join(mod_words))

        else:
            mod_words = words
            self.text_widget.setText(' '.join(mod_words))
        self.speaker_dropdown.setCurrentText(self.corpus.utt_speak_mapping[self.utterance])

    def showError(self, e):
        reply = DetailedMessageBox()
        reply.setDetailedText(str(e))
        ret = reply.exec_()

    def play_audio(self):
        if self.m_audioOutput.state() in [QtMultimedia.QMediaPlayer.StoppedState,
                                          QtMultimedia.QMediaPlayer.PausedState]:
            self.m_audioOutput.play()
        elif self.m_audioOutput.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.m_audioOutput.pause()

    def zoom_in(self):
        shift = round((self.max_time - self.min_time) * 0.25, 3)
        cur_duration = self.max_time - self.min_time
        if cur_duration < 2:
            return
        if cur_duration - 2 * shift < 1:
            shift = (cur_duration - 1) / 2
        self.min_time += shift
        self.max_time -= shift
        cur_time = self.m_audioOutput.currentTime()
        self.update_plot(self.min_time, self.max_time)
        self.m_audioOutput.setStartTime(cur_time)
        self.m_audioOutput.setCurrentTime(cur_time)
        self.updatePlayTime(cur_time)

    def zoom_out(self):
        shift = round((self.max_time - self.min_time) * 0.25, 3)
        cur_duration = self.max_time - self.min_time
        if cur_duration + 2 * shift > 20:
            shift = (20 - cur_duration) / 2
        self.min_time -= shift
        self.max_time += shift
        if self.max_time > self.wav_info['duration']:
            self.max_time = self.wav_info['duration']
        if self.min_time < 0:
            self.min_time = 0
        cur_time = self.m_audioOutput.currentTime()
        self.update_plot(self.min_time, self.max_time)
        self.m_audioOutput.setStartTime(cur_time)
        self.m_audioOutput.setCurrentTime(cur_time)
        self.updatePlayTime(cur_time)

    def pan_left(self):
        shift = round((self.max_time - self.min_time) * 0.25, 3)
        if self.min_time < shift:
            shift = self.min_time
        self.min_time -= shift
        self.max_time -= shift
        if self.min_time < 0:
            self.max_time -= self.min_time
            self.min_time = 0
        cur_time = self.m_audioOutput.currentTime()
        self.update_plot(self.min_time, self.max_time)
        self.m_audioOutput.setStartTime(cur_time)
        self.m_audioOutput.setCurrentTime(cur_time)
        self.updatePlayTime(cur_time)

    def pan_right(self):
        shift = round((self.max_time - self.min_time) * 0.25, 3)
        self.min_time += shift
        self.max_time += shift
        if self.max_time > self.wav_info['duration']:
            self.min_time -= self.max_time - self.wav_info['duration']
            self.min_time = round(self.min_time, 4)
            self.max_time = self.wav_info['duration']
        cur_time = self.m_audioOutput.currentTime()
        self.update_plot(self.min_time, self.max_time)
        self.m_audioOutput.setStartTime(cur_time)
        self.m_audioOutput.setCurrentTime(cur_time)
        self.updatePlayTime(cur_time)

    def updatePlayTime(self, time):
        if not time:
            return
        #if self.max_time and time > self.max_time:
        #    return
        if self.sr:
            pos = int(time * self.sr)
            self.line.setPos(pos)

    def notified(self, current_time):
        if self.m_audioOutput.min_time is None:
            return
        self.updatePlayTime(current_time)

    def handleAudioState(self, state):
        if state == QtMultimedia.QMediaPlayer.StoppedState:
            self.m_audioOutput.setPosition(self.m_audioOutput.start_time)
            self.updatePlayTime(self.m_audioOutput.currentTime())


class InformationWidget(QtWidgets.QWidget):  # pragma: no cover
    resetDictionary = QtCore.pyqtSignal()
    saveDictionary = QtCore.pyqtSignal(object)
    newSpeaker = QtCore.pyqtSignal()

    def __init__(self, parent):
        super(InformationWidget, self).__init__(parent=parent)
        self.setMaximumWidth(500)
        # self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.dictionary = None
        self.corpus = None
        self.g2p_model = None

        self.tabs = QtWidgets.QTabWidget()

        layout = QtWidgets.QVBoxLayout()
        dict_layout = QtWidgets.QVBoxLayout()
        self.dictionary_widget = QtWidgets.QTableWidget()
        # self.dictionary_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.dictionary_widget.verticalHeader().setVisible(False)
        self.dictionary_widget.setColumnCount(2)
        self.dictionary_widget.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.dictionary_widget.setHorizontalHeaderLabels(['Word', 'Pronunciation'])
        dict_layout.addWidget(self.dictionary_widget)

        button_layout = QtWidgets.QHBoxLayout()
        self.reset_button = QtWidgets.QPushButton('Reset dictionary')
        self.save_button = QtWidgets.QPushButton('Save dictionary')

        self.reset_button.clicked.connect(self.resetDictionary.emit)
        self.save_button.clicked.connect(self.create_dictionary_for_save)

        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.save_button)

        dict_layout.addLayout(button_layout)

        dict_widget = QtWidgets.QWidget()
        dict_widget.setLayout(dict_layout)

        speaker_layout = QtWidgets.QVBoxLayout()
        self.speaker_widget = QtWidgets.QTableWidget()
        self.speaker_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.speaker_widget.verticalHeader().setVisible(False)
        self.speaker_widget.setColumnCount(2)
        self.speaker_widget.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.speaker_widget.setHorizontalHeaderLabels(['Speaker', 'Utterances'])
        speaker_layout.addWidget(self.speaker_widget)

        add_layout = QtWidgets.QHBoxLayout()
        self.speaker_edit = QtWidgets.QLineEdit()
        self.save_speaker_button = QtWidgets.QPushButton('Add speaker')

        self.speaker_edit.returnPressed.connect(self.save_speaker)
        self.save_speaker_button.clicked.connect(self.save_speaker)

        add_layout.addWidget(self.speaker_edit)
        add_layout.addWidget(self.save_speaker_button)

        speaker_layout.addLayout(add_layout)

        speak_widget = QtWidgets.QWidget()
        speak_widget.setLayout(speaker_layout)

        self.tabs.addTab(dict_widget, 'Dictionary')
        self.tabs.addTab(speak_widget, 'Speakers')
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def save_speaker(self):
        new_speaker = self.speaker_edit.text()
        if new_speaker in self.corpus.speak_utt_mapping:
            return
        if not new_speaker:
            return
        self.corpus.speak_utt_mapping[new_speaker] = []
        self.refresh_speakers()
        self.newSpeaker.emit()

    def refresh_speakers(self):
        if self.corpus is None:
            return
        speakers = sorted(self.corpus.speak_utt_mapping.keys())
        self.speaker_widget.setRowCount(len(speakers))

        for i, s in enumerate(speakers):
            self.speaker_widget.setItem(i, 0, QtWidgets.QTableWidgetItem(s))
            self.speaker_widget.setItem(i, 1, QtWidgets.QTableWidgetItem(str(len(self.corpus.speak_utt_mapping[s]))))

    def update_corpus(self, corpus):
        self.corpus = corpus
        self.refresh_speakers()

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
            pron = {'pronunciation': pronunciation, 'probability': None}
            words[word].append(pron)
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
            pronunciation = results[word][0]
            pron = {'pronunciation': tuple(pronunciation.split(' ')), 'probability': 1}
            self.dictionary.words[word].append(pron)
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
