import yaml
import os
import sys
import traceback
from PyQt5 import QtGui, QtCore, QtWidgets

from ..config import TEMP_DIR

from ..corpus.align_corpus import AlignableCorpus

from ..dictionary import Dictionary
from ..models import G2PModel, AcousticModel, LanguageModel, IvectorExtractor
from ..utils import get_available_g2p_languages, get_pretrained_g2p_path, \
    get_available_acoustic_languages, get_pretrained_acoustic_path, \
    get_available_dict_languages, get_dictionary_path, \
    get_available_ivector_languages, get_pretrained_ivector_path, \
    get_available_lm_languages, get_pretrained_language_model_path

from ..helper import setup_logger

from .widgets import UtteranceListWidget, UtteranceDetailWidget, InformationWidget, DetailedMessageBox

from .workers import ImportCorpusWorker


class ColorEdit(QtWidgets.QPushButton):
    def __init__(self, color, parent=None):
        super(ColorEdit, self).__init__(parent=parent)
        self.color = color
        self.updateIcon()
        self.clicked.connect(self.openDialog)

    def updateIcon(self):
        pixmap = QtGui.QPixmap(100, 100)
        pixmap.fill(self.color)
        icon = QtGui.QIcon(pixmap)
        self.setIcon(icon)

    def openDialog(self):
        color = QtWidgets.QColorDialog.getColor()

        if color.isValid():
            self.color = color
            self.updateIcon()


class OptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(OptionsDialog, self).__init__(parent=parent)
        self.base_config = {}
        self.base_config.update(parent.config)

        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.appearance_widget = QtWidgets.QWidget()
        self.appearance_widget.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.tab_widget.addTab(self.appearance_widget, 'Appearance')

        appearance_layout = QtWidgets.QFormLayout()

        self.background_color_edit = ColorEdit(QtGui.QColor(self.base_config['background_color']))
        appearance_layout.addRow('Background', self.background_color_edit)

        self.play_line_color_edit = ColorEdit(QtGui.QColor(self.base_config['play_line_color']))
        appearance_layout.addRow('Play line', self.play_line_color_edit)

        self.selected_range_color_edit = ColorEdit(QtGui.QColor(self.base_config['selected_range_color']))
        appearance_layout.addRow('Selected utterance range color', self.selected_range_color_edit)

        self.selected_line_color_edit = ColorEdit(QtGui.QColor(self.base_config['selected_line_color']))
        appearance_layout.addRow('Selected utterance line color', self.selected_line_color_edit)

        self.break_line_color_edit = ColorEdit(QtGui.QColor(self.base_config['break_line_color']))
        appearance_layout.addRow('Break line color', self.break_line_color_edit)

        self.wave_line_color_edit = ColorEdit(QtGui.QColor(self.base_config['wave_line_color']))
        appearance_layout.addRow('Waveform color', self.wave_line_color_edit)

        self.text_color_edit = ColorEdit(QtGui.QColor(self.base_config['text_color']))
        appearance_layout.addRow('Text color', self.text_color_edit)

        self.interval_background_color_edit = ColorEdit(QtGui.QColor(self.base_config['interval_background_color']))
        appearance_layout.addRow('Unselected utterance color', self.interval_background_color_edit)

        self.plot_text_font_edit = QtWidgets.QSpinBox()
        self.plot_text_font_edit.setMinimum(1)
        self.plot_text_font_edit.setValue(self.base_config['plot_text_font'])
        appearance_layout.addRow('Plot text font', self.plot_text_font_edit)

        self.plot_text_width_edit = QtWidgets.QSpinBox()
        self.plot_text_width_edit.setMinimum(1)
        self.plot_text_width_edit.setMaximum(1000)
        self.plot_text_width_edit.setValue(self.base_config['plot_text_width'])
        appearance_layout.addRow('Plot text width', self.plot_text_width_edit)

        self.appearance_widget.setLayout(appearance_layout)

        self.key_bind_widget = QtWidgets.QWidget()
        self.key_bind_widget.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.tab_widget.addTab(self.key_bind_widget, 'Key shortcuts')

        key_bind_layout = QtWidgets.QFormLayout()

        self.autosave_edit = QtWidgets.QCheckBox()
        self.autosave_edit.setChecked(self.base_config['autosave'])
        self.autosave_edit.setFocusPolicy(QtCore.Qt.ClickFocus)
        key_bind_layout.addRow('Autosave on exit', self.autosave_edit)

        self.play_key_bind_edit = QtWidgets.QKeySequenceEdit()
        self.play_key_bind_edit.setKeySequence(QtGui.QKeySequence(self.base_config['play_keybind']))
        self.play_key_bind_edit.setFocusPolicy(QtCore.Qt.ClickFocus)
        key_bind_layout.addRow('Play audio', self.play_key_bind_edit)

        self.zoom_in_key_bind_edit = QtWidgets.QKeySequenceEdit()
        self.zoom_in_key_bind_edit.setKeySequence(QtGui.QKeySequence(self.base_config['zoom_in_keybind']))
        self.zoom_in_key_bind_edit.setFocusPolicy(QtCore.Qt.ClickFocus)
        key_bind_layout.addRow('Zoom in', self.zoom_in_key_bind_edit)

        self.zoom_out_key_bind_edit = QtWidgets.QKeySequenceEdit()
        self.zoom_out_key_bind_edit.setKeySequence(QtGui.QKeySequence(self.base_config['zoom_out_keybind']))
        self.zoom_out_key_bind_edit.setFocusPolicy(QtCore.Qt.ClickFocus)
        key_bind_layout.addRow('Zoom out', self.zoom_out_key_bind_edit)

        self.pan_left_key_bind_edit = QtWidgets.QKeySequenceEdit()
        self.pan_left_key_bind_edit.setKeySequence(QtGui.QKeySequence(self.base_config['pan_left_keybind']))
        self.pan_left_key_bind_edit.setFocusPolicy(QtCore.Qt.ClickFocus)
        key_bind_layout.addRow('Pan left', self.pan_left_key_bind_edit)

        self.pan_right_key_bind_edit = QtWidgets.QKeySequenceEdit()
        self.pan_right_key_bind_edit.setKeySequence(QtGui.QKeySequence(self.base_config['pan_right_keybind']))
        self.pan_right_key_bind_edit.setFocusPolicy(QtCore.Qt.ClickFocus)
        key_bind_layout.addRow('Pan right', self.pan_right_key_bind_edit)

        self.merge_key_bind_edit = QtWidgets.QKeySequenceEdit()
        self.merge_key_bind_edit.setKeySequence(QtGui.QKeySequence(self.base_config['merge_keybind']))
        self.merge_key_bind_edit.setFocusPolicy(QtCore.Qt.ClickFocus)
        key_bind_layout.addRow('Merge utterances', self.merge_key_bind_edit)

        self.split_key_bind_edit = QtWidgets.QKeySequenceEdit()
        self.split_key_bind_edit.setKeySequence(QtGui.QKeySequence(self.base_config['split_keybind']))
        self.split_key_bind_edit.setFocusPolicy(QtCore.Qt.ClickFocus)
        key_bind_layout.addRow('Split utterances', self.split_key_bind_edit)

        self.delete_key_bind_edit = QtWidgets.QKeySequenceEdit()
        self.delete_key_bind_edit.setKeySequence(QtGui.QKeySequence(self.base_config['delete_keybind']))
        self.delete_key_bind_edit.setFocusPolicy(QtCore.Qt.ClickFocus)
        key_bind_layout.addRow('Delete utterance', self.delete_key_bind_edit)

        self.save_key_bind_edit = QtWidgets.QKeySequenceEdit()
        self.save_key_bind_edit.setKeySequence(QtGui.QKeySequence(self.base_config['save_keybind']))
        self.save_key_bind_edit.setFocusPolicy(QtCore.Qt.ClickFocus)
        key_bind_layout.addRow('Save current file', self.save_key_bind_edit)

        self.key_bind_widget.setLayout(key_bind_layout)

        layout = QtWidgets.QVBoxLayout()

        button_layout = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton('Save')
        self.save_button.clicked.connect(self.accept)
        self.save_button.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setFocusPolicy(QtCore.Qt.ClickFocus)

        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addWidget(self.tab_widget)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.setWindowTitle('Preferences')

    def generate_config(self):
        out = {
            'autosave': self.autosave_edit.isChecked(),
            'play_keybind': self.play_key_bind_edit.keySequence().toString(),
            'delete_keybind': self.delete_key_bind_edit.keySequence().toString(),
            'save_keybind': self.save_key_bind_edit.keySequence().toString(),
            'split_keybind': self.split_key_bind_edit.keySequence().toString(),
            'merge_keybind': self.merge_key_bind_edit.keySequence().toString(),
            'zoom_in_keybind': self.zoom_in_key_bind_edit.keySequence().toString(),
            'zoom_out_keybind': self.zoom_out_key_bind_edit.keySequence().toString(),
            'pan_left_keybind': self.pan_left_key_bind_edit.keySequence().toString(),
            'pan_right_keybind': self.pan_right_key_bind_edit.keySequence().toString(),
            'background_color': self.background_color_edit.color,
            'play_line_color': self.play_line_color_edit.color,
            'selected_range_color': self.selected_range_color_edit.color,
            'selected_line_color': self.selected_line_color_edit.color,
            'break_line_color': self.break_line_color_edit.color,
            'wave_line_color': self.wave_line_color_edit.color,
            'text_color': self.text_color_edit.color,
            'interval_background_color': self.interval_background_color_edit.color,
            'plot_text_font': self.plot_text_font_edit.value(),
            'plot_text_width': self.plot_text_width_edit.value(),
        }
        return out


class Application(QtWidgets.QApplication):
    def notify(self, receiver, e):
        #if e and e.type() == QtCore.QEvent.KeyPress:
        #    if e.key() == QtCore.Qt.Key_Tab:
        #        return False
        return super(Application, self).notify(receiver, e)


class MainWindow(QtWidgets.QMainWindow):  # pragma: no cover
    configUpdated = QtCore.pyqtSignal(object)
    corpusLoaded = QtCore.pyqtSignal(object)
    dictionaryLoaded = QtCore.pyqtSignal(object)
    g2pLoaded = QtCore.pyqtSignal(object)
    ivectorExtractorLoaded = QtCore.pyqtSignal(object)
    acousticModelLoaded = QtCore.pyqtSignal(object)
    languageModelLoaded = QtCore.pyqtSignal(object)
    saveCompleted = QtCore.pyqtSignal(object)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Tab:
            event.ignore()
            return
        super(MainWindow, self).keyPressEvent(event)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.config_path = os.path.join(TEMP_DIR, 'config.yaml')
        self.corpus = None
        self.dictionary = None
        self.acoustic_model = None
        self.g2p_model = None
        self.language_model = None

        self.list_widget = UtteranceListWidget(self)
        self.detail_widget = UtteranceDetailWidget(self)
        self.information_widget = InformationWidget(self)
        self.configUpdated.connect(self.detail_widget.update_config)

        self.load_config()
        self.configUpdated.connect(self.save_config)

        self.setup_key_binds()

        self.corpusLoaded.connect(self.detail_widget.update_corpus)
        self.corpusLoaded.connect(self.list_widget.update_corpus)
        self.dictionaryLoaded.connect(self.detail_widget.update_dictionary)
        self.list_widget.utteranceChanged.connect(self.detail_widget.update_utterance)
        self.list_widget.updateView.connect(self.detail_widget.update_plot)
        self.list_widget.utteranceMerged.connect(self.detail_widget.refresh_view)
        self.list_widget.utteranceDeleted.connect(self.detail_widget.refresh_view)
        self.list_widget.fileChanged.connect(self.detail_widget.update_file_name)
        self.detail_widget.selectUtterance.connect(self.list_widget.select_utterance)
        self.detail_widget.refreshCorpus.connect(self.list_widget.refresh_corpus)
        self.detail_widget.createUtterance.connect(self.list_widget.create_utterance)
        self.detail_widget.utteranceUpdated.connect(self.list_widget.update_utterance_text)
        self.detail_widget.utteranceChanged.connect(self.list_widget.setFileSaveable)
        self.detail_widget.updateSpeaker.connect(self.update_speaker)
        self.corpusLoaded.connect(self.information_widget.update_corpus)
        self.dictionaryLoaded.connect(self.list_widget.update_dictionary)
        self.dictionaryLoaded.connect(self.information_widget.update_dictionary)
        self.g2pLoaded.connect(self.information_widget.update_g2p)
        self.detail_widget.lookUpWord.connect(self.information_widget.look_up_word)
        self.detail_widget.createWord.connect(self.information_widget.create_pronunciation)
        self.list_widget.saveFile.connect(self.save_file)
        self.saveCompleted.connect(self.list_widget.setFileSaveable)
        self.information_widget.resetDictionary.connect(self.load_dictionary)
        self.information_widget.saveDictionary.connect(self.save_dictionary)
        self.information_widget.newSpeaker.connect(self.detail_widget.refresh_speaker_dropdown)
        self.status_bar = QtWidgets.QStatusBar()
        self.warning_label = QtWidgets.QLabel('<span style=\"font-weight:600; color:#ff0000;\" >Warning: This is alpha '
                                              'software, there will be bugs and issues. Please back up any data before '
                                              'using.</span>')
        self.status_label = QtWidgets.QLabel()
        self.status_bar.addPermanentWidget(self.warning_label, 1)
        self.status_bar.addPermanentWidget(self.status_label)
        self.setStatusBar(self.status_bar)
        self.wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()

        layout.addWidget(self.list_widget)
        layout.addWidget(self.detail_widget)
        layout.addWidget(self.information_widget)
        self.wrapper.setLayout(layout)
        self.setCentralWidget(self.wrapper)

        self.create_actions()
        self.create_menus()
        self.default_directory = TEMP_DIR
        self.logger = setup_logger('annotator', self.default_directory, 'debug')
        self.setWindowTitle("MFA Annotator")
        self.loading_corpus = False
        self.loading_dictionary = False
        self.loading_g2p = False
        self.loading_ie = False
        self.loading_am = False
        self.loading_lm = False
        self.saving_dictionary = False
        self.saving_utterance = False

        self.corpus_worker = ImportCorpusWorker(logger=self.logger)

        #self.corpus_worker.errorEncountered.connect(self.showError)
        self.corpus_worker.dataReady.connect(self.finalize_load_corpus)
        self.load_corpus()
        self.load_dictionary()
        self.load_g2p()
        self.load_ivector_extractor()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.config['autosave']:
            print('Saving!')
            self.save_file(self.list_widget.current_file)
        a0.accept()

    def load_config(self):
        self.config = {
            'temp_directory': TEMP_DIR,
            'current_corpus_path': None,
            'current_acoustic_model_path': None,
            'current_dictionary_path': None,
            'current_g2p_model_path': None,
            'current_language_model_path': None,
            'current_ivector_extractor_path': None,
            'autosave': True,
            'play_keybind': 'Tab',
            'delete_keybind': 'Delete',
            'save_keybind': '',
            'split_keybind': 'Ctrl+S',
            'merge_keybind': 'Ctrl+M',
            'zoom_in_keybind': 'Ctrl+I',
            'zoom_out_keybind': 'Ctrl+O',
            'pan_left_keybind': 'Left',
            'pan_right_keybind': 'Right',
            'background_color': 'black',
            'play_line_color': 'red',
            'selected_range_color': 'blue',
            'selected_line_color': 'green',
            'break_line_color': 'white',
            'wave_line_color': 'white',
            'text_color': 'white',
            'interval_background_color': 'darkGray',
            'plot_text_font': 12,
            'plot_text_width': 400,

        }
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf8') as f:
                self.config.update(yaml.load(f, Loader=yaml.SafeLoader))

        for k, v in self.config.items():
            if 'color' in k:
                self.config[k] = QtGui.QColor(v)

        os.makedirs(self.config['temp_directory'], exist_ok=True)
        self.configUpdated.emit(self.config)

    def update_speaker(self, old_utterance, speaker):

        if old_utterance is None:
            return
        if old_utterance not in self.corpus.utt_speak_mapping:
            return
        old_speaker = self.corpus.utt_speak_mapping[old_utterance]

        if old_speaker == speaker:
            return
        if not speaker:
            return
        new_utt = old_utterance.replace(old_speaker, speaker)
        file = self.corpus.utt_file_mapping[old_utterance]
        text = self.corpus.text_mapping[old_utterance]
        seg = self.corpus.segments[old_utterance]
        self.corpus.add_utterance(new_utt, speaker, file, text, seg=seg)
        self.corpus.delete_utterance(old_utterance)

        self.list_widget.refresh_corpus(new_utt)
        self.information_widget.refresh_speakers()
        self.detail_widget.refresh_utterances()
        self.list_widget.utteranceChanged.emit(new_utt, False)
        self.list_widget.setFileSaveable(True)


    def save_config(self):
        with open(self.config_path, 'w', encoding='utf8') as f:
            to_output = {}
            for k, v in self.config.items():
                if 'color' in k:
                    to_output[k] = v.name()
                else:
                    to_output[k] = v
            yaml.dump(to_output, f)

    def open_options(self):
        dialog = OptionsDialog(self)
        if dialog.exec_():
            self.config.update(dialog.generate_config())
            self.refresh_shortcuts()
            self.configUpdated.emit(self.config)

    def create_actions(self):
        self.change_temp_dir_act = QtWidgets.QAction(
            parent=self, text="Change temporary directory",
            statusTip="Change temporary directory", triggered=self.change_temp_dir)

        self.options_act = QtWidgets.QAction(
            parent=self, text="Preferences...",
            statusTip="Edit preferences", triggered=self.open_options)

        self.load_corpus_act = QtWidgets.QAction(
            parent=self, text="Load a corpus",
            statusTip="Load a corpus", triggered=self.change_corpus)

        self.load_acoustic_model_act = QtWidgets.QAction(
            parent=self, text="Load an acoustic model",
            statusTip="Load an acoustic model", triggered=self.change_acoustic_model)

        self.load_dictionary_act = QtWidgets.QAction(
            parent=self, text="Load a dictionary",
            statusTip="Load a dictionary", triggered=self.change_dictionary)

        self.load_g2p_act = QtWidgets.QAction(
            parent=self, text="Load a G2P model",
            statusTip="Load a G2P model", triggered=self.change_g2p)

        self.load_lm_act = QtWidgets.QAction(
            parent=self, text="Load a language model",
            statusTip="Load a language model", triggered=self.change_lm)

        self.load_ivector_extractor_act = QtWidgets.QAction(
            parent=self, text="Load an ivector extractor",
            statusTip="Load an ivector extractor", triggered=self.change_ivector_extractor)

    def setup_key_binds(self):
        self.play_act = QtWidgets.QAction(
            parent=self, text="Play audio",
            statusTip="Play current loaded file", triggered=self.detail_widget.play_audio)

        self.zoom_in_act = QtWidgets.QAction(
            parent=self, text="Zoom in",
            statusTip="Zoom in", triggered=self.detail_widget.zoom_in)

        self.zoom_out_act = QtWidgets.QAction(
            parent=self, text="Zoom out",
            statusTip="Zoom out", triggered=self.detail_widget.zoom_out)

        self.pan_left_act = QtWidgets.QAction(
            parent=self, text="Pan left",
            statusTip="Pan left", triggered=self.detail_widget.pan_left)

        self.pan_right_act = QtWidgets.QAction(
            parent=self, text="Pan right",
            statusTip="Pan right", triggered=self.detail_widget.pan_right)

        self.merge_act = QtWidgets.QAction(
            parent=self, text="Merge utterances",
            statusTip="Merge utterances", triggered=self.list_widget.merge_utterances)

        self.split_act = QtWidgets.QAction(
            parent=self, text="Split utterances",
            statusTip="Split utterances", triggered=self.list_widget.split_utterances)

        self.delete_act = QtWidgets.QAction(
            parent=self, text="Delete utterances",
            statusTip="Delete utterances", triggered=self.list_widget.delete_utterances)

        self.save_act = QtWidgets.QAction(
            parent=self, text="Save file",
            statusTip="Save a current file", triggered=self.save_file)

        self.refresh_shortcuts()

        self.addAction(self.play_act)
        self.addAction(self.zoom_in_act)
        self.addAction(self.zoom_out_act)
        self.addAction(self.pan_left_act)
        self.addAction(self.pan_right_act)
        self.addAction(self.merge_act)
        self.addAction(self.split_act)
        self.addAction(self.delete_act)
        self.addAction(self.save_act)


    def refresh_shortcuts(self):
        self.play_act.setShortcut(QtGui.QKeySequence(self.config['play_keybind']))
        self.zoom_in_act.setShortcut(QtGui.QKeySequence(self.config['zoom_in_keybind']))
        self.zoom_out_act.setShortcut(QtGui.QKeySequence(self.config['zoom_out_keybind']))
        self.pan_left_act.setShortcut(QtGui.QKeySequence(self.config['pan_left_keybind']))
        self.pan_right_act.setShortcut(QtGui.QKeySequence(self.config['pan_right_keybind']))
        self.merge_act.setShortcut(QtGui.QKeySequence(self.config['merge_keybind']))
        self.split_act.setShortcut(QtGui.QKeySequence(self.config['split_keybind']))
        self.delete_act.setShortcut(QtGui.QKeySequence(self.config['delete_keybind']))
        self.save_act.setShortcut(QtGui.QKeySequence(self.config['save_keybind']))


    def create_menus(self):
        self.corpus_menu = self.menuBar().addMenu("Corpus")
        self.corpus_menu.addAction(self.load_corpus_act)

        self.file_menu = self.menuBar().addMenu("Edit")
        self.file_menu.addAction(self.change_temp_dir_act)
        self.file_menu.addAction(self.options_act)
        self.dictionary_menu = self.menuBar().addMenu("Dictionary")
        self.dictionary_menu.addAction(self.load_dictionary_act)
        downloaded_dictionaries_models = self.dictionary_menu.addMenu("Downloaded dictionary")
        for lang in get_available_dict_languages():
            lang_action = QtWidgets.QAction(
                parent=self, text=lang,
                statusTip=lang, triggered=lambda: self.change_dictionary(lang))
            downloaded_dictionaries_models.addAction(lang_action)
        self.acoustic_model_menu = self.menuBar().addMenu("Acoustic model")
        self.acoustic_model_menu.addAction(self.load_acoustic_model_act)
        downloaded_acoustic_models = self.acoustic_model_menu.addMenu("MFA acoustic model")
        for lang in get_available_acoustic_languages():
            lang_action = QtWidgets.QAction(
                parent=self, text=lang,
                statusTip=lang, triggered=lambda: self.change_acoustic_model(lang))
            downloaded_acoustic_models.addAction(lang_action)

        self.g2p_model_menu = self.menuBar().addMenu("G2P model")
        self.g2p_model_menu.addAction(self.load_g2p_act)
        downloaded_g2p_models = self.g2p_model_menu.addMenu("MFA G2P model")
        for lang in get_available_g2p_languages():
            lang_action = QtWidgets.QAction(
                parent=self, text=lang,
                statusTip=lang, triggered=lambda: self.change_g2p(lang))
            downloaded_g2p_models.addAction(lang_action)

        #self.language_model_menu = self.menuBar().addMenu("Language model")
        #self.language_model_menu.addAction(self.load_lm_act)
        #downloaded_language_models = self.language_model_menu.addMenu("MFA language model")
        #for lang in get_available_lm_languages():
        #    lang_action = QtWidgets.QAction(
        #        parent=self, text=lang,
        #        statusTip=lang, triggered=lambda: self.change_lm(lang))
        #    downloaded_language_models.addAction(lang_action)

        #self.ivector_menu = self.menuBar().addMenu("Speaker classification")
        #self.ivector_menu.addAction(self.load_ivector_extractor_act)
        #downloaded_ie_models = self.ivector_menu.addMenu("MFA ivector extractor")
        #for lang in get_available_ivector_languages():
        #    lang_action = QtWidgets.QAction(
        #        parent=self, text=lang,
        #        statusTip=lang, triggered=lambda: self.change_ivector_extractor(lang))
        #    downloaded_ie_models.addAction(lang_action)

    def change_temp_dir(self):
        self.configUpdated.emit(self.config)

    def change_corpus(self):
        default_dir = self.default_directory
        if self.config['current_corpus_path']:
            default_dir = os.path.dirname(self.config['current_corpus_path'])
        corpus_directory = QtWidgets.QFileDialog.getExistingDirectory(caption='Select a corpus directory',
                                                                      directory=default_dir)
        if not corpus_directory or not os.path.exists(corpus_directory):
            return
        print(corpus_directory)
        self.default_directory = os.path.dirname(corpus_directory)
        self.config['current_corpus_path'] = corpus_directory
        self.load_corpus()
        self.configUpdated.emit(self.config)

    def load_corpus(self):
        self.loading_corpus = True
        directory = self.config['current_corpus_path']
        if directory is None or not os.path.exists(directory):
            return
        self.corpusLoaded.emit(None)
        self.status_bar.showMessage('Starting to load the corpus...', 1000)
        self.status_label.setText('Loading corpus...')
        self.corpus_worker.setParams(directory, self.config['temp_directory'])
        self.corpus_worker.start()

    def finalize_load_corpus(self, corpus):
        self.corpus = corpus
        self.loading_corpus = False
        self.corpusLoaded.emit(self.corpus)
        self.status_label.setText('Corpus {} loaded!'.format(self.corpus.name))

    def change_acoustic_model(self, lang=None):
        if not isinstance(lang, str):
            am_path, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select an acoustic model',
                                                               directory=self.default_directory,
                                                               filter="Model files (*.zip)")
        else:
            am_path = get_pretrained_acoustic_path(lang)
        if not am_path or not os.path.exists(am_path):
            return
        self.default_directory = os.path.dirname(am_path)
        self.config['current_acoustic_model_path'] = am_path
        self.load_acoustic_model()
        self.configUpdated.emit(self.config)

    def load_acoustic_model(self):
        self.loading_am = True
        am_path = self.config['current_acoustic_model_path']
        if am_path is None or not os.path.exists(am_path):
            return
        am_name, _ = os.path.splitext(os.path.basename(am_path))
        self.acoustic_model = AcousticModel(am_path, root_directory=self.config['temp_directory'])
        self.acousticModelLoaded.emit(self.acoustic_model)
        self.loading_am = False

    def change_ivector_extractor(self, lang=None):
        if not isinstance(lang, str):
            ie_path, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select a ivector extractor model',
                                                               directory=self.default_directory,
                                                               filter="Model files (*.zip)")
        else:
            ie_path = get_pretrained_ivector_path(lang)
        if not ie_path or not os.path.exists(ie_path):
            return
        self.default_directory = os.path.dirname(ie_path)
        self.config['current_ivector_extractor_path'] = ie_path
        self.load_ivector_extractor()
        self.configUpdated.emit(self.config)

    def load_ivector_extractor(self):
        self.loading_ie = True
        ie_path = self.config['current_ivector_extractor_path']
        if ie_path is None or not os.path.exists(ie_path):
            return
        ie_name, _ = os.path.splitext(os.path.basename(ie_path))
        self.ie_model = IvectorExtractor(ie_path, root_directory=self.config['temp_directory'])
        self.ivectorExtractorLoaded.emit(self.ie_model)
        self.loading_ie = False

    def change_dictionary(self, lang=None):
        if not isinstance(lang, str):
            lang = None
        if lang is None:
            dictionary_path, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select a dictionary',
                                                                       directory=self.default_directory,
                                                                       filter="Dictionary files (*.dict *.txt)")
        else:
            dictionary_path = get_dictionary_path(lang)
        if not dictionary_path or not os.path.exists(dictionary_path):
            return
        self.default_directory = os.path.dirname(dictionary_path)
        self.config['current_dictionary_path'] = dictionary_path
        self.load_dictionary()
        self.configUpdated.emit(self.config)

    def load_dictionary(self):
        self.loading_dictionary = True
        dictionary_path = self.config['current_dictionary_path']
        if dictionary_path is None or not os.path.exists(dictionary_path):
            return

        dictionary_name, _ = os.path.splitext(os.path.basename(dictionary_path))
        dictionary_temp_dir = os.path.join(self.config['temp_directory'], dictionary_name)
        self.dictionary = Dictionary(dictionary_path, dictionary_temp_dir)
        self.dictionaryLoaded.emit(self.dictionary)
        self.loading_dictionary = False

    def save_file(self, file_name):
        if self.saving_utterance:
            return
        self.saving_utterance = True

        self.status_label.setText('Saving {}...'.format(file_name))
        try:
            self.corpus.save_text_file(file_name)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(traceback.format_exception(exc_type, exc_value, exc_traceback))
            reply = DetailedMessageBox()
            reply.setDetailedText('\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
            ret = reply.exec_()
        self.saving_utterance = False
        #self.corpusLoaded.emit(self.corpus)
        self.saveCompleted.emit(False)
        self.status_label.setText('Saved {}!'.format(file_name))

    def save_dictionary(self, words):
        if self.saving_dictionary:
            return
        self.dictionary.words = words
        self.saving_dictionary = True
        with open(self.config['current_dictionary_path'], 'w', encoding='utf8') as f:
            for word, prons in sorted(self.dictionary.words.items()):
                for p in prons:
                    pronunciation = ' '.join(p['pronunciation'])
                    f.write('{} {}\n'.format(word, pronunciation))
        self.saving_dictionary = False
        self.dictionaryLoaded.emit(self.dictionary)

    def load_g2p(self):
        self.loading_g2p = True
        g2p_path = self.config['current_g2p_model_path']
        if g2p_path is None or not os.path.exists(g2p_path):
            return
        g2p_name, _ = os.path.splitext(os.path.basename(g2p_path))
        self.g2p_model = G2PModel(g2p_path, root_directory=self.config['temp_directory'])
        self.g2pLoaded.emit(self.g2p_model)
        self.loading_g2p = False

    def change_g2p(self, lang=None):
        if not isinstance(lang, str):
            g2p_path, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select a g2p model',
                                                                directory=self.default_directory,
                                                                filter="Model files (*.zip)")
        else:
            g2p_path = get_pretrained_g2p_path(lang)
        if not g2p_path or not os.path.exists(g2p_path):
            return
        self.default_directory = os.path.dirname(g2p_path)
        self.config['current_g2p_model_path'] = g2p_path
        self.load_g2p()
        self.configUpdated.emit(self.config)

    def load_lm(self):
        pass

    def change_lm(self, lang=None):
        if not isinstance(lang, str):
            lm_path, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select a language model',
                                                               directory=self.default_directory,
                                                               filter="Model files (*.zip)")
        else:
            lm_path = get_pretrained_language_model_path(lang)
        if not lm_path or not os.path.exists(lm_path):
            return
        self.default_directory = os.path.dirname(lm_path)
        self.config['current_language_model_path'] = lm_path
        self.load_lm()
        self.configUpdated.emit(self.config)
