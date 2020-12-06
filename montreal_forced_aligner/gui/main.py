import yaml
import os
from PyQt5 import QtGui, QtCore, QtWidgets

from ..config import TEMP_DIR

from ..corpus import Corpus

from ..dictionary import Dictionary
from ..models import G2PModel, AcousticModel, LanguageModel, IvectorExtractor
from ..utils import get_available_g2p_languages, get_pretrained_g2p_path, \
    get_available_acoustic_languages, get_pretrained_acoustic_path, \
    get_available_dict_languages, get_dictionary_path, \
    get_available_ivector_languages, get_pretrained_ivector_path, \
    get_available_lm_languages, get_pretrained_language_model_path

from .widgets import UtteranceListWidget, UtteranceDetailWidget, InformationWidget


class MainWindow(QtWidgets.QMainWindow):  # pragma: no cover
    configUpdated = QtCore.pyqtSignal()
    corpusLoaded = QtCore.pyqtSignal(object)
    dictionaryLoaded = QtCore.pyqtSignal(object)
    g2pLoaded = QtCore.pyqtSignal(object)
    ivectorExtractorLoaded = QtCore.pyqtSignal(object)
    acousticModelLoaded = QtCore.pyqtSignal(object)
    languageModelLoaded = QtCore.pyqtSignal(object)

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

        self.load_config()
        self.configUpdated.connect(self.save_config)
        self.list_widget = UtteranceListWidget(self)
        self.corpusLoaded.connect(self.list_widget.update_corpus)
        self.detail_widget = UtteranceDetailWidget(self)
        self.corpusLoaded.connect(self.detail_widget.update_corpus)
        self.dictionaryLoaded.connect(self.detail_widget.update_dictionary)
        self.list_widget.utteranceChanged.connect(self.detail_widget.update_utterance)
        self.information_widget = InformationWidget(self)
        self.dictionaryLoaded.connect(self.list_widget.update_dictionary)
        self.dictionaryLoaded.connect(self.information_widget.update_dictionary)
        self.g2pLoaded.connect(self.information_widget.update_g2p)
        self.detail_widget.lookUpWord.connect(self.information_widget.look_up_word)
        self.detail_widget.createWord.connect(self.information_widget.create_pronunciation)
        self.detail_widget.saveUtterance.connect(self.save_utterance)
        self.information_widget.resetDictionary.connect(self.load_dictionary)
        self.information_widget.saveDictionary.connect(self.save_dictionary)
        self.wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()

        layout.addWidget(self.list_widget)
        layout.addWidget(self.detail_widget)
        layout.addWidget(self.information_widget)
        self.wrapper.setLayout(layout)
        self.setCentralWidget(self.wrapper)

        self.create_actions()
        self.create_menus()
        self.default_directory = os.path.dirname(TEMP_DIR)
        self.setWindowTitle("MFA Annotator")
        self.loading_corpus = False
        self.loading_dictionary = False
        self.loading_g2p = False
        self.loading_ie = False
        self.loading_am = False
        self.loading_lm = False
        self.saving_dictionary = False
        self.saving_utterance = False

        self.load_corpus()
        self.load_dictionary()
        self.load_g2p()
        self.load_ivector_extractor()

    def load_config(self):
        self.config = {
            'temp_directory': TEMP_DIR,
            'current_corpus_path': None,
            'current_acoustic_model_path': None,
            'current_dictionary_path': None,
            'current_g2p_model_path': None,
            'current_language_model_path': None,
            'current_ivector_extractor_path': None,

        }
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf8') as f:
                self.config.update(yaml.load(f, Loader=yaml.SafeLoader))

        os.makedirs(self.config['temp_directory'], exist_ok=True)

    def save_config(self):
        with open(self.config_path, 'w', encoding='utf8') as f:
            yaml.dump(self.config, f)

    def create_actions(self):
        self.change_temp_dir_act = QtWidgets.QAction(
            parent=self, text="Change temporary directory",
            statusTip="Change temporary directory", triggered=self.change_temp_dir)

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

    def create_menus(self):

        self.file_menu = self.menuBar().addMenu("File")
        self.file_menu.addAction(self.change_temp_dir_act)
        self.corpus_menu = self.menuBar().addMenu("Corpus")
        self.corpus_menu.addAction(self.load_corpus_act)
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
        self.configUpdated.emit()

    def change_corpus(self):
        corpus_directory = QtWidgets.QFileDialog.getExistingDirectory(caption='Select a corpus directory',
                                                                      directory=self.default_directory)
        if not corpus_directory or not os.path.exists(corpus_directory):
            return
        print(corpus_directory)
        self.default_directory = os.path.dirname(corpus_directory)
        self.config['current_corpus_path'] = corpus_directory
        self.load_corpus()
        self.configUpdated.emit()

    def load_corpus(self):
        self.loading_corpus = True
        directory = self.config['current_corpus_path']
        if directory is None or not os.path.exists(directory):
            return
        corpus_name = os.path.basename(directory)
        corpus_temp_dir = os.path.join(self.config['temp_directory'], corpus_name)
        self.corpus = Corpus(directory, corpus_temp_dir)
        self.loading_corpus = False
        self.corpusLoaded.emit(self.corpus)

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
        self.configUpdated.emit()

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
        self.configUpdated.emit()

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
        self.configUpdated.emit()

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

    def save_utterance(self, utterance, new_text):
        if self.saving_utterance:
            return
        self.saving_utterance = True
        self.corpus.update_utterance_text(utterance, new_text)
        self.saving_utterance = False
        self.corpusLoaded.emit(self.corpus)

    def save_dictionary(self, words):
        if self.saving_dictionary:
            return
        self.dictionary.words = words
        self.saving_dictionary = True
        with open(self.config['current_dictionary_path'], 'w', encoding='utf8') as f:
            for word, prons in sorted(self.dictionary.words.items()):
                for p in prons:
                    pronunciation = ' '.join(p[0])
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
        self.configUpdated.emit()

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
        self.configUpdated.emit()
