from .config import TEMP_DIR


class Transcriber(object):
    def __init__(self, corpus, dictionary, acoustic_model, language_model, temp_directory=None,
                 call_back=None, debug=False, verbose=False):
        self.corpus = corpus
        self.dictionary = dictionary
        self.acoustic_model = acoustic_model
        self.language_model = language_model

        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.verbose = verbose
        self.debug = debug
        self.setup()

    def setup(self):
        self.dictionary.write()
        self.corpus.initialize_corpus()
        self.acoustic_model.feature_config.generate_features(self.corpus)