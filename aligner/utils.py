import re

from aligner.dictionary import OrthographicDictionary


def check_tools():
    pass


def no_dictionary(corpus_object, output_directory):
    """Creates a dictionary based on the orthography.

    When the --nodict option is specified, the aligner uses the orthography to construct pronunciations for
    words in the corpus.

    Parameters
    ----------
    corpus_object
        Corpus to align
    output_directory : str
        Specifies where to put the newly-created dictionary

    Returns
    -------
    dictionary
        Orthographic dictionary created from the corpus

    """

    created_dict = {}
    text = corpus_object.text_mapping
    for i in text:
        split = text[i].split(' ')
        for word in split:
            updated = re.sub('\W', '', word)
            pronunciation = list(updated)
            if list(word)[0] != '[' and list(word)[0] != '{' and list(word)[0] != '<':
                transcription = pronunciation
                created_dict[word] = transcription
    d = OrthographicDictionary(created_dict, output_directory)
    return d
