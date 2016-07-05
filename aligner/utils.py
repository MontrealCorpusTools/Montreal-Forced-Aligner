import re

from aligner.dictionary import OrthographicDictionary


def check_tools():
    pass

def no_dictionary(corpus_object, output_directory):
	created_dict = {}
	text = corpus_object.text_mapping
	for i in text:
		split = text[i].split(' ')
		for word in split:
			updated = re.sub('[^a-z]', '', word)
			pronunciation = list(updated)
			if list(word)[0] != '[' and list(word)[0] != '{' and list(word)[0] != '<':
				transcription = ' '.join(pronunciation)
				created_dict[word] = transcription
	d = OrthographicDictionary(created_dict, output_directory)
	return d


