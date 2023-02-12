
import os
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess

class Sentences(object):
    def __init__(self, files_path, files_list):
        self.files_list = files_list
        self.files_path = files_path
        self.sentence_count = 0
        self.epoch = 0

    def __iter__(self):
        print(f"Epoch {self.epoch}")
        self.epoch += 1

        # Load all the txt files.
        files = self.files_list
        for fname in files:
            file_path = os.path.join(self.files_path, fname)
            with open(file_path) as f_input:
                corpus = f_input.read()
            raw_sentences = sent_tokenize(corpus)
            for sentence in raw_sentences:
                if len(sentence) > 0:
                    self.sentence_count += 1
                    yield simple_preprocess(sentence)