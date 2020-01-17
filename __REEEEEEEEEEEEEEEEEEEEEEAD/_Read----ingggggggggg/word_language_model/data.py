import os
from io import open
import torch

def ss(s):
    import sys
    print(s)
    sys.exit(1)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        # self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.train = self.tokenize('/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/language-model-data/small.txt')
        # self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        # self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:

                # print(line)
                # print(line.split())
                words = line.split() + ['<eos>']
                # print(words)
                # print(self.dictionary)
                for word in words:
                    self.dictionary.add_word(word)
                    # print(self.dictionary.idx2word)
                    # print(self.dictionary.word2idx)


        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                    # print(ids)
                    # ss('in data.py')
                idss.append(torch.tensor(ids).type(torch.int64))
            # print(idss)
            # ss('in data.py')
            ids = torch.cat(idss)
            # print(ids)

        return ids
