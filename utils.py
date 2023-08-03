"""Utilities for reading files."""

import collections.abc

def progress(iterable):
    """Iterate over `iterable`, showing progress if appropriate."""
    import os, sys
    if os.isatty(sys.stderr.fileno()):
        try:
            import tqdm
            return tqdm.tqdm(iterable)
        except ImportError:
            return iterable
    else:
        return iterable

class Vocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""
    def __init__(self):
        words = {'<BOS>', '<EOS>', '<UNK>'}
        self.num_to_word = list(words)    
        self.word_to_num = {word:num for num, word in enumerate(self.num_to_word)}
    def add(self, word):
        if word in self: return
        num = len(self.num_to_word)
        self.num_to_word.append(word)
        self.word_to_num[word] = num
    def discard(self, word):
        raise NotImplementedError()
    def __contains__(self, word):
        return word in self.word_to_num
    def __len__(self):
        return len(self.num_to_word)
    def __iter__(self):
        return iter(self.num_to_word)

    def numberize(self, word):
        """Convert a word into a number."""
        if word in self.word_to_num:
            return self.word_to_num[word]
        else: 
            return self.word_to_num['<UNK>']

    def denumberize(self, num):
        """Convert a number into a word."""
        return self.num_to_word[num]

def read_parallel(filename):
    """Read data from the file named by `filename`.

    The file should be in the format:

    我 不 喜 欢 沙 子 \t I do n't like sand

    where \t is a tab character.

    Argument: filename
    Returns: list of pairs of lists of strings. <BOS> and <EOS> are added to all sentences.
    """
    data = []
    for line in open(filename):
        fline, eline = line.split('\t')
        fwords = ['<BOS>'] + fline.split() + ['<EOS>']
        ewords = ['<BOS>'] + eline.split() + ['<EOS>']
        data.append((fwords, ewords))
    return data

def read_mono(filename):
    """Read sentences from the file named by `filename`.

    Argument: filename
    Returns: list of lists of strings. <BOS> and <EOS> are added to each sentence.
    """
    data = []
    for line in open(filename):
        words = ['<BOS>'] + line.split() + ['<EOS>']
        data.append(words)
    return data
