import collections
import os
import re

import numpy as np
from six.moves import cPickle

from .functional import *

idx_files = ["data/words_vocab.pkl", "data/characters_vocab.pkl"]

[idx_to_word, idx_to_char] = [cPickle.load(open(file, "rb")) for file in idx_files]
[word_to_idx, char_to_idx] = [dict(zip(idx, range(len(idx)))) for idx in [idx_to_word, idx_to_char]]

max_word_len = np.amax([len(word) for word in idx_to_word])


def encode_characters(characters: list):
    """
    [summary] encodes a list of characters to idx

    Args:
        characters ([type]): [description] a list containing string character

    Returns:
        [type]: [description] list of encoded characters to char_idx
    """
    word_len = len(characters)
    to_add = max_word_len - word_len
    characters_idx = [char_to_idx[i] for i in characters] + to_add * [char_to_idx[""]]
    return characters_idx


def preprocess_data(data_files: list, idx_files: list, tensor_files: list, file: bool, str=""):
    """
    A function to pre-processes into word and character tensors

    Args:
        data_files (list): [description] a list of data files
        idx_files (list): [description] a list of idx files that contain ids for corresponding word and characters
        tensor_files (list): [description] files with word/character tensor information
        file (bool): [description] indicates whether or not to read an existing file
        str (str, optional): [description]. start string for a potential new file. Defaults to ''.
    """

    if file:
        data = [open(file, "r").read() for file in data_files]
    else:
        data = [str + "\n"]

    data_words = [[line.split() for line in target.split("\n")] for target in data]
    data_words = [[[word for word in target if word in idx_to_word] for target in yo] for yo in data_words]

    word_tensor = np.array([[list(map(word_to_idx.get, line)) for line in target] for target in data_words])
    np.save(tensor_files[0][0], word_tensor[0])

    character_tensor = np.array([[list(map(encode_characters, line)) for line in target] for target in data_words])
    np.save(tensor_files[1][0], character_tensor[0])
