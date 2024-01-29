import torch 
from torch import nn
import torch.nn.functional as F

import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy

from torch.utils.data import Dataset, DataLoader

import os
import json

import pandas as pd

CURRENT_DIR = os.getcwd()
VOCAB_FILE_NAME = "vocab.pth"

VOCAB_SAVE_PATH = os.path.join(CURRENT_DIR, "models", VOCAB_FILE_NAME)

# Load the English model to tokenize English text
eng = spacy.load("en_core_web_sm")

# we need to tokenize text to encode words as numbers
def eng_tokenize(text):
    "Tokenize an english text and returns a list of tokens"
    return [token.text for token in eng.tokenizer(text)]

# load model vocabulary 
vocab = torch.load(VOCAB_SAVE_PATH)


# this transformation converts from tokens to indices and adds special tokens 
tokens_to_ids = T.Sequential(
    T.VocabTransform(vocab), #convert from tokens to ids
    T.AddToken(1, begin=True), # add <sos> token to the start 
    T.AddToken(2, begin=False), # add <eos> to the end 
    )


def text_transform(row):
    "Return tokens from a text and convert to indices"
    tokens = tokens_to_ids(eng_tokenize(row))
    return tokens


def applyPadding(batch):
    """
    Apply padding to batch of tokenized sentences
    """
    return T.ToTensor(0)(batch)


class myDataset(Dataset):
    def __init__(self, data, transform, data_key='data', id_col="id", text_col="text"):
        self.text_col = text_col
        if isinstance(data, dict):
            self.data = data[data_key]
            self.data = pd.DataFrame.from_dict(self.data)
            if id_col:
                self.data.set_index(keys=id_col, inplace=True)
        else:
            raise Exception("A JSON file is expected")
        self.transform = transform
        #print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx][self.text_col]
        #print(row)
        row = self.transform(row)
        return row
