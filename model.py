import torch
from torch import nn
import torch.nn.functional as F

import os

import torchtext.transforms as T

from preprocessing import tokens_to_ids, text_transform, eng, eng_tokenize

CURRENT_DIR = os.getcwd()
MODEL_FILE_NAME = "TextClassifierModel.pth"
EMBED_FILE_NAME = "embeddings.pth"

MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "models", MODEL_FILE_NAME)
EMBED_SAVE_PATH = os.path.join(CURRENT_DIR, "models", EMBED_FILE_NAME)

#setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


class ConvClassifier(nn.Module):
    
    def __init__(self, size_vocab, embedding_dim, n_filters, 
                 filter_sizes, output_dim=1, pad_ix=0, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings=size_vocab, 
                                      embedding_dim=embedding_dim, 
                                      padding_idx=pad_ix)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels=1, out_channels=n_filters, 
                                    kernel_size = (fs, embedding_dim)) for fs in filter_sizes
                                    ])
        
        self.classifier = nn.Linear(
                                    in_features=len(filter_sizes) *  n_filters, 
                                    out_features=output_dim
                                    )
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self,text):
        """
        Defines the processing ot the tokenized text
        """
        # converts indices to word embeddings
        embedded = self.embedding(text)

        # Converts to shape as expected by Conv2d  
        embedded = embedded.unsqueeze(1)

        # Applies convolution to extract features
        feature_maps = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # Applies max pooling to generate the input for dense layer. This allows dealing with
        # variable-length sequences 
        pooled = [F.max_pool1d(fm, kernel_size=fm.shape[2]).squeeze(2) for fm in feature_maps]

        # Adds a dropout layer to reduce overfitting
        cat = self.dropout(torch.cat(pooled, dim=1))
        
        out = self.classifier(cat)
        
        return out
    

VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
NUM_FILTERS = 100
FILTER_SIZES = [2, 3, 4, 5, 6]

convClassifier = ConvClassifier(VOCAB_SIZE, EMBEDDING_DIM, 
                                NUM_FILTERS, FILTER_SIZES, 
                                dropout=0.6)

convClassifier.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device(device)))

convClassifier = convClassifier.to(device)

embeddings = torch.load(EMBED_SAVE_PATH)

# we need to populate our model embedding layer with the pretrained embeddings 
#convClassifier.embedding.weight.data.copy_(embeddings)
#convClassifier.embedding.weight.data[3] = torch.zeros(EMBEDDING_DIM)
#convClassifier.embedding.weight.data[0] = torch.zeros(EMBEDDING_DIM)


def make_predictions(model, data, return_prediction=True, threshold=0.5, device=device):
    """
    Returns the probability 'prob' of a news title being real.
    If return_precition = True then it also returns a prediction according to
    pred = (prob >= threshold)
    """
    response_list = []

    model.eval()
    with torch.inference_mode():
        for i, row in enumerate(data):
            logit = model(torch.tensor(row).unsqueeze(0).to(device))
            prob = torch.sigmoid(logit).item()
    
            if return_prediction:
                pred = int(prob >= threshold)

            response_list.append({"id":i, "prob": prob, "pred":pred})
    
    return response_list


def predict_sentence(model, sentence, threshold=0.5, device=device):
    """
    Make a prediction from a single text string
    """
    transformed_sentence = text_transform(sentence)

    # convert to tensor of expected shape
    transformed_sentence = torch.tensor(transformed_sentence).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        logit = model(transformed_sentence)

    # apply sigmoid to get a prob
    pred = torch.sigmoid(logit).item()
    
    pred = pred, int(pred >= threshold)

    return {"prob": pred[0], "pred": pred[1]}

