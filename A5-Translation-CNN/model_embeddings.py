#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change"


class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        # A4 code
        self.embed_size = embed_size
        self.vocab = vocab
        self.embed_char_size = 50
        self.pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), self.embed_char_size, padding_idx=self.pad_token_idx)
        # End A4 code

        # YOUR CODE HERE for part 1f
        self.cnn = CNN(self.embed_char_size, self.embed_size, kernel_size=5)
        self.highway = Highway(self.embed_size)
        # END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        out = []
        for batch_words in input_tensor:
            X_emb = self.embeddings(batch_words)  # At each time step
            batch_size, m_word, e_char = X_emb.shape
            X_emb = X_emb.view(batch_size, e_char, m_word)  # (src_len*batch, emb_char, m_word) for nn.Conv1d
            X_conv = self.cnn(X_emb)  # (src_len*batch, m_word) after pooling
            X_highway = self.highway(X_conv)
            out.append(X_highway)  # (batch, emb)
        return torch.stack(out, dim=0)  # Concatenate all time steps, (sentence_length, batch_size, embed_size)

        # X_emb = self.embeddings(input_tensor)
        # src_len, batch_size, m_word, e_char = X_emb.shape
        # X_emb = X_emb.view(src_len * batch_size, e_char, m_word)  # (src_len*batch, emb_char, m_word) for nn.Conv1d
        # X_conv = self.cnn(X_emb)  # (src_len*batch, m_word) after pooling
        # X_highway = self.highway(X_conv).view(src_len, batch_size, -1)  # (src_len, batch, emb)
        # return X_highway
