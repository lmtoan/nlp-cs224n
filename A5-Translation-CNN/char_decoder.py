#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        # YOUR CODE HERE for part 2a
        # TODO - Initialize as an nn.Module.
        # - Initialize the following variables:
        # self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        # self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        # self.decoderCharEmb: Embedding matrix of character embeddings
        # self.target_vocab: vocabulary for the target language
        ###
        # Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        # - Set the padding_idx argument of the embedding matrix.
        # - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.

        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size, bidirectional=False)
        self.target_vocab = target_vocab
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size, padding_idx=self.target_vocab.char2id['<pad>'])

        # END YOUR CODE

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        # YOUR CODE HERE for part 2b
        # TODO - Implement the forward pass of the character decoder.
        X = self.decoderCharEmb(input)  # (seq_length, batch, emb)
        output, dec_hidden = self.charDecoder(X, dec_hidden)  # (seq_length, batch, hidden_size)
        seq_length, batch_size, hidden_size = output.shape
        scores = self.char_output_projection(output.view(seq_length * batch_size, hidden_size))
        return scores.view(seq_length, batch_size, -1), dec_hidden
        # END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        # YOUR CODE HERE for part 2c
        # TODO - Implement training forward pass.
        ###
        # Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        # - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        scores, dec_hidden = self.forward(char_sequence, dec_hidden)  # (seq_length, batch, len_vocab)
        seq_length, batch_size, len_vocab = scores.shape
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        loss = 0
        for sid in range(1, seq_length):
            preds = scores[sid - 1, :, :].view(-1, len_vocab)  # (batch, len_vocab)
            targets = char_sequence[sid, :]  # (batch, )
            loss += loss_fn(preds, targets)
        return loss
        # END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        # YOUR CODE HERE for part 2d
        # TODO - Implement greedy decoding.
        # Hints:
        # - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        # - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        # - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        # Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].shape[1]
        current_char = torch.LongTensor([[self.target_vocab.start_of_word] for _ in range(batch_size)]).t().to(device)
        dec_hidden = initialStates
        outputs = [[self.target_vocab.start_of_word for _ in range(batch_size)]]
        for i in range(1, max_length):
            scores, dec_hidden = self.forward(current_char, dec_hidden)  # (1, batch, vocab_size)
            probs = torch.argmax(scores.squeeze(dim=0), dim=1)
            current_char = probs.unsqueeze(dim=0)  # (1, batch)
            outputs.append(probs.data.cpu().numpy())
        outputs = np.array(outputs).T
        decodedWords = []
        for b in range(batch_size):
            word = ""
            for cid in outputs[b, 1:]:
                if cid == self.target_vocab.end_of_word:
                    break
                word += self.target_vocab.id2char[cid]
            decodedWords.append(word)
        return decodedWords
        # END YOUR CODE
