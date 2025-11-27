"""Torch"""

import torch
from torch import nn


class SimpleAttentionNetwork(torch.nn.Module):
    def __init__(
        self, vocab_size, word_emb_size=3, query_key_length=4, nb_outputs_by_word=2
    ):
        """Intit the SimpleAttentionNetwork

        Args:
            vocab_size (_type_): Size of the vocabulary of the model.
            word_emb_size (int, optional): Dimentionality of word embeddings. Defaults to 3.
            query_key_length (int, optional): Dimensionality of query vectors. Defaults to 4.
            nb_outputs_by_word (int, optional): Dimensionality of output for each word. Defaults to 2.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, word_emb_size)

        self.weights_query = torch.rand(
            (word_emb_size, query_key_length), dtype=torch.float32
        )
        self.weights_key = torch.rand(
            (word_emb_size, query_key_length), dtype=torch.float32
        )
        self.weights_value = torch.rand(
            (word_emb_size, nb_outputs_by_word), dtype=torch.float32
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)

        q = x @ self.weights_query
        k = x @ self.weights_key
        v = x @ self.weights_value
        scores = q @ torch.transpose(k, 0, 1)

        weights = self.softmax(scores)
        one_vector_output_by_word = weights @ v
        mean, _ = torch.std_mean(one_vector_output_by_word, dim=0)
        return mean
