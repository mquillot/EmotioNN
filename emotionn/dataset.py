"""Textual Emotion Detection Dataset declaration"""

from typing import Any
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import torch
from tokenizers import Tokenizer


class TextualEmotionDetectionDataset(Dataset):
    """Textual Emotion Detection Dataset"""

    data: pd.DataFrame
    "DataFrame containing the text and label data"

    # vocabulary: torchtext.vocab.Vocab
    # "Vocabulary used to create one hot tensors"

    tokenizer: Any
    "Tokenizer used to extract tokens from raw sentences"

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: Tokenizer,
    ):
        """_summary_

        Args:
            csv_path (str | Path): path of the csv containing data
            tokenizer (Tokenizer): tokenizer
            vocabulary (_type_, optional): vocabulary, unknown token index is length of vocab. Defaults to None.
        """
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_path, sep=",")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (torch.tensor(self.tokenizer.encode(row["text"]).ids), row["label"])
