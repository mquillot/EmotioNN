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

    tokenizer: Any
    "Tokenizer used to extract tokens from raw sentences"

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: Tokenizer,
    ):
        """Initialize the Dataset with a CSV file

        A tokenizer from HuggingFace library is also required.
        Then to this tokenizer, a vocabulary id is automatically
        computed for each token when getting an item (with
        `__getitem__` method)

        Args:
            csv_path (str | Path): path of the csv containing data
            tokenizer (Tokenizer): tokenizer from HuggingFace library
        """
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_path, sep=",")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (torch.tensor(self.tokenizer.encode(row["text"]).ids), row["label"])
