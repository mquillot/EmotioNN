"""Train an Simple Attention Mechanism NN to recognize emotion in a text

Dataset from there: https://huggingface.co/datasets/dair-ai/emotion"""

from emotionn.torch_attention import SimpleAttentionNetwork
from emotionn.dataset import TextualEmotionDetectionDataset
from torch.utils.data import DataLoader
from torch import optim
import logging
from torch import nn
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import pandas as pd


CHECKPOINTS_FOLDER = "checkpoints"
MAX_N_EPOCHS = 10

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Create the tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    data = pd.read_csv("data/Emotion-detection-from-text/training.csv")

    tokenizer.train_from_iterator(
        iterator=[row["text"] for _, row in data.iterrows()],
        trainer=BpeTrainer(),
    )

    # Load training set and dataloader
    train_set = TextualEmotionDetectionDataset(
        csv_path="data/Emotion-detection-from-text/training.csv", tokenizer=tokenizer
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
    )

    # Load validation set and dataloader
    val_set = TextualEmotionDetectionDataset(
        csv_path="data/Emotion-detection-from-text/validation.csv",
        tokenizer=tokenizer,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
    )

    logging.info("Length of training vocabulary: %s+1", tokenizer.get_vocab_size())

    # Create model and Optimizer
    model = SimpleAttentionNetwork(
        vocab_size=tokenizer.get_vocab_size(),  # +1 for unknown token
        word_emb_size=10,
        query_key_length=6,
        nb_outputs_by_word=6,
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    start_epoch: int = 0

    # Load model and optimizer params if last checkpoint
    last_checkpoint = Path(CHECKPOINTS_FOLDER, "last_checkpoint.tar")
    if last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # Not checkpoints, create the folder if necessary
    if not Path(CHECKPOINTS_FOLDER).is_dir():
        Path(CHECKPOINTS_FOLDER).mkdir(parents=True, exist_ok=True)

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    model.train()
    for epoch in range(start_epoch, MAX_N_EPOCHS):
        logging.info("Epoch: %s", epoch)

        # Training step
        train_losses = []
        for texts, labels in tqdm(train_dataloader):
            for text, label in zip(texts, labels):
                model_output = model(text)
                optimizer.zero_grad()
                loss = loss_fn(model_output, label)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.detach().cpu().item())
        logging.info("Train loss: %s", np.mean(train_losses))

        # Validation step (with early stopping)
        val_losses = []
        well_classified = 0
        with torch.no_grad():
            for texts, labels in val_dataloader:
                for text, label in zip(texts, labels):
                    model_output = model(text)

                    # Well classified?
                    if torch.equal(torch.argmax(model_output, dim=0), label):
                        well_classified += 1

                    loss = loss_fn(model_output, label)

                    val_losses.append(loss.detach().cpu().item())

                    torch.argmax(model_output, dim=0)

        logging.info("Val loss: %s", np.mean(val_losses))
        logging.info(f"Val accuracy: {well_classified / len(val_set) * 100}%")

        torch.save(
            obj={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": np.mean(train_losses),
                "val_loss": np.mean(val_losses),
            },
            f=Path(CHECKPOINTS_FOLDER, "last_checkpoint.tar"),
        )
        # TODO: implement early stopping
