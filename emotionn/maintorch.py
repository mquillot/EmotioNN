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

from ignite.contrib.handlers import PiecewiseLinear
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Accuracy
from ignite.handlers import EarlyStopping
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Checkpoint
from ignite.handlers import DiskSaver
from ignite.handlers import global_step_from_engine

CHECKPOINTS_FOLDER = "checkpoints"
MAX_N_EPOCHS = 100
CHECKPOINTS_ID_TO_LOAD = None

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Select device
    # TODO: make selection possible from ArgParse
    device = torch.device("cpu")  # of cuda:0, or mps

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
        csv_path="data/Emotion-detection-from-text/training.csv",
        tokenizer=tokenizer,
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
        device=device,
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    start_epoch: int = 0

    num_training_steps = MAX_N_EPOCHS * len(train_dataloader)
    milestones_values = [
        (0, 5e-5),
        (num_training_steps, 0.0),
    ]

    # Create the lr scheduler
    lr_scheduler = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=milestones_values
    )

    model = model.to(device)

    def train_step(engine, batch):
        """Training funtion for PyTorchIgnite"""
        model.train()
        token_ids: torch.Tensor = batch[0].to(device)
        labels: torch.Tensor = batch[1].to(device)

        outputs = model(token_ids)

        assert outputs.shape[1] == 6  # Nb output word, nb labels

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    trainer = Engine(train_step)
    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    def evaluate_step(engine, batch):
        """Evaluate function for PyTorch Ignite"""
        model.eval()

        token_ids: torch.Tensor = batch[0].to(device)
        labels: torch.Tensor = batch[1].to(device)

        with torch.no_grad():
            outputs = model(token_ids)

        return {"y_pred": outputs, "y": labels}

    train_evaluator = Engine(evaluate_step)
    validation_evaluator = Engine(evaluate_step)

    Accuracy().attach(train_evaluator, "accuracy")
    Accuracy().attach(validation_evaluator, "accuracy")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_dataloader)
        metrics = train_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        print(
            f"Training Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        validation_evaluator.run(val_dataloader)
        metrics = validation_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        print(
            f"Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}"
        )

    # Early stopping
    def score_function(engine):
        val_accuracy = engine.state.metrics["accuracy"]
        return val_accuracy

    handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    validation_evaluator.add_event_handler(Events.COMPLETED, handler)

    # Save checkpoint
    to_save = {"model": model, "optimizer": optimizer, "trainer": trainer}
    checkpoint = Checkpoint(
        to_save=to_save,
        save_handler=DiskSaver(
            CHECKPOINTS_FOLDER,
            create_dir=True,
            require_empty=False,
        ),
        score_function=score_function,
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer),
    )

    validation_evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)

    # Load checkpoint
    if CHECKPOINTS_ID_TO_LOAD is not None:
        torch_checkpoint = torch.load(
            str(
                Path(
                    CHECKPOINTS_FOLDER,
                    "checkpoint_" + str(CHECKPOINTS_ID_TO_LOAD) + ".pt",
                )
            ),
            map_location=device,
        )
        Checkpoint.load_objects(
            to_load=to_save,
            checkpoint=torch_checkpoint,
        )

    trainer.run(train_dataloader, max_epochs=MAX_N_EPOCHS)
