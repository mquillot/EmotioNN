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
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Checkpoint
from ignite.handlers import DiskSaver
from ignite.handlers import global_step_from_engine

from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
import torch.nn.functional as F

CHECKPOINTS_FOLDER = "checkpoints"
MAX_N_EPOCHS = 1000
CHECKPOINTS_ID_TO_LOAD = None
RUN_ID = "run_02"
LOG_INTERVAL = 100

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Select device
    # TODO: make selection possible from ArgParse
    device = torch.device("cpu")  # or cuda:0, or mps

    # Create the tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    data = pd.read_csv("data/Emotion-detection-from-text/training.csv")

    tokenizer.train_from_iterator(
        iterator=[row["text"] for _, row in data.iterrows()],
        trainer=BpeTrainer(),
    )

    def padding_batch(batch):
        max_len = max([len(e[0]) for e in batch])
        new_elements = []
        for element in batch:
            new_elements.append(
                F.pad(
                    element[0],
                    pad=(0, max_len - len(element[0])),
                    mode="constant",
                    value=0,
                )
            )
        return (torch.stack(new_elements), torch.tensor([e[1] for e in batch]))

    # Load training set and dataloader
    train_set = TextualEmotionDetectionDataset(
        csv_path="data/Emotion-detection-from-text/training.csv",
        tokenizer=tokenizer,
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        collate_fn=padding_batch,
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

        return {"y_pred": outputs, "y": labels, "criterion_kwargs": {}}

    train_evaluator = Engine(evaluate_step)
    validation_evaluator = Engine(evaluate_step)

    Accuracy().attach(train_evaluator, "accuracy")
    Accuracy().attach(validation_evaluator, "accuracy")
    Loss(loss_fn).attach(train_evaluator, "loss")
    Loss(loss_fn).attach(validation_evaluator, "loss")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_dataloader)
        metrics = train_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]

        logging.info(
            f"Training Results - Epoch: {engine.state.epoch}  "
            f"Avg loss {avg_loss:.3f}, Avg accuracy: {avg_accuracy:.3f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        validation_evaluator.run(val_dataloader)
        metrics = validation_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        logging.info(
            f"Validation Results - Epoch: {engine.state.epoch}  "
            f"Avg loss {avg_loss:.3f}, Avg accuracy: {avg_accuracy:.3f}"
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
            str(Path(CHECKPOINTS_FOLDER, RUN_ID)),
            create_dir=True,
            require_empty=False,
        ),
        score_function=score_function,
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer),
    )

    validation_evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)

    # # Load checkpoint
    if CHECKPOINTS_ID_TO_LOAD is not None:
        torch_checkpoint = torch.load(
            str(
                Path(
                    CHECKPOINTS_FOLDER,
                    RUN_ID,
                    "checkpoint_" + str(CHECKPOINTS_ID_TO_LOAD) + ".pt",
                )
            ),
            map_location=device,
        )
        Checkpoint.load_objects(
            to_load=to_save,
            checkpoint=torch_checkpoint,
        )

    # Create a tensorboard logger
    with TensorboardLogger(log_dir=f"experiments/{RUN_ID}") as tb_logger:
        # Attach the logger to the trainer to log training loss at each iteration
        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED(every=LOG_INTERVAL),
            tag="training",
            output_transform=lambda loss: {"loss_iteration": loss},
        )

        # Attach handler for plotting both evaluators' metrics after every epoch completes
        for tag, evaluator in [
            ("training", train_evaluator),
            ("validation", validation_evaluator),
        ]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names="all",
                global_step_transform=global_step_from_engine(trainer),
            )

    trainer.run(train_dataloader, max_epochs=MAX_N_EPOCHS)
