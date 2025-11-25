import logging
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cs336_basics.loss import cross_entropy
from cs336_basics.modules import RotaryPositionalEmbedding, Transformer
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import (
    cosine_learning_rate_schedule,
    data_loading,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def clear_checkpoints(path: str, num_max: int):
    checkpoints = [
        chkpt
        for chkpt in os.listdir(path)
        if chkpt.endswith(".pth") and chkpt != "best_model.pth"
    ]
    if len(checkpoints) > num_max:
        logging.info(
            f"the number of saved checkpoint is larger than {num_max}, "
            "{checkpoints[0]} is deleted"
        )
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        os.remove(os.path.join(path, checkpoints[0]))


def train(config: dict, resume=False):
    # common config
    record_path = config["record_path"]
    device = config["device"]
    # data config
    train_data_path = config["train_data_path"]
    val_data_path = config["val_data_path"]
    # model config
    vocab_size = config["vocab_size"]
    context_length = config["context_length"]
    num_layers = config["num_layers"]
    d_model = config["d_model"]
    num_heads = config["num_heads"]
    d_ff = config["d_ff"]
    model_eps = config["model_eps"]
    # Rope config
    theta = config["theta"]
    # optimizer config
    learning_rate = config["learning_rate"]
    betas = config["betas"]
    opt_eps = config["opt_eps"]
    weight_decay = config["weight_decay"]
    # training config
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_total_processes = config["num_total_processes"]
    log_step = config["log_step"]
    early_stop = config["early_stop"]
    checkpoint_max_num = config["checkpoint_max_num"]
    # gradient clipping
    grad_max_norm = config["max_norm"]
    grad_eps = config["grad_eps"]
    # learning rate schedule
    lr_max = config["lr_max"]
    lr_min = config["lr_min"]
    warm_step = config["warm_step"]

    writer = SummaryWriter(record_path)

    train_data = np.load(train_data_path, mmap_mode="r")
    val_data = np.load(val_data_path, mmap_mode="r")

    model = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        eps=model_eps,
        device=torch.device(device),
    )
    rope = RotaryPositionalEmbedding(
        theta=theta,
        dim=d_model // num_heads,
        max_seq_len=context_length,
        device=torch.device(device),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=opt_eps,
        weight_decay=weight_decay,
    )

    writer.add_graph(
        model,
        input_to_model=torch.randint(
            0, vocab_size, (batch_size, context_length)
        ),
    )

    step_each_epoch = num_total_processes // (
        num_epochs * batch_size * context_length
    )
    val_step = len(val_data) // batch_size
    loss_min = float("inf")
    early_stop_count = 0
    start_epoch = 1

    if resume:
        start_epoch = load_checkpoint(
            src=record_path, model=model, optimizer=optimizer
        )

    for epoch in range(start_epoch, num_epochs + 1):
        logging.info(f"-- training epoch {epoch} ".ljust(80, "-"))

        for step in range(step_each_epoch):
            data, target = data_loading(
                train_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )

            pred = model(data, rope)

            loss = cross_entropy(pred, target)

            loss.backward()
            gradient_clipping(
                model.parameters(), max_norm=grad_max_norm, eps=grad_eps
            )

            for g in optimizer.param_groups:
                g["lr"] = cosine_learning_rate_schedule(
                    (epoch - 1) * step_each_epoch + step,
                    lr_max,
                    lr_min,
                    t_w=warm_step,
                    t_c=step_each_epoch * num_epochs - warm_step,
                )
            optimizer.step()
            optimizer.zero_grad()

            if step % log_step == 0:
                logging.info(
                    f"epoch: {epoch}, step: {step}, loss: {loss.detach().item()}"  # noqa
                )
                writer.add_scalar(
                    tag="train_loss",
                    scalar_value=loss.detach().item(),
                    global_step=(epoch - 1) * step_each_epoch + step,
                )

        logging.info("saving checkpoint...")
        save_checkpoint(
            model, optimizer, epoch, f"{record_path}/epoch_{epoch}.pth"
        )
        clear_checkpoints(record_path, checkpoint_max_num)

        # validation
        val_loss = 0.0
        with torch.no_grad():
            for step in range(val_step):
                data, target = data_loading(
                    val_data, batch_size, context_length, device
                )
                loss = cross_entropy(model(data), target).detach().item()
                writer.add_scalar(
                    tag="validation_loss",
                    scalar_value=loss,
                    global_step=(epoch - 1) * val_step + step,
                )
                val_loss += loss
        val_loss /= val_step
        logging.info(f"epoch: {epoch}, validation loss: {val_loss}")

        if val_loss < loss_min:
            logging.info("saving current best model...")
            save_checkpoint(
                model, optimizer, epoch, f"{record_path}/best_model.pth"
            )
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop:
            logging.info(
                f"model has not been optimized for {early_stop_count} epoches, "
                "stop training"
            )
            break

    logging.info("training completed")
