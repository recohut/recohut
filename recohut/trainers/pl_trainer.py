# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/trainers/trainers.pl_trainer.ipynb (unless otherwise specified).

__all__ = ['pl_trainer']

# Cell
from typing import Any, Iterable, List, Optional, Tuple, Union, Callable
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Cell
def pl_trainer(model, datamodule, max_epochs=10, val_epoch=5, gpus=None, log_dir=None,
               model_dir=None, monitor='val_loss', mode='min', *args, **kwargs):
    log_dir = log_dir if log_dir is not None else os.getcwd()
    model_dir = model_dir if model_dir is not None else os.getcwd()

    logger = TensorBoardLogger(save_dir=log_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=model_dir,
        filename="recommender",
    )

    trainer = Trainer(
    max_epochs=max_epochs,
    logger=logger,
    check_val_every_n_epoch=val_epoch,
    callbacks=[checkpoint_callback],
    num_sanity_val_steps=0,
    gradient_clip_val=1,
    gradient_clip_algorithm="norm",
    gpus=gpus
    )

    trainer.fit(model, datamodule=datamodule)
    test_result = trainer.test(model, datamodule=datamodule)
    return test_result