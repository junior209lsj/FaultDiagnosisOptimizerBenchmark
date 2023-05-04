import fdob
import torch
import os
import numpy as np
import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import json
import copy
from torch.utils.data import DataLoader

from typing import Dict, Any

def train(train_loader: DataLoader,
          val_loader: DataLoader,
          model: torch.nn.Module,
          model_kwargs: Dict,
          opt: torch.optim.Optimizer,
          opt_kwargs: Dict,
          loss: Any,
          loss_kwargs: Dict,
          n_epochs: int,
          random_seed: int,
          n_gpu: int,
          result_dir: str):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = train_loader.batch_size

    n_steps_d = len(train_loader.dataset) // (batch_size * 1)
    exp_name = result_dir

    if not os.path.isdir(f"{exp_name}"):
        os.makedirs(f"{exp_name}")
    
    # Fix seed
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    if model_kwargs is not None:
        model_train = model(**model_kwargs)
    else:
        model_train = model()
    if opt_kwargs is not None:
        optim = opt(model_train.parameters(), **opt_kwargs)
    else:
        optim = opt(model_train.parameters())

    if loss_kwargs is not None:
        loss_fn = loss(**loss_kwargs)
    else:
        loss_fn = loss()
    
    logger = CSVLogger(f"{result_dir}/training", name="log")
    training_module = fdob.PlModule(model_train, optim, loss_fn, True)
    callback = ModelCheckpoint(monitor="val_loss",
                                dirpath=f"{result_dir}/best_model",
                                filename=f"model",
                                save_top_k=1,
                                mode="min")
    trainer = pl.Trainer(gpus=[n_gpu],
                        max_epochs=n_epochs,
                        val_check_interval=n_steps_d,
                        default_root_dir=f"{result_dir}",
                        callbacks=[callback],
                        logger=logger)
    trainer.fit(model=training_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    training_module.load_from_checkpoint(f"{result_dir}/best_model/model.ckpt",
                                        model=model_train, optimizer=optim,
                                        loss_fn=loss_fn)
    result = trainer.test(model=training_module, dataloaders=val_loader)


    optimizer_info = copy.deepcopy(opt_kwargs)
    optimizer_info["name"] = opt.__name__

    with open(f"{result_dir}/optim_info.json", "w") as f: 
        json.dump(optimizer_info, f, indent=4)

    return result


def test(test_loader: DataLoader,
          model: torch.nn.Module,
          model_kwargs: Dict,
          opt: torch.optim.Optimizer,
          opt_kwargs: Dict,
          loss: Any,
          loss_kwargs: Dict,
          n_epochs: int,
          random_seed: int,
          n_gpu: int,
          result_dir: str,
          test_name: str):
    
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = test_loader.batch_size

    n_steps_d = len(test_loader.dataset) // (batch_size * 1)
    exp_name = result_dir

    if not os.path.isdir(f"{exp_name}"):
        os.makedirs(f"{exp_name}")
    
    # Fix model seed
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    if model_kwargs is not None:
        model_train = model(**model_kwargs)
    else:
        model_train = model()
    if opt_kwargs is not None:
        optim = opt(model_train.parameters(), **opt_kwargs)
    else:
        optim = opt(model_train.parameters())

    if loss_kwargs is not None:
        loss_fn = loss(**loss_kwargs)
    else:
        loss_fn = loss()
    
    logger = CSVLogger(f"{result_dir}/testing/{test_name}", name="log")
    training_module = fdob.PlModule(model_train, optim, loss_fn, True)
    callback = ModelCheckpoint(monitor="val_loss",
                                dirpath=f"{result_dir}/best_model",
                                filename=f"model",
                                save_top_k=1,
                                mode="min")
    trainer = pl.Trainer(gpus=[n_gpu],
                        max_epochs=n_epochs,
                        val_check_interval=n_steps_d,
                        default_root_dir=f"{result_dir}",
                        callbacks=[callback],
                        logger=logger)
    training_module.load_from_checkpoint(f"{result_dir}/best_model/model.ckpt",
                                        model=model_train, optimizer=optim,
                                        loss_fn=loss_fn)
    result = trainer.test(model=training_module, dataloaders=test_loader)

    return result