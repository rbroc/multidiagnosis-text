import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from classifier import BaselineClassifier

import itertools

from scripts.util import create_argparser

import wandb

from typing import Tuple

from dataset import FasttextDataset, BowDataset

from keras.preprocessing.text import Tokenizer

from pathlib import Path

def make_params():
    models = ['fasttext_fasttext', 
              'bow_tfidf_100', 
              'bow_tfidf_1000', 
              'bow_tfid_10000']
    diagnosis = [None, 'ASD', 'SCHZ', 'DEPR']
    combs = list(itertools.product(models, diagnosis))
    params = [{'model': c[0], 
               'diagnosis': c[1], 
               'mtype': 'binary' if c[1] is not None else 'multi',
               'input_size': 300 if 'fasttext' in c[0] else int(c[0].split('_')[-1]),
               'num_classes': 2 if c[1] is not None else 4} 
              for c in combs]
    return params


def create_dataloaders(config, training_dataset, val_dataset) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(training_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    return train_loader, val_loader

def create_trainer(config) -> pl.Trainer:
    wandb_logger = WandbLogger()
    callbacks = [ModelCheckpoint(
        monitor="val_loss", mode="min")]
    if config.patience:
        early_stopping = EarlyStopping("val_loss", patience=config.patience)
        callbacks.append(early_stopping)
    
    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=config.log_step,
        val_check_interval=config.val_check_interval,
        callbacks=callbacks,
        gpus=config.gpus,
        profiler=config.profiler,
        max_epochs=config.max_epochs,
        default_root_dir=config.default_root_dir,
        weights_save_path=os.path.join(config.default_root_dir, config.logs_dir, config.run_name),
        precision=config.precision,
        auto_lr_find=config.auto_lr_find,
    )
    return trainer


if __name__ == "__main__": 
    yml_path = os.path.join(
        os.path.dirname(__file__), "config", "baseline_config.yaml"
    )
    parser = create_argparser(yml_path)
    arguments = parser.parse_args()
    params = make_params()
    
    for param in params:
        for learning_rate in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
            if 'fasttext' in param['model']:
                train_dataset = FasttextDataset('train', param['mtype'], param['diagnosis'])
                val_dataset = FasttextDataset('validation', param['mtype'], param['diagnosis'])
            elif 'bow' in param['model']:
                train_dataset = BowDataset('train', param['input_size'], param['mtype'], param['diagnosis'])
                val_dataset = BowDataset('validation', param['input_size'], param['mtype'], param['diagnosis'])
                
            print(f"[INFO] Starting {param['model'], param['mtype'], param['diagnosis']}...")
            # setup wandb config
            run = wandb.init(config=arguments,
                project="huggingface",
                dir=arguments.default_root_dir,
                allow_val_change=True,
                reinit=True)

            config = run.config
            if param['mtype'] != 'binary':
                run.name = f"baseline_{param['model']}_{learning_rate}"
            else:
                run.name = f"baseline_{param['model']}_binary_{param['diagnosis']}_{learning_rate}"
            config.run_name = run.name
            run.log({"feat_set" : param['model']})

            # Create dataloaders, model, and trainer
            train_loader, val_loader = create_dataloaders(config, train_dataset, val_dataset)
            model = BaselineClassifier(
                num_classes=param['num_classes'],
                input_size=param['input_size'], 
                learning_rate=learning_rate, # config.learning_rate,
                train_loader=train_loader,
                val_loader=val_loader,
                weights=train_dataset.weights)
            trainer = create_trainer(config)

            if config.auto_lr_find:
                lr_finder = trainer.tuner.lr_find(model)
                config.update({"learning_rate": lr_finder.suggestion()}, allow_val_change=True)
                fig = lr_finder.plot(suggest=True)
                run.log({"lr_finder.plot": fig})
                run.log({"found_lr" : lr_finder.suggestion()})

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)        
            
            run.finish()