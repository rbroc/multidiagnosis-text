from pathlib import Path
from transformers import (TrainingArguments,
                          AutoModelForSequenceClassification,
                          EarlyStoppingCallback)
from utils import make_dataset, make_trainer
import numpy as np
import pandas as pd
import json
import argparse
from datasets import load_metric
import torch
from sklearn.utils import compute_class_weight
from transformers import EarlyStoppingCallback

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--model-id', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--train-examples-per-device', type=int, default=8)
parser.add_argument('--eval-examples-per-device', type=int, default=8)
parser.add_argument('--learning-rate', type=float, default=5e-5)
parser.add_argument('--warmup-steps', type=int, default=500)
parser.add_argument('--weight-decay', type=float, default=0.001)
parser.add_argument('--logging-steps', type=int, default=100)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--early-stopping-patience', type=int, default=10)
parser.add_argument('--binary', type=str, default=None)
parser.add_argument('--freeze-layers', type=int, default=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = json.load(open(Path('data') / 'labels.json')) 
PRED_COLUMNS = ['trial_id', 'id', 'label', 
                'prediction', 'confidence', 'scores', 
                'model_name', 'split', 'binary', 'type']


def make_lab_dicts(binary):
    if binary is not None:
        ldict = {'TD': LABELS['TD'],
                 binary: 1}
    else:
        ldict = LABELS
    rev_ldict = dict(zip(ldict.values(), ldict.keys()))
    return ldict, rev_ldict


def compute_metrics(pred):
    prec, rec, f1, accuracy = (load_metric(m) 
                            for m in ('precision', 
                                      'recall', 
                                      'f1', 
                                      'accuracy'))
    logits, labels = pred
    pclass = np.argmax(logits, axis=-1)
    precision = prec.compute(predictions=pclass, references=labels, average='macro')["precision"]
    recall = rec.compute(predictions=pclass, references=labels, average='macro')["recall"]
    f1_score = f1.compute(predictions=pclass, references=labels, average='macro')["f1"]
    accuracy = accuracy.compute(predictions=pclass, references=labels)["accuracy"]                      
    return {"precision": precision, 
            "recall": recall, 
            "f1": f1_score, 
            "accuracy": accuracy}


# Training module
def _make_trainer(model_id,
                  checkpoint, 
                  train_dataset, val_dataset,
                  epochs, 
                  train_examples_per_device, 
                  eval_examples_per_device,
                  learning_rate,
                  warmup_steps, 
                  weight_decay,
                  logging_steps,
                  gradient_accumulation_steps,
                  early_stopping_patience,
                  freeze_layers,
                  binary):
    ''' Train model 
    Args:
        model_id: unique model id
        checkpoint: path to model checkpoint or pretrained from model hub
        train_dataset: training dataset
        val_dataset: validation dataset
        epochs: training epochs
        train_examples_per_device: examples per device at training
        eval_examples_per_device: examples per device at eval
        warmup_steps: nr optimizer warmup steps
        weight_decay: weight decay for Adam optimizer
        logging_steps: how often we want to log metrics
    '''

    # Make directories
    fstr = 'freeze' if freeze_layers == 1 else 'nofreeze'
    bstr = f'batch-{train_examples_per_device}'
    if binary is not None:
        mid = f'{binary}_'
    else:
        mid = 'multiclass_'
    mid += f'{model_id}_lr-{learning_rate}_wdecay-{weight_decay}_wsteps-{warmup_steps}_{fstr}'
    mid += f'_{bstr}'
    logpath = Path('logs') / 'transformers' / mid 
    respath = Path('models') / mid
    logpath.mkdir(exist_ok=True, parents=True)
    respath.mkdir(exist_ok=True, parents=True)

    # Label mapping
    num_labels = 4 if binary is None else 2
    ldict, rev_ldict = make_lab_dicts(binary)
    
    # Set up trainer
    training_args = TrainingArguments(
        output_dir=logpath,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_examples_per_device,
        per_device_eval_batch_size=eval_examples_per_device,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=str(logpath),
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_strategy='steps',
        logging_steps=logging_steps,
        save_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        run_name=mid,
        load_best_model_at_end=True,
        #fp16=True,
        save_total_limit=1
    )

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                               num_labels=num_labels, 
                                                               id2label=rev_ldict,
                                                               label2id=ldict,
                                                               problem_type='single_label_classification').to(device)
    if freeze_layers==1:
        modules = [model.base_model.embeddings]
        try:
            modules += model.base_model.encoder.layer
        except:
            modules += model.base_model.transformer.layer
        for module in modules:
            for p in module.parameters():
                p.requires_grad = False
    
    # Initialize trainer
    weights = torch.tensor(compute_class_weight('balanced', 
                                               list(ldict.keys()), 
                                               train_dataset.labels),
                           dtype=torch.float).to(device)
    trainer = make_trainer(weights, 
                           model=model,
                           args=training_args,
                           train_dataset=train_dataset,
                           eval_dataset=val_dataset,
                           compute_metrics=compute_metrics,
                           callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)])

    return trainer, respath, mid


def evaluate(trainer, dataset, split, binary, model_id):
    _, rev_ldict = make_lab_dicts(binary)
    # Get predictions
    outs = trainer.predict(test_dataset=dataset)
    top = torch.max(torch.tensor(outs.predictions), axis=1)
    # Extract metrics
    labels = [rev_ldict[l] for l in outs.label_ids]
    predictions = [rev_ldict[p] for p in top.indices.numpy()]
    confidences = top.values.numpy()
    model_names = [model_id] * len(predictions)
    trial_ids = dataset.ids
    ids = ['_'.join(i.split('_')[:-1]) for i in trial_ids]
    # Hyperparameters
    binary = 0 if args.binary is None else 1
    binaries = [binary] * len(predictions)
    types = ['text'] * len(predictions)
    splits = [split] * len(predictions)
    scores = outs.predictions
    # Make dataframe
    pdf = pd.DataFrame(zip(trial_ids, ids, labels,
                           predictions, confidences, scores,
                           model_names, splits, binaries, types),
                       columns=PRED_COLUMNS)
    # Make output path
    output_path = Path('logs') / 'transformers' / 'json_outs'
    output_path.mkdir(parents=True, exist_ok=True)
    outfile = str(output_path / f'{mid}_{split}.jsonl')
    # Save
    pdf.to_json(outfile, orient='records', lines=True) 


# Execute
if __name__=='__main__':
    args = parser.parse_args()
    train_ds, val_ds, test_ds = (make_dataset(args.checkpoint, s, args.binary) 
                                 for s in ['train', 'validation', 'test'])
    trainer, respath, mid = _make_trainer(args.model_id,
                                          args.checkpoint,
                                          train_ds, 
                                          val_ds, 
                                          args.epochs, 
                                          args.train_examples_per_device, 
                                          args.eval_examples_per_device,
                                          args.learning_rate,
                                          args.warmup_steps,
                                          args.weight_decay,
                                          args.logging_steps,
                                          args.gradient_accumulation_steps,
                                          args.early_stopping_patience,
                                          args.freeze_layers,
                                          args.binary)
    trainer.train()
    trainer.save_model(str(respath))

    # Predict on both validation and test
    for ds, spl in zip([train_ds, val_ds, test_ds],
                        ['train', 'val', 'test']):
        evaluate(trainer, ds, spl, args.binary, mid)


    