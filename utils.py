import pandas as pd
from dataset import TextDataset
from transformers import Trainer, AutoTokenizer
from transformers.trainer_utils import speed_metrics 
from torch.utils.data import Dataset
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import time
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import math
import json

ldict = json.load(open(Path('data') / 'labels.json')) 

def make_dataset(tknzr, split='train', binary=None):
    ''' Make dataset from transcripts and train / val ids 
    Args:
        tknzr: pretrained tokenizer name
    '''
    DPATH = Path('data') / 'processed'
    data = pd.read_csv(DPATH/f'{split}.csv') # edit and shuffle?
    if binary:
        data = data[data['Group']==binary]
    txt, lab, ids = zip(*data[['Transcript', 'Diagnosis', 'ExampleID']].to_records(index=False))
    tokenizer = AutoTokenizer.from_pretrained(tknzr)
    enc = tokenizer(list(txt), truncation=True, padding=True)
    label_mapping = ldict if binary is None else {'TD':0, binary:1}
    dataset = TextDataset(enc, lab, ids, label_mapping)
    return dataset


def make_trainer(weights, **kwargs):
    ''' Create trainer with custom loss and metrics loop '''
    
    class TextTrainer(Trainer):

        def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
        ) -> Dict[str, float]:
            """
            Run evaluation and returns metrics.
            The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
            (pass it to the init `compute_metrics` argument).
            You can also subclass and override this method to inject custom behavior.
            Args:
                eval_dataset (`Dataset`, *optional*):
                    Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                    accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                    method.
                ignore_keys (`Lst[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.
                metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                    An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                    "eval_bleu" if the prefix is "eval" (default)
            Returns:
                A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
                dictionary also contains the epoch number which comes from the training state.
            """
            # memory metrics - must set up as early as possible
            self._memory_tracker.start()

            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            start_time = time.time()

            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

            self.log(output.metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
            self._memory_tracker.stop_and_update_metrics(output.metrics)

            # Print full classification report
            #### HOTFIX: doesn't work for binary
            # crf = str(Path(self.args.output_dir) / f'report_epoch-{int(self.state.epoch)}.txt')
            # predictions = np.argmax(output.predictions, axis=-1)
            # cm = confusion_matrix(output.label_ids, predictions)
            # cm = pd.DataFrame(cm, columns=ldict.keys())
            # cm.index = ldict.keys()
            # cr = classification_report(output.label_ids, 
            #                            predictions, 
            #                            target_names=list(ldict.keys()))
            # with open(str(crf), 'a') as of:
            #     of.write('*** Confusion matrix ***\n\n')
            #     of.write(cm.to_string())
            #     of.write('\n\n\n*** Classification report *** \n\n')
            #     of.write(cr)
            # print(cr)
            return output.metrics

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get('logits')
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return TextTrainer(**kwargs)


