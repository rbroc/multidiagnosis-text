import pandas as pd
import glob
import numpy as np
from sklearn.metrics import f1_score
from pathlib import Path
import shutil

fs = glob.glob('transformers/json_outs/*')
classes = ['multiclass', 'ASD', 'DEPR', 'SCHZ']
models = list(set([f.split('/')[2].split('_')[1] for f in fs]))

# Find best files
best_models = []
for c in classes:
    for m in models:
        files = [f for f in fs if c in f and m in f]
        max_f1 = 0
        best_model = None
        for f in files:
            if '_val.json' in f:
                df = pd.read_json(f, lines=True)
                f1 = f1_score(df.label, df.prediction, labels=df.label.unique(), average='macro')
                f1 = np.round(f1, 4)
                if f1 > max_f1:
                    max_f1 = f1
                    best_model = '_'.join(f.split('/')[-1].split('_')[:-1])
        print(c, m, f1)
        best_models.append(best_model)

# Move remaining files
archive_path = Path('archive') 
archive_path.mkdir(parents=True, exist_ok=True)
json_outpath = archive_path / 'json_outs'
json_outpath.mkdir(parents=True, exist_ok=True)
model_outpath = archive_path / 'models'
model_outpath.mkdir(parents=True, exist_ok=True)
for f in fs:
    file_id = f.split('/')[-1]
    model_id = '_'.join(file_id.split('_')[:-1])
    if model_id not in best_models:
        shutil.move(f, str(json_outpath / file_id))

# Check 
remaining_files = glob.glob('transformers/json_outs/*')
assert len(remaining_files) == len(classes) * len(models) * 3

# Move weights and biases folders
folders = [fol for fol in glob.glob('transformers/*') if 'json_outs' not in fol]
for fol in folders:
    folder_id = fol.split('/')[1]
    if folder_id not in best_models:
        shutil.rmtree(fol, str(archive_path / folder_id))

# Check 
remaining_folders = glob.glob('transformers/*')
assert len(remaining_folders) == len(classes) * len(models) + 1

# Move models
model_folders = glob.glob('../models/*')
for m in model_folders:
    model_id = m.split('/')[2]
    if model_id not in best_models:
        shutil.rmtree(m, str(model_outpath / model_id))

# Check 
remaining_models = glob.glob('../models/*')
assert len(remaining_models) == len(classes) * len(models)
