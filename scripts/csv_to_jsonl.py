import pandas as pd
import json
import os
import glob

def convert_files():
    folders = [f for f in os.listdir('../logs/baselines') 
               if not f.endswith('.json') and '_5es' in f]
    for f in folders:
        # Get all files
        jfolder = f'../logs/baselines/json_outs'
        os.makedirs(jfolder, exist_ok=True)
        train_pred_files = glob.glob(f'../logs/baselines/{f}/pred_train_*')
        val_pred_files = glob.glob(f'../logs/baselines/{f}/pred_val_*')
        test_pred_files = glob.glob(f'../logs/baselines/{f}/pred_test_*')
        for t,s in zip([train_pred_files, val_pred_files, test_pred_files],
                       ['train', 'val', 'test']):
            for f in t:
                fold = f.split('/')[3]
                target_class = fold.split('_')[-1] if len(fold.split('_'))==5 else 'multiclass'
                model_name = '-'.join(f.split('/')[-1].split('.')[0].split('_')[3:])
                outfile = f'{jfolder}/baseline_{target_class}_{model_name}_{s}.jsonl'
                # Read, convert and dump
                df = pd.read_pickle(f)
                df['model_name'] = df['model_name'].apply(lambda x: '-'.join(x.split('_')[1:]))
                df.to_json(outfile, orient='records', lines=True) 
                
if __name__=='__main__':
    convert_files()                                        