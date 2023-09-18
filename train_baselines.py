import pandas as pd
import json
import numpy as np
import datetime
from multiprocessing import Pool
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
from keras.preprocessing.text import Tokenizer
from pathlib import Path
from itertools import product
from sklearn.model_selection import (GridSearchCV, 
                                     RandomizedSearchCV,
                                     PredefinedSplit,
                                     StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, f1_score, 
                             precision_score, recall_score, 
                             accuracy_score, confusion_matrix)
from sklearn.utils import compute_sample_weight
import argparse


# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--run-id', type=str, 
                    default=None,
                    help='Unique run name')
parser.add_argument('--avg-type', type=str, default='micro')
parser.add_argument('--binary', type=str, default=False)
parser.add_argument('--early-stopping', type=int, default=5)
parser.add_argument('--subset', type=str, default=None)
parser.add_argument('--estimator', type=str, default='xgb')

# Set up data
vpath = Path('./data') / 'vecs'
desc = ['syllables_per_token_mean',
        'sentence_length_mean',
        'polarity',
        'proportion_unique_tokens',
        'dependency_distance_std',
        'flesch_reading_ease',
        'token_length_mean',
        'prop_adjacent_dependency_relation_mean',
        'flesch_kincaid_grade',
        'rix',
        'automated_readability_index',
        'token_length_std',
        'gunning_fog',
        'n_tokens',
        'lix',
        'coleman_liau_index',
        'sentence_length_std',
        'syllables_per_token_std',
        'dependency_distance_mean',
        'prop_adjacent_dependency_relation_std',
        'n_unique_tokens',
        'n_characters',
        'n_sentences']


# Parameters for grid search
def _make_estimator_params(estimator):
    if estimator == 'xgb':
        params = {
                  'learning_rate': [.001, .01, .1, .5],
                  'min_child_weight': [1, 3, 5, 10],
                  'gamma': [0, .5, 1., 2.],
                  'subsample': [.6, .8, 1.0],
                  'colsample_bytree': [.25, .5, .75, 1.],
                  'max_depth': [1, 3, 5],
                  'reg_alpha' :[0, .1, .1, 5.],
                  'reg_lambda': [.1, 1., 5.],
                  'n_estimators': [20, 100, 500]
                  }

    else:
        params = {'n_estimators': [1, 5, 10, 20, 50, 100, 200],
                  'max_depth': [1, 3, 5, None],
                  'criterion': ['entropy', 'gini'],
                  'oob_score': [True, False],
                  }
    #params_test = dict(zip(params.keys(),
    #                       [v[0] for v in params.values()]))
    return params


def _make_params(logpath, avg_type, binary, es, subset, estimator):
    bow_params = list(product(['tfidf'], [100, 1000, 10000]))
    bow_names =  [i[0] + '_' + str(i[1]) for i in bow_params]
    bow_which = ['bow']*len(bow_names)
    fasttext_names = ['fasttext']
    fasttext_which = ['fasttext']
    desc_names = desc + ['alldesc']
    desc_which =  ['descriptives']*(len(desc)+1)
    if subset == 'fasttext':
        which = fasttext_which
        names = fasttext_names
    elif subset == 'bow':
        which = bow_which
        names = bow_names
    elif subset == 'desc':
        which = desc_which
        names = desc_names
    else:
        which = fasttext_which + bow_which + desc_which
        names = fasttext_names + bow_names + desc_names
    return list(zip(which, names, 
                    [logpath]*len(names), 
                    [avg_type]*len(names),
                    [binary]*len(names),
                    [es]*len(names),
                    [estimator]*len(names)))


# Helper functions
def _get_X(df, feat, scaler):
    ''' Helper function to prepare data for descriptives-based model '''
    X = df[feat].values
    if len(X.shape) == 1:
        X = X.reshape(-1,1)
    X = scaler.fit_transform(X)
    return X 


def prepare_descriptives(train_metrics, val_metrics, test_metrics, feat):
    ''' Prepares descriptives data for classifier
    feat (str): name of the feature or 'all' if all features
    '''
    scaler = StandardScaler()
    if feat == 'alldesc':
        feat = train_metrics.drop(['text','labels','vec','File','ExampleID'], 
                                   axis=1).columns.tolist()
    
    # Prepare data
    train_X = _get_X(train_metrics, feat, scaler)
    val_X = _get_X(val_metrics, feat, scaler)
    test_X = _get_X(test_metrics, feat, scaler)
    return train_X, val_X, test_X


def prepare_fasttext(train_metrics, val_metrics, test_metrics):
    ''' Prepares fasttext data for classifier '''
    train_X = np.stack(train_metrics['vec'].values)
    val_X = np.stack(val_metrics['vec'].values)
    test_X = np.stack(test_metrics['vec'].values)
    return train_X, val_X, test_X


def prepare_bow(train_metrics, val_metrics, test_metrics, 
                which, model_name, logpath,
                n_words=1000, mode='tfidf'):
    ''' Prepare BoW data for classifier'''
    tknzr = Tokenizer(num_words=n_words)
    tkpath = str(logpath / f'tokenizer_{which}_{model_name}.pkl')
    print('Fitting BoW tokenizer...')
    tknzr.fit_on_texts(train_metrics['text'].tolist())
    pkl.dump(tknzr, open(tkpath, "wb"))
    print('Done!')
    train_X = tknzr.texts_to_matrix(train_metrics['text'].tolist(), 
                                    mode=mode)
    val_X = tknzr.texts_to_matrix(val_metrics['text'].tolist(), 
                                  mode=mode)
    test_X = tknzr.texts_to_matrix(test_metrics['text'].tolist(), 
                                   mode=mode)
    # Save bow tokenizers! 
    return train_X, val_X, test_X


def _prepare_data(train_metrics, val_metrics, test_metrics, 
                  which, model_name, logpath, **kwargs):
    ''' Prepare data given info on which type of baseline '''
    if which == 'bow':
        train_X, val_X, test_X = prepare_bow(train_metrics, 
                                             val_metrics, 
                                             test_metrics,
                                             which,
                                             model_name,
                                             logpath,
                                             **kwargs)
    elif which == 'fasttext':
        train_X, val_X, test_X = prepare_fasttext(train_metrics, 
                                                  val_metrics, 
                                                  test_metrics)
    elif which == 'descriptives':
        train_X, val_X, test_X = prepare_descriptives(train_metrics, 
                                                      val_metrics, 
                                                      test_metrics, 
                                                      **kwargs)
    else:
        raise ValueError('which must be one of bow, fasttext or descriptives')
    return train_X, val_X, test_X


def _score(y_true, y_pred, ldict):
    ''' Metrics to score models '''
    pred_lab = np.argmax(y_pred, axis=1)
    a = accuracy_score(y_true, pred_lab)
    f1 = f1_score(y_true, pred_lab, average='macro')
    p = precision_score(y_true, pred_lab, average='macro')
    r = recall_score(y_true, pred_lab, average='macro')
    f1_micro = f1_score(y_true, pred_lab, average='micro')
    p_micro = precision_score(y_true, pred_lab, average='micro')
    r_micro = recall_score(y_true, pred_lab, average='micro')
    cr = classification_report(y_true, pred_lab, 
                               labels=list(ldict.values()),
                               target_names=list(ldict.keys()))
    cm = confusion_matrix(y_true, pred_lab)
    return a, f1, p, r, f1_micro, p_micro, r_micro, cr, cm


def _save_predictions(preds, split, labels, exs,
                      which, model_name, logpath, ldict, 
                      estimator):
    ''' Saves predictions for each example '''
    rev_ldict = dict(zip(ldict.values(), ldict.keys()))
    pred_labels = np.argmax(preds,axis=1)
    str_pred_labels = [rev_ldict[p] for p in pred_labels]
    str_labels = [rev_ldict[p] for p in labels]
    pred_max = np.max(preds, axis=1)
    is_bin = 1 if len(ldict.keys())==2 else 0
    pred_df = pd.DataFrame(zip(exs,
                               ['_'.join(e.split('_')[:-1]) for e in exs],
                               str_labels,
                               str_pred_labels, 
                               pred_max, 
                               preds,
                               [f'{estimator}_{model_name}']*len(exs),
                               [split]*len(exs),
                               [is_bin]*len(exs),
                               ['text']*len(exs)), 
                            columns=['trial_id',
                                     'id',
                                     'label',
                                     'prediction',
                                     'confidence', 
                                     'scores',
                                     'model_name',
                                     'split',  
                                     'binary',
                                     'type'])
    
    # model name
    ofile = logpath / f'pred_{split}_{which}_{model_name}.pkl'
    pred_df.to_pickle(ofile)


def _save_scores(X, y, which, model_name, logpath, ofile, split, exs, grid, ldict, estimator):
    ''' Save scores '''
    preds = grid.best_estimator_.predict_proba(X)
    _save_predictions(preds, split, y, exs, which, model_name, logpath, ldict, estimator)
    acc, f1, precision, recall, f1_micro, precision_micro, recall_micro, cr, cm = _score(y, 
                                                                                         preds,
                                                                                         ldict)
    with open(str(ofile), 'w') as of:
        of.write('*** Confusion matrix ***\n\n')
        of.write(np.array2string(cm)) 
        of.write('\n\n\n*** Classification report *** \n\n')
        of.write(cr)
    outs = dict(zip([f'{split}_acc', 
                     f'{split}_f1_macro', 
                     f'{split}_precision_macro', 
                     f'{split}_recall_macro', 
                     f'{split}_f1_micro',
                     f'{split}_precision_micro', 
                     f'{split}_recall_micro'],
                    [acc, f1, precision, recall, 
                     f1_micro, precision_micro, recall_micro]))
    return outs
    

def fit_predict(which, 
                model_name,
                logpath,
                avg_type='micro',
                eval_on_test=True,
                binary=False,
                es=10,
                estimator='xgb',
                **kwargs):
    ''' Args:
    which (str): bow, fasttext or descriptives
    model_name (str): ID for the model
    eval_on_test (bool): whether to also evaluate on test 
        set or only validation set
    kwargs: kwargs to prepare data calls
    '''
    # Load the data, and subset if needed
    train_metrics = pd.read_pickle(str( vpath/'train.csv'))
    train_metrics = train_metrics.sample(frac=1, random_state=0)
    val_metrics = pd.read_pickle(str(vpath/'validation.csv'))
    val_metics = val_metrics.sample(frac=1, random_state=0)
    test_metrics = pd.read_pickle(str(vpath/'test.csv'))
    test_metrics = test_metrics.sample(frac=1, random_state=0)
    
    if binary is not False:
        train_metrics = train_metrics[train_metrics['ExampleID'].str.split('_').str[0]==binary]
        train_metrics['labels'] = (train_metrics['labels']>0).astype(int)
        val_metrics = val_metrics[val_metrics['ExampleID'].str.split('_').str[0]==binary]
        val_metrics['labels'] = (val_metrics['labels']>0).astype(int)        
        test_metrics = test_metrics[test_metrics['ExampleID'].str.split('_').str[0]==binary]
        test_metrics['labels'] = (test_metrics['labels']>0).astype(int)
        test_ex = test_metrics['ExampleID']
        ldict = dict(zip(['TD', binary], [0,1]))
    else:          
        ldict = json.load(open(str(Path('./data') / 'labels.json')))
    train_ex = train_metrics['ExampleID']
    val_ex = val_metrics['ExampleID']
    test_ex = test_metrics['ExampleID']
        
    # Prepare data
    train_X, val_X, test_X = _prepare_data(train_metrics,
                                           val_metrics,
                                           test_metrics,
                                           which, 
                                           model_name,
                                           logpath,
                                           **kwargs)
    train_y = train_metrics['labels'].values
    val_y = val_metrics['labels'].values
    test_y = test_metrics['labels'].values

    # Pass explicit validation set
    train_idx = np.full((train_X.shape[0],), -1, dtype=int)
    val_idx = np.full((val_X.shape[0],), 0, dtype=int)
    fold_idx = np.append(train_idx, val_idx)
    ps = PredefinedSplit(fold_idx)

    # Set up XGBoost
    sw = compute_sample_weight('balanced', y=np.concatenate([train_y, val_y],
                                                            axis=0))
    train_sw = compute_sample_weight('balanced', y=train_y)
    num_class = 4 if binary is False else None
    objective = 'multi:softmax' if binary is False else 'binary:logistic'
    eval_metric = 'mlogloss' if binary is False else 'logloss'
    scoring = f'f1_{avg_type}' if binary is False else 'f1'
    if estimator == 'xgb':
        est_class = XGBClassifier(num_class=num_class,
                                  objective=objective,
                                  eval_metric=eval_metric,
                                  n_jobs=20,
                                  use_label_encoder=False)
        grid = RandomizedSearchCV(estimator=est_class,
                                  param_distributions=_make_estimator_params(estimator),
                                  scoring=scoring, 
                                  cv=ps,
                                  verbose=2,
                                  return_train_score=True,
                                  refit=False,
                                  n_iter=1000)
    else:
        est_class = RandomForestClassifier(n_jobs=20)
        grid = GridSearchCV(estimator=est_class,
                            param_grid=_make_estimator_params(estimator),
                            scoring=scoring, 
                            cv=ps,
                            verbose=2,
                            return_train_score=True,
                            refit=False)

    grid.fit(np.concatenate([train_X,val_X], axis=0),
             np.concatenate([train_y,val_y], axis=0),
             sample_weight=sw,
             verbose=False)
    
    # Get best model and fit on training data only
    if estimator == 'xgb':
        model = XGBClassifier(**grid.best_params_, 
                              num_class=num_class,
                              objective=objective, 
                              eval_metric=eval_metric,
                              n_jobs=20, 
                              use_label_encoder=False)
        model.fit(train_X, train_y, 
                  eval_set=[(val_X, val_y)],
                  early_stopping_rounds=es,
                  sample_weight=train_sw,
                  verbose=False)
    else:
        model = RandomForestClassifier(**grid.best_params_, 
                                       n_jobs=20)
        model.fit(train_X, train_y, 
                  sample_weight=train_sw)

    grid.best_estimator_ = model

    # Make output file names
    ofile_train = logpath / f'train_{which}_{model_name}.txt'
    ofile_val = logpath / f'val_{which}_{model_name}.txt'
    ofile_test = logpath / f'test_{which}_{model_name}.txt'
    model_path = logpath / f'model_{which}_{model_name}.pkl'
    result_path = logpath / f'grid_{which}_{model_name}.csv'

    # Predict and evaluate on train, val and set
    outs = _save_scores(train_X, train_y, 
                        which, model_name, logpath, 
                        ofile_train, 'train', train_ex, 
                        grid, ldict, estimator)
    pkl.dump(grid.best_estimator_, open(str(model_path), "wb"))
    pd.DataFrame(grid.cv_results_).to_csv(str(result_path))
    val_outs = _save_scores(val_X, val_y, 
                            which, model_name, logpath, 
                            ofile_val, 'val', val_ex, 
                            grid, ldict, estimator)
    outs.update(val_outs)
    
    # Predict and evaluate on test set
    if eval_on_test:
        test_outs = _save_scores(test_X, test_y, 
                                 which, model_name, logpath,
                                 ofile_test, 'test', test_ex, 
                                 grid, ldict, estimator)
        outs.update(test_outs)
    
    # Print results
    print(model_name)
    outs.update({'model': f'{estimator}_{model_name}'})
    return outs


def run_baseline(p):
    ''' Single-model function 
    Args:
    p (tuple): (model_type, model_id), schema is loosely defined
        in make_params
    '''
    print(f'*** Running {p[0]}, {p[1]} ***')
    if p[0] == 'fasttext':
        outs = fit_predict(p[0], p[1], p[2], p[3], 
                           es=p[5],
                           binary=p[4], 
                           estimator=p[6])
    elif p[0] == 'bow':
        print(int(p[1].split('_')[1]))
        outs = fit_predict(p[0], p[1], p[2], p[3],
                           es=p[5],
                           n_words=int(p[1].split('_')[1]), 
                           mode=p[1].split('_')[0],
                           binary=p[4],
                           estimator=p[6])
    elif p[0] == 'descriptives':
        outs = fit_predict(p[0], p[1], p[2], p[3], 
                           es=p[5], 
                           feat=p[1],
                           binary=p[4],
                           estimator=p[6])
    return outs
    

if __name__=='__main__':
    args = parser.parse_args()
    if args.run_id is None:
        rid = str(datetime.datetime.now())
    else:
        if args.binary is False:
            rid = args.run_id
        else:
            rid = f'{args.run_id}_binary_{args.binary}'
    logpath = Path('logs') / 'baselines' / rid
    logpath.mkdir(parents=True, exist_ok=True)
    pm = _make_params(logpath, args.avg_type, 
                      args.binary, args.early_stopping,
                      args.subset, args.estimator)
    results = [run_baseline(p) for p in pm]
    with open(str(logpath)+'.json', 'w') as of:
        of.write(json.dumps(results))
    

    
