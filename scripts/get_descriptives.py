import pandas as pd
import textdescriptives as td
import json
import dacy
from dacy.sentiment import add_berttone_polarity

# Load labels, spacy pipeline and datasets
labels = json.load(open('../data/labels.json'))


if __name__=='__main__':

    # Set up spacy
    nlp = dacy.load('medium')
    nlp.add_pipe("textdescriptives")
    nlp = add_berttone_polarity(nlp, force_extension=True)
    pdict = dict(zip(['negative', 'neutral', 'positive'],
                    [-1, 0, 1]))

    # Read files
    train = pd.read_csv('../data/processed/train.csv')
    val = pd.read_csv('../data/processed/validation.csv')
    test = pd.read_csv('../data/processed/test.csv')

    # Compute metrics and add labels
    print('Computing metrics on training set...')
    train_d = list(nlp.pipe(train['Transcript']))
    print('Computing metrics on validation set...')
    val_d = list(nlp.pipe(val['Transcript']))
    print('Computing metrics on test set...')
    test_d = list(nlp.pipe(test['Transcript']))

    print('Descriptives -> dataframe')
    train_metrics = pd.concat([td.extract_df(d) for d in train_d],
                            ignore_index=True)
    val_metrics = pd.concat([td.extract_df(d) for d in val_d],
                            ignore_index=True)
    test_metrics = pd.concat([td.extract_df(d) for d in test_d],
                             ignore_index=True)

    train_metrics['polarity'] = [pdict[d._.polarity] for d in train_d]
    val_metrics['polarity'] = [pdict[d._.polarity] for d in val_d]
    test_metrics['polarity'] = [pdict[d._.polarity] for d in test_d]

    train_metrics['labels'] = train['Diagnosis'].apply(lambda x: labels[x])
    val_metrics['labels'] = val['Diagnosis'].apply(lambda x: labels[x])
    test_metrics['labels'] = test['Diagnosis'].apply(lambda x: labels[x])

    assert train_metrics.shape[0] == train.shape[0]
    assert val_metrics.shape[0] == val.shape[0]
    assert test_metrics.shape[0] == test.shape[0]
    
    train_metrics['ExampleID'] = train['ExampleID']
    val_metrics['ExampleID'] = val['ExampleID']
    test_metrics['ExampleID'] = test['ExampleID']
    train_metrics['File'] = train['File']
    val_metrics['File'] = val['File']
    test_metrics['File'] = test['File']

    # Drop columns with NAs
    train_metrics.dropna(axis=1, inplace=True)
    val_metrics = val_metrics[train_metrics.columns]
    test_metrics = test_metrics[train_metrics.columns]

    # Store annotations
    print('Saving...')
    train_metrics.to_csv('../data/descriptives/train.csv')
    val_metrics.to_csv('../data/descriptives/validation.csv')
    test_metrics.to_csv('../data/descriptives/test.csv')
