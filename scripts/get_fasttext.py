import fasttext
import pandas as pd
import json
# from fasttext import util

labels = json.load(open('../data/labels.json'))
train = pd.read_csv('../data/descriptives/train.csv', index_col=0)
val = pd.read_csv('../data/descriptives/validation.csv', index_col=0)
test = pd.read_csv('../data/descriptives/test.csv', index_col=0)

if __name__=='__main__':
    #fasttext.util.download_model('da', if_exists='ignore') 
    ft = fasttext.load_model('../resources/cc.da.300.bin')
    # Featurize
    train['vec'] = train['text'].apply(lambda x: ft.get_sentence_vector(x.replace('\n','')))
    val['vec'] = val['text'].apply(lambda x: ft.get_sentence_vector(x.replace('\n','')))
    test['vec'] = test['text'].apply(lambda x: ft.get_sentence_vector(x.replace('\n','')))
    # With vectors
    train.to_pickle('../data/vecs/train.csv')
    val.to_pickle('../data/vecs/validation.csv')
    test.to_pickle('../data/vecs/test.csv')
   

