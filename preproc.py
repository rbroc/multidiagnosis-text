import pandas as pd
from pathlib import Path
import seaborn as sns

oedict = {'Bloede': 'Bløde',
          'Boern': 'Børn',
          'Doedens': 'Dødens',
          'Efterfoelger': 'Efterfølger',
          'Foerst': 'Først',
          'Forfoelgelse': 'Forfølgelse',
          'Forsoeger': 'Forsøger',
          'Goer': 'Gør',
          'Koerer': 'Kører',
          'Koerte': 'Kørte',
          'Proever': 'Prøver',
          'Soerger': 'Sørger',
          'Stoeder': 'Støder',
          'Stoedte': 'Stødte',
          'Vægtloest': 'Vægtløst',
          'afsloerer': 'afslører',
          'behoever': 'behøver',
          'besoeg': 'besøg',
          'bloedte': 'blødte',
          'boern': 'børn',
          'doede': 'døde',
          'doedt.': 'dødt',
          'doer': 'dør',
          'droener': 'drøner',
          'efterfoelg': 'efterfølg',
          'etapeloeb': 'etapeløb',
          'floej': 'fløj',
          'foele': 'føle',
          'foelge': 'følge',
          'foelte': 'følte',
          'foer': 'før',
          'foeste': 'første',
          'forfoelge': 'forfølge',
          'forloeb': 'forløb',
          'formålsloese': 'formålsløse',
          'forsoeg': 'forsøg',
          'goer': 'gør',
          'groed': 'grød',
          'hjoerne': 'hjørne',
          'hjælpeloes': 'hjælpeløs',
          'hoejre': 'højre',
          'hoer': 'hør',
          'humoer': 'humør',
          'indroemme': 'indrømme',
          'kaploeb': 'kapløb',
          'koere': 'køre',
          'koerte': 'kørte',
          'ligefoer': 'ligefør',
          'livloes': 'livløs',
          'loeb': 'løb',
          'loefte': 'løfte',
          'meningsloest': 'meningsløst',
          'moede': 'møde',
          'moenster': 'mønster',
          'moenstre': 'mønstre',
          'nervoes': 'nervøs',
          'noedt': 'nødt',
          'noedvendigvis': 'nødvendigvis',
          'oedelægge': 'ødelægge',
          'oeh': 'øh',
          'oeje': 'øje',
          'oejnene': 'øjnene',
          'oenske': 'ønske',
          'oeverst': 'øverst',
          'opfoerte': 'opførte',
          'opgoer': 'opgør',
          'oproer': 'oprør',
          'opsoeg': 'opsøg',
          'planloest': 'planløst',
          'proev': 'prøv',
          'rastloes': 'rastløs',
          'roe-roede': 'røde',
          'roed': 'rød',
          'roed,': 'rød,',
          'roed.': 'rød.',
          'roed...': 'rød...',
          'roede,': 'røde,',
          'roede.': 'røde.',
          'roede..': 'røde..',
          'roede...': 'røde...',
          'roeffel': 'røffel',
          'roeg': 'røg',
          'roer': 'rør',
          'rumskibs-vægloest-agtigt': 'rumskibs-vægløst-agtigt',
          'samhoerighed': 'samhørighed',
          'sammenstoed': 'sammenstød',
          'selvfoelgelig': 'selvfølgelig',
          'sideloebende': 'sideløbende',
          'skoed': 'skød',
          'skoejter': 'skøjter',
          'skoen': 'skøn',
          'skoert': 'skørt',
          'snoed': 'snød',
          'snoerrer': 'snørrer',
          'soed': 'sød',
          'soeg': 'søg',
          'soen': 'søn',
          'soerger': 'sørger',
          'soesteren': 'søsteren',
          'spoeg': 'spøg',
          'spoerge': 'spørge',
          'stoed': 'stød',
          'stoerre': 'større',
          'stoetter': 'støtte',
          'stroeget': 'strøget',
          'svoemme': 'svømme',
          'titte-boeh-leg': 'titte-bøh-leg',
          'toehoe..': 'tøhø..',
          'toer': 'tør',
          'troejen': 'trøjen',
          'troest': 'trøst',
          'udgoer': 'udgør',
          'undersoege': 'undersøge',
          'understoettet': 'understøttet'}


if __name__=='__main__':
    # Load  
    rpath = Path('data') / 'raw'
    ppath = Path('data') / 'processed'
    df = pd.read_csv(rpath/'participants.csv',  sep=',')

    # Removing 3 rows
    df = df[~((df['Gender']=='F/M')|(df['Age'].isnull()))]

    # Only keep relevant columns
    df.drop(['SharedID', 
            'OverallStudy', 
            'Diagnosis2', 
            'Education', 
            'Unnamed: 11'], axis=1, inplace=True)
    df['ID'] = df['ID'].astype(str)

    # Read in transcripts
    trans = pd.read_csv(rpath/'data.csv')
    print(f'We have {trans.shape[0]} in the raw file')

    # Remove mixed diagnoses
    trans = trans[trans['Group']!='Mixed']
    trans = trans[trans['Diagnosis']!='Mixed']
    trans = trans[trans['Diagnosis']!='ASD&Schizophrenia']

    # Remap names
    trans['Group'] = trans['Group'].replace({'Schizophrenia': 'SCHZ', 
                                             'Depression': 'DEPR'})
    trans['Diagnosis'] = trans['Diagnosis'].replace({'Schizophrenia': 'SCHZ', 
                                                     'Depression': 'DEPR', 
                                                     'Control': 'TD'})
    
    # Exclude some subjects (remission, chronic, controls with no data)
    remission = trans[(trans["Diagnosis"] == "DEPR") & (trans["Recovered"] == 1)]
    trans = trans.drop(remission.index)
    trans.loc[trans["Sub File"].str.contains("dpc", na=False), "Study"] = 2
    trans = trans[trans["File"] != "Depression-Controls-DK-Triangles-2-Sheet1.csv"]
    print(f'We have {trans.shape[0]} transcripts removing mixed diagnoses etc.')

    # Make ID and remap wrong ones
    trans['ID'] = trans['Group'] + '_' + trans['Diagnosis'] + '_' + trans['Study'].astype(str) + '_' + trans['Subject'].astype(str)
    trans['ID'].replace({'ASD_ASD_3_308': 'ASD_ASD_3_108',
                         'ASD_ASD_3_309': 'ASD_ASD_3_109',
                         'ASD_ASD_3_419': 'ASD_ASD_4_419',
                         'ASD_ASD_3_321': 'ASD_ASD_4_421',
                         'ASD_ASD_3_322': 'ASD_ASD_4_422',
                         'ASD_ASD_3_325': 'ASD_ASD_4_425',
                         'ASD_ASD_3_327': 'ASD_ASD_4_427',
                         'ASD_ASD_3_328': 'ASD_ASD_4_428',
                         'ASD_ASD_3_435': 'ASD_ASD_4_435',
                         'ASD_ASD_3_436': 'ASD_ASD_4_436',
                         'ASD_ASD_3_438': 'ASD_ASD_4_438',
                         'ASD_TD_3_408' : 'ASD_TD_4_408',
                         'ASD_TD_3_409' : 'ASD_TD_4_409',
                         'ASD_TD_3_412' : 'ASD_TD_4_412',
                         'ASD_TD_3_419' : 'ASD_TD_4_419',
                         'ASD_TD_3_421' : 'ASD_TD_4_421',
                         'ASD_TD_3_422' : 'ASD_TD_4_422',
                         'ASD_TD_3_425' : 'ASD_TD_4_425',
                         'ASD_TD_3_427' : 'ASD_TD_4_427',
                         'ASD_TD_3_428' : 'ASD_TD_4_428',
                         'ASD_TD_3_435' : 'ASD_TD_4_435',
                         'ASD_TD_3_436' : 'ASD_TD_4_436',
                         'ASD_TD_3_438' : 'ASD_TD_4_438'}, inplace=True)


    # Explore overlaps between two files
    no_id = set(trans.ID.unique()) - set(df.ID.unique())
    no_trans = set(df.ID.unique()) - set(trans.ID.unique())
    trans = trans[trans['ID']!='ASD_ASD_3_312']
    print(f'We have {trans.shape[0]} transcripts after removing 1 subject')

    # Check length and exclude 
    trans['Transcript'] = trans['Transcript'].astype(str)
    trans['n_wds'] = trans.Transcript.apply(lambda x: len(x.split())).values
    print(f'Total transcripts: {trans.shape[0]}')
    print(f" Below 5 wds: {round((trans['n_wds'] < 5).sum() / trans.shape[0], 2)}, {(trans['n_wds'] < 5).sum()}")
    print(f" Below 10 wds: {round((trans['n_wds'] < 10).sum() / trans.shape[0], 2)}, {(trans['n_wds'] < 10).sum()}")

    # Only keep data above 5 words
    trans = trans[trans['n_wds']>=5]
    print(f'We have {trans.shape[0]} transcripts after filtering by length')

    # Plot histogram
    trans['Diagnosis_binary'] = trans['Diagnosis'].map({'ASD': '1', 
                                                        'DEPR': '1', 
                                                        'SCHZ': '1', 
                                                        'TD': '0'})
    sns.displot(data=trans, 
                x='n_wds', col='Group', 
                hue='Diagnosis_binary', 
                binwidth=5)

    # Do some processing to the text
    trans['Transcript'] = trans['Transcript'].str.replace('\[sic\]', '')
    trans['Transcript'] = trans['Transcript'].replace(oedict, 
                                                      regex=True)

    # Save full
    trans['ExampleID'] = trans['ID'] + '_' + trans['Trial'].astype(str)
    trans = trans[['File', 'ExampleID', 'Diagnosis', 'Group', 'ID', 'Transcript']]
    trans.to_csv(ppath/'full.csv', sep=',', index=False)

    # Do train-test split
    train, val, test = [trans[trans['ID'].isin(pd.read_csv(f'data/splits/{s}_split.csv', 
                                                           sep=',').ID.unique())] 
                        for s in ['train', 'validation', 'test']]
    train.to_csv(ppath/'train.csv', sep=',', index=False)
    val.to_csv(ppath/'validation.csv', sep=',', index=False)
    test.to_csv(ppath/'test.csv', sep=',', index=False)

    # Print breakdown of subjects
    with open('data/outs/set_sizes.txt', 'w') as f:
        print(f'\nN training examples: {train.shape[0]}, N subj: {train.ID.nunique()}', file=f)
        print(train.groupby(['Group', 'Diagnosis']).ID.nunique(), file=f)
        print(f'\nN val examples: {val.shape[0]}, N subj: {val.ID.nunique()}', file=f)
        print(val.groupby(['Group', 'Diagnosis']).ID.nunique(), file=f)
        print(f'\nN test examples: {test.shape[0]}, N subj: {test.ID.nunique()}', file=f)
        print(test.groupby(['Group', 'Diagnosis']).ID.nunique(), file=f)

    # Print
    with open('data/outs/missing_transcripts.txt', 'w') as f:
        print('Ids with no transcript', file=f)
        for n in no_trans:
            print(n, file=f)
