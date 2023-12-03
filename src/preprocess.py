import argparse
import pandas as pd
import nltk
import re
from tqdm import tqdm
import requests
from pprint import pprint
from src.constants import *

KEYWORD = ['cardiomegaly', 'cardiothoracic', 'cardiothymic', 'cardiac', 'heart', 'size', 'enlarged', 'enlargement', 'shadow', 'contour', 'silhouette',
           'edema', 'failure', 'chf', 'vascular', 'congest', 'congestion', 'pulmonary',
           'reticulonodular', 'fibrointerstitial', 'infiltration', 'infiltrate', 'infiltrative', 'fibroinfiltration', 'fibrotic', 'fibro-reticular-nodular', 'fibrocalcific', 'patchy', 'density', 'interstitial', 'opacification',
           'consolidation',
           'opacity', 'opacities', 'translucency', 'airspace', 'air', 'space', 'disease', 'marking', 'pattern', 'lung', 'reticular', 'reticulation', 'parenchymal', 'scarring', 'peribronchial', 'wall', 'thickening', 'scar', 'recticular',
           'pleural', 'fluid', 'effusion', 'pneumohydrothorax', 'hydropneumothorax', 'hydro', 'pneumothorax', 'pneumohydrothoraces', 'hydropneumothoraces', 'pneumothoraces',
           'atelectasis',
           'mass', 'cavitary', 'lesion', 'carcinoma', 'neoplasm', 'tumor',
           'nodular', 'densities', 'nodule', 'granuloma']

def editDistance(df):
    # use edit distance from nltk -> has problems in similar words such as 'opacification' and 'calcification'
    print('Start preprocessing data')
    imp = df['Report Impression']
    imp = imp.str.strip()
    imp = imp.replace('\n',' ', regex=True)
    imp = imp.replace('\s+', ' ', regex=True)
    imp = imp.str.strip()

    idx = []
    for i in tqdm(imp.index):
        words = imp[i].split()
        l = []
        n = 0
        for word in words:
            for k in KEYWORD:
                if nltk.edit_distance(word.lower(), k.lower()) < 0.2*len(k):
                    new_word = k.lower()
                    n = 1
                    if re.search(r'\w\.\B', word):
                        new_word += '.'
                    break
                else: 
                    new_word = word 
            l.append(new_word)
        if n ==1:
            idx.append(i)
        df.at[i,'Report Impression'] = ' '.join(l)
    return df, imp, idx

def spellingCorrector(df):
    # tool from https://sapling.ai/docs/api/edits-overview/
    # has query limits (free: cannot use to label many reports)
    print('Start preprocessing data')
    imp = df['Report Impression']
    imp = imp.str.strip()
    imp = imp.replace('\n',' ', regex=True)
    imp = imp.replace('\s+', ' ', regex=True)
    imp = imp.str.strip()

    new_imp = []
    for i in tqdm(imp):
        try:
            response = requests.post(
                "https://api.sapling.ai/api/v1/spellcheck",
                json={
                    "key": "PJ5IG5GH5E19TIQOBB7CY0TGN2SRDDHU",
                    "text": i,
                    "session_id": "report session",
                    "medical": True
                }
            )
            resp_json = response.json()
            if 200 <= response.status_code < 300:
                edits = resp_json['edits']
                pprint(edits)
            else:
                print('Error: ', resp_json)
        except Exception as e:
            print('Error: ', e)

        new_report = i
        for edit in edits:
            start = edit['start'] + edit['sentence_start']
            end = edit['end'] + edit['sentence_start']
            replacement = edit['replacement']
            new_report = new_report.replace(i[start:end], replacement)
        new_imp.append(new_report)
    
    df['Report Impression'] = pd.DataFrame({'Report Impression':new_imp})
    return df    

def compareData(imp, new_df, idx, path_com):
    d = {'Before preprocess': [], 'After preprocess': []}
    for i in idx:
        d['Before preprocess'].append(imp[i])
        d['After preprocess'].append(new_df.at[i,'Report Impression'])

    df_res = pd.DataFrame(d)
    df_res = df_res.fillna('')

    df_res.to_csv(path_com, index=False)


def saveData(df_pre, path_out):
    print('\nSave preprocessed data')
    df_pre.to_csv(path_out, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess report before fedding to the models.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='path to csv file of dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='path to csv file for output (preprocessed data)')
    parser.add_argument('--compare', type=str, required=False,
                        help='path to csv file for comapring preprocessed data')
    args = parser.parse_args()
    path_in = args.dataset
    path_out = args.output
    path_com = args.compare

    # read data
    df_in = pd.read_csv(path_in)

    # preprocess data
    df_pre = spellingCorrector(df_in)

    # compare before and after preprocess
    # compareData(imp, df_pre, idx, path_com)

    # save to file
    saveData(df_pre, path_out)