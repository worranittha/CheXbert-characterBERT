import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer
from src.constants import *

N_CONDITIONS = len(CONDITIONS)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def select_data(df, path=False):
    """Select data from input dataset
    @param df (dataframe): input dataset
    @param path (bool): whether to keep image paths from input dataset or not

    @return select_df (dataframe): selected data
    """
    if not path:
        select_df = df[['Image Index', 'Report Impression'] + CONDITIONS].copy()
    else:
        select_df = df[['Image Index', 'Image Path', 'Report Impression'] + CONDITIONS].copy()
    print('df shape:', select_df.shape)
    print('df columns:', select_df.columns)
    return select_df

def rename(df, extra_word=None):
    """Rename column in dataset
    @param df (dataframe): input dataset
    @param extra_word (bool): whether columns in dataset have extra keyword or not (such as Inspectra or BERT)

    @return (dataframe): renamed data
    """
    if extra_word == None:
        return df.rename(columns={'Reports': 'Report Impression'})
    else:
        return df.rename(columns={'Reports': 'Report Impression',
                                  'Cardiomegaly '+extra_word: 'Cardiomegaly', 
                                  'Edema '+extra_word: 'Edema', 
                                  'Inspectra Lung Opacity v1 '+extra_word: 'Inspectra Lung Opacity v1', 
                                  'Pleural Effusion '+extra_word: 'Pleural Effusion', 
                                  'Atelectasis '+extra_word: 'Atelectasis', 
                                  'Nodule '+extra_word: 'Nodule', 
                                  'Mass '+extra_word: 'Mass'})
    
def replace_and_drop(df):
    """Replace -1 with NaN and drop NaN value
    @param df (dataframe): input dataset

    @return new_df (dataframe): replaced and dropped data
    """
    new_df = df.replace(-1.0, np.nan)
    new_df = new_df.dropna()
    # print('nan index:', df.index[df.isna().any(axis=1)])
    print('new df shape:', new_df.shape)
    return new_df

def all_2_int(df):
    """Change all labels to int
    @param df (dataframe): input dataset

    @return df (dataframe): dataset with int values of all labels
    """
    for i in CONDITIONS:
        df[i] = df[i].astype(int)
    return df

def remove_longer_report(df, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):
    """Remove reports that have subword tokens more than 512 (exceed the maximum of input tokens of BERT)
    @param df (dataframe): input dataset
    @param tokenizer (model): BERT tokenizer for tokenizing reports

    @return df_drop (dataframe): dataset that is already dropped longer reports
    """
    imp = df['Report Impression']
    imp = imp.str.strip()
    imp = imp.replace('\n',' ', regex=True)
    imp = imp.replace('\s+', ' ', regex=True)
    imp = imp.str.strip()

    drop_idx = []
    for i in tqdm(range(imp.shape[0])):
        tokenized_imp = tokenizer.tokenize(imp.iloc[i])
        if len(tokenizer.encode_plus(tokenized_imp)['input_ids']) > 512:
            drop_idx.append(i)
    print('number of longer reports:', len(drop_idx))

    df_drop = df.drop(df.index[drop_idx])
    df_drop = df_drop.reset_index(drop=True)
    print('df shape before drop longer report', df.shape)
    print('df shape after drop longer report', df_drop.shape)

    return df_drop

def split_train(df, train_path, valid_path, random_state=42): 
    """Split dataset to train and validate
    @param df (dataframe): input dataset
    @param train_path (string): path to csv file of training set
    @param valid_path (string): path to csv file of validation set
    @param random_state (int): number of random states
    """
    df_train, df_valid = train_test_split(df, test_size=0.1, random_state=random_state)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    print('df shape:', df.shape)
    print('df_train shape:', df_train.shape)
    print('df_valid shape:', df_valid.shape)
    print('--------------------------------------')

    print('save to file')
    df_train.to_csv(train_path, index=False)
    df_valid.to_csv(valid_path, index=False)

def split_train_test(df, train_path, valid_path, test_path, random_state=42): 
    """Split dataset to train, validate and test
    @param df (dataframe): input dataset
    @param train_path (string): path to csv file of training set
    @param valid_path (string): path to csv file of validation set
    @param test_path (string): path to csv file of test set
    @param random_state (int): number of random states
    """
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state)
    df_test, df_valid = train_test_split(df_test, test_size=0.5, random_state=random_state)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print('df shape:', df.shape)
    print('df_train shape:', df_train.shape)
    print('df_valid shape:', df_valid.shape)
    print('df_test shape:', df_test.shape)
    print('--------------------------------------')

    print('save to file')
    df_train.to_csv(train_path, index=False)
    df_valid.to_csv(valid_path, index=False)
    df_test.to_csv(test_path, index=False)

def count_data(df=None, path=None, dataset_type=None):
    """Count number of negative and positive states in each observation in dataset
    @param df (dataframe): input dataset
    @param path (string): path to csv file of input dataset
    @param dataset_type (string): type of dataset (train/validate/test)
    """
    if path:
        df = pd.read_csv(path)
        print(dataset_type)
    print('df shape:', df.shape)
    for i in CONDITIONS:
        n_0 = df[i].value_counts()[0] if 0 in df[i].unique() else 0
        n_1 = df[i].value_counts()[1] if 1 in df[i].unique() else 0
        n_u = df[i].value_counts()[-1] if -1 in df[i].unique() else 0
        print('{}, 0: {}, 1: {}, -1: {}'.format(i, n_0, n_1, n_u))
    print('-------------------------------------------------------')

def compare_annotation(path_1, path_2):
    """Count number of reports with different labels between 2 datasets in each observation
    @param path_1 (string): path to csv file of dataset 1
    @param path_2 (string): path to csv file of dataset 2
    """
    df_1 = pd.read_csv(path_1)
    df_2 = pd.read_csv(path_2)

    n = dict.fromkeys(CONDITIONS,0)
    n_0_to_1 = dict.fromkeys(CONDITIONS,0)
    n_1_to_0 = dict.fromkeys(CONDITIONS,0)
    df_1['Difference'] = ''

    for i in range(df_1.shape[0]):
        for j in CONDITIONS:
            if df_1.at[i,j] != df_2.at[i,j]:
                if df_1.at[i,j] == 0:
                    n_0_to_1[j] += 1
                else:
                    n_1_to_0[j] += 1
                n[j] += 1
                if df_1.at[i,'Difference'] != '':
                    df_1.at[i,'Difference'] += ', ' + j
                else:
                    df_1.at[i,'Difference'] = j

    diff = df_1.loc[df_1['Difference'] != ''].shape[0]

    print('Number of reports:', df_1.shape[0])
    print('Number of different annotations:', diff, '->', (diff/df_1.shape[0])*100)
    print('Number in each observaiton:',n)
    print('From 0 to 1:',n_0_to_1)
    print('From 1 to 0:',n_1_to_0)

def findDiffData(true_path, pred_path, name, value):
    """Find reports with different labels between 2 datasets in specific observation and actual label
    @param true_path (string): path to csv file of reports with actual labels
    @param pred_path (string): path to csv file of reports with predicted labels
    @param name (string): specific observaiton name
    @param value (int): specific actual label
    """
    df_true = pd.read_csv(true_path)
    df_pred = pd.read_csv(pred_path)

    df = df_true[df_true[name] != df_pred[name]]
    df = df[df[name] == value]

    for i in df.index:
        print('ground truth: {}, bert: {}'.format(df_true.at[i, name], df_pred.at[i, name]))
        print(df.at[i,'Image Index'], df.at[i,'Report Impression'])
        print('---------------------------------------------------------------')
