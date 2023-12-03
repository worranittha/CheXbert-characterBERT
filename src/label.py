import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import utils
from models.bert_labeler import bert_labeler
from bert_tokenizer import tokenize
from transformers import BertTokenizer
from collections import OrderedDict
from dataset.unlabeled_dataset import UnlabeledDataset
from constants import *
from tqdm import tqdm
from models.character_bert import CharacterBertModel
import ast

def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len,
       where the reports have no associated labels
    @param sample_list (List): A list of samples. Each sample is a dictionary with
                               keys 'imp', 'len' as returned by the __getitem__
                               function of ImpressionsDataset

    @returns batch (dictionary): A dictionary with keys 'imp' and 'len' but now
                                 'imp' is a tensor with padding and batch size as the
                                 first dimension. 'len' is a list of the length of 
                                 each sequence in batch
    """
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                  batch_first=True,
                                                  padding_value=PAD_IDX)
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list}
    return batch

def load_unlabeled_data(csv_path, charbert, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False):
    """ Create UnlabeledDataset object for the input reports
    @param csv_path (string): path to csv file containing reports
    @param charbert (bool): whether model use CharacterBERT or not
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not  
    
    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(csv_path, charbert)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    return loader
    
def label(checkpoint_path, csv_path, charbert, ld=None):
    """Labels a dataset of reports
    @param checkpoint_path (string): location of saved model checkpoint 
    @param csv_path (string): location of csv with reports
    @param charbert (bool): whether model use CharacterBERT or not

    @returns y_pred (List[List[int]]): Labels for each of 7 conditions, per report  
    """
    if ld == None:
        ld = load_unlabeled_data(csv_path, charbert)
    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # use characterBERT
    if charbert:
        print('use CharacterBERT')
        charmodel = CharacterBertModel.from_pretrained('./pretrained-models/medical_character_bert/')
        charmodel = charmodel.to(device)
        # model.bert.embeddings.word_embeddings = charmodel.embeddings.word_embeddings                        # replace BERT's wordpiece embedding with CharacterBERT's wordpiece embedding
        # model.bert.embeddings = charmodel.embeddings                                                      # replace BERT's embedding with CharacterBERT's embedding

    hidden_size = model.bert.pooler.dense.in_features
    model.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 2, bias=True) for _ in range(7)]).cuda()     # add .cuda() to use parameters from GPU not CPU
    
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model) #to utilize multiple GPU's
    #     model = model.to(device)
    #     checkpoint = torch.load(checkpoint_path, map_location=device)    
    #     model.load_state_dict(checkpoint['model_state_dict'])
    # else:
    #     model = model.to(device)
    #     checkpoint = torch.load(checkpoint_path, map_location=device)
    #     new_state_dict = OrderedDict()
    #     for k, v in checkpoint['model_state_dict'].items():
    #         name = k[7:] # remove `module.`
    #         new_state_dict[name] = v
    #     model.load_state_dict(new_state_dict)     

    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)                                           # load checkpoint from file
    model.load_state_dict(checkpoint['model_state_dict'])                                                   # load model and optimizer from checkpoint

    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]                                                           # y_pred[i] keeps results of all reprots in condition[i]
    y_pred_score = [[] for _ in range(len(CONDITIONS))]                                                     # y_pred_score[i] keeps normalize prediction scores of results of all reprots in condition[i]

    print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % BATCH_SIZE)
    with torch.no_grad():
        for i, data in enumerate(tqdm(ld)):
            batch = data['imp']                                                                             #(batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            inputs = {
                "input_ids": batch,
                "attention_mask": attn_mask,
                "charbert": charbert
            }
            out = model(**inputs)                                                                           # out (list) shape is (7,batch_size,2)

            for j in range(len(out)):
                #out[j] (tensor) shape is (batch_size,2)
                # curr_y_pred = torch.nn.functional.softmax(out[j], dim=-1).argmax(dim=1) #shape is (batch_size)
                # y_pred_score[j].append(torch.nn.functional.softmax(out[j], dim=-1).max(dim=1).values) #append tensor to list

                curr_y_pred = out[j].argmax(dim=1)                                                          # curr_y_pred (tensor) shape is (batch_size)
                y_pred[j].append(curr_y_pred)                                                               # append tensor to list (y_pred[j] is a list of tensor, shape is number of input reports/batch_size )
                y_pred_score[j].append(torch.nn.functional.softmax(out[j], dim=-1)[:,1])

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)                                                         # concat to create 1 tensor for each observation (y_pred[j] is a tensor, shape is number of input reports )
            y_pred_score[j] = torch.cat(y_pred_score[j], dim=0)
        
    if was_training:
        model.train()
                                                                                                            # y_pred is a list, shape is number of observations
    y_pred = [t.tolist() for t in y_pred]                                                                   # y_pred[j] is a list, shape is number of input reports 
    y_pred_score = [t.tolist() for t in y_pred_score]
    
    return y_pred, y_pred_score

def ensemble_model(checkpoint_paths, csv_path, charberts):
    """Ensemble models
    @param checkpoint_paths (List[string]): list of path to saved model checkpoints
    @param csv_path (string): path to csv file containing reports
    @param charberts (List[bool]): list of conditions whether models use CharacterBERT or not

    @returns y_preds (List[List[int]]): labels for each of 7 conditions, per report 
    @returns y_pred_avg (List[List[int]]): prediction scores for each of 7 conditions, per report 
    """
    y_pred_scores = []
    for i,path in enumerate(checkpoint_paths):
        print('Model {}'.format(i))
        _, y_pred_score = label(path, csv_path, charberts[i])
        y_pred_scores.append(np.array(y_pred_score))
    
    y_pred_avg = np.mean(y_pred_scores, axis=0)
    y_pred_avg = np.average(y_pred_scores, weights=None, axis=0)
    y_preds = np.where(y_pred_avg >= 0.5, 1, 0)

    return y_preds, y_pred_avg

def save_preds(y_pred, csv_path, out_path):
    """Save predictions as out_path/labeled_reports.csv 
    @param y_pred (List[List[int]]): list of predictions for each report
    @param csv_path (string): path to csv containing reports
    @param out_path (string): path to output directory
    """
    y_pred = np.array(y_pred)
    y_pred = y_pred.T
    
    df = pd.DataFrame(y_pred, columns=CONDITIONS)
    reports = pd.read_csv(csv_path)['Report Impression']

    df['Report Impression'] = reports.tolist()
    new_cols = ['Report Impression'] + CONDITIONS
    df = df[new_cols]
    
    df.to_csv(os.path.join(out_path, 'labeled_reports.csv'), index=False)

def save_preds_with_score(y_pred, y_pred_score, csv_path, out_path):
    """Save predictions with their scores as out_path/labeled_reports.csv 
    @param y_pred (List[List[int]]): list of predictions for each report
    @param y_pred_score (List[List[int]]): list of prediction scores for each report
    @param csv_path (string): path to csv containing reports
    @param out_path (string): path to output directory
    """
    y_pred = np.array(y_pred)
    y_pred = y_pred.T

    y_pred_score = np.array(y_pred_score)
    y_pred_score = y_pred_score.T

    y_all = np.concatenate((y_pred, y_pred_score), axis=1)
    
    new_con = [i+' Probability' for i in CONDITIONS]
    df = pd.DataFrame(y_all, columns=CONDITIONS+new_con)
    
    raw_df = pd.read_csv(csv_path, engine='python')
    reports = raw_df['Report Impression']
    img_idx = raw_df['Image Index']

    df['Report Impression'] = reports.tolist()
    df['Image Index'] = img_idx.tolist()

    new_cols = ['Image Index','Report Impression'] + CONDITIONS + new_con
    df = df[new_cols] #sort new column name

    df.to_csv(out_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label a csv file containing radiology reports')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                        help='path to csv containing reports. The reports should be \
                              under the \"Report Impression\" column')
    parser.add_argument('-o', '--output_dir', type=str, nargs='?', required=True,
                        help='path to intended output folder')
    parser.add_argument('-c', '--checkpoint', type=str, nargs='?', required=True,
                        help='path to the pytorch checkpoint')
    parser.add_argument('--charbert', type=str, nargs='?', required=False,
                        help='condition for using CharacterBERT')
    parser.add_argument('--ensemble', type=str, nargs='?', required=False,
                        help='condition for ensemble model')
    args = parser.parse_args()
    
    csv_path = args.data
    out_path = args.output_dir
    checkpoint_path = args.checkpoint
    charbert = args.charbert
    ensemble = args.ensemble
    
    if not ensemble:
        y_pred, y_pred_score = label(checkpoint_path, csv_path, charbert)
    else:
        checkpoint_paths = ast.literal_eval(checkpoint_path)
        charberts = ast.literal_eval(charbert)
        y_pred, y_pred_score = ensemble_model(checkpoint_paths, csv_path, charberts)
        
    # save_preds(y_pred, csv_path, out_path)
    save_preds_with_score(y_pred, y_pred_score, csv_path, out_path)
