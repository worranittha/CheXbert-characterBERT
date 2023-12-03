import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, recall_score, precision_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from CheXbert.src.constants import *

N_CONDITIONS = len(CONDITIONS)

def plot_roc(df_true, df_pred):
    for n in range(N_CONDITIONS):
        y_test = df_true[CONDITIONS[n]]        
        y_pred_score = df_pred[CONDITIONS[n]+' Probability']
        
        # auc
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_score, pos_label=1)
        roc_auc = auc(fpr, tpr)

        sens, spec, idx = choose_operating_point(fpr, tpr)
        print('Optimal Threshold=%f' % (thresholds[idx]))

        plt.figure(dpi=100)
        plt.title(n)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc, marker='.')
        if idx != None:
            plt.scatter(fpr[idx], tpr[idx], marker='o', color='black')
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

def choose_operating_point(fpr, tpr):
    sens = 0
    spec = 0
    J = 0
    idx = 0
    for i, (_fpr, _tpr) in enumerate(zip(fpr, tpr)):
        if _tpr - _fpr > J:
            sens = _tpr
            spec = 1-_fpr
            J = _tpr - _fpr
            idx = i
    return sens, spec, idx

def evaluate(true_path, pred_path, pred_score=True, thres=False):
    """ Calculate AUC, Sensitivity, Specificity, PPV, NPV and Accuracy
    @param true_path (string): path to csv file containing reports with actual labels 
    @param pred_path (string): path to csv file containing reports with predicted labels 
    @param pred_score (bool): whether to calculate AUC or not
    @param thres (bool): whether to calculate optimal thresholds or not
    """
    df_true = pd.read_csv(true_path)
    df_pred = pd.read_csv(pred_path)

    score = [[] for _ in CONDITIONS]
    score_name = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy']
    if pred_score:
        score_name = ['AUC'] + score_name
    # if thres:
    #     plot_roc(df_true, df_pred)

    for n in range(N_CONDITIONS):
        y_test = df_true[CONDITIONS[n]]
        y_pred = df_pred[CONDITIONS[n]]
        
        if pred_score:
            y_pred_score = df_pred[CONDITIONS[n]+' Probability']
            # auc
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_score, pos_label=1)
            roc_auc = auc(fpr, tpr)
            if thres:
                _, _, idx = choose_operating_point(fpr, tpr)
                print('Optimal Threshold=%f' % (thresholds[idx]))
            score[n].append(roc_auc)

        # sensitivity (recall)
        score[n].append(recall_score(y_test, y_pred, zero_division=0, pos_label=1))
        # specificity
        score[n].append(recall_score(y_test, y_pred, zero_division=1, pos_label=0))
        # PPV
        score[n].append(precision_score(y_test, y_pred, zero_division=0, pos_label=1))
        # NPV
        score[n].append(precision_score(y_test, y_pred, zero_division=0, pos_label=0))
        # accuracy
        score[n].append(accuracy_score(y_test, y_pred))

    score_arr = np.array(score)
    score.append([sum(score_arr[:,i])/7 for i in range(len(score_name))])

    if not thres:
        display(pd.DataFrame(score, index=CONDITIONS+[''], columns=score_name))

def number_of_diff(true_path, pred_path):
    """ Calculate number of different labels between 2 datasets
    @param true_path (string): path to csv file containing reports with actual labels 
    @param pred_path (string): path to csv file containing reports with predicted labels 
    """
    df_true = pd.read_csv(true_path)
    df_pred = pd.read_csv(pred_path)
    print('df_true shape:', df_true.shape)
    print('df_pred shape:', df_pred.shape)
    
    df_true['Conflict Check'] = ''
    for i in df_true.index:
        for n in CONDITIONS:
            if df_pred.at[i,n] != df_true.at[i,n]:
                df_true.at[i,'Conflict Check'] = 1

    diff_res = df_true.loc[df_true['Conflict Check'] == 1]
    print('number of diff reports:', diff_res.shape[0])

def saveAllDiffData(true_path, path_1, path_2, model_name_1, model_name_2, out_path):
    """ Save reports with different labels between 2 models and also compare to actual label in each observation 
    @param true_path (string): path to csv file containing reports with actual labels 
    @param path_1 (string): path to csv file containing reports with predicted labels from model 1
    @param path_2 (string): path to csv file containing reports with predicted labels from model 2
    @param model_name_1 (string): name of model 1 
    @param model_name_2 (string): name of model 2
    @param out_path (string): path to xlsx file for output reports
    """
    df_true = pd.read_csv(true_path)
    df_1 = pd.read_csv(path_1)
    df_2 = pd.read_csv(path_2)
    print('df_true shape:', df_true.shape)
    print('df_1 shape:', df_1.shape)
    print('df_2 shape:', df_2.shape)

    for name in CONDITIONS:
        df_diff = df_true[df_1[name] != df_2[name]]
        d = {}
        d2 = {'0': [], '1': [], '2': [], '3': []}

        for i in df_diff.index:
            n = int(df_diff.at[i, name])
            if df_1.at[i, name] == n:
                d2[str(n*2)].append(df_diff.at[i,'Report Impression'])
            else:
                d2[str((n*2)+1)].append(df_diff.at[i,'Report Impression'])

        d['True {}, {} 0, {} 1'.format(0, model_name_1, model_name_2)] = pd.Series(d2['0'])
        d['True {}, {} 1, {} 0'.format(0, model_name_1, model_name_2)] = pd.Series(d2['1'])
        d['True {}, {} 0, {} 1'.format(1, model_name_1, model_name_2)] = pd.Series(d2['3'])
        d['True {}, {} 1, {} 0'.format(1, model_name_1, model_name_2)] = pd.Series(d2['2'])

        df_res = pd.DataFrame(d)
        df_res = df_res.fillna('')

        # wrong both CheXbert and CharacterBERT
        df_res_0 = df_true[(df_true[name] == 0) & (df_1[name] == 1) & (df_2[name] == 1)]['Report Impression'].to_frame().reset_index(drop=True)
        df_res_1 = df_true[(df_true[name] == 1) & (df_1[name] == 0) & (df_2[name] == 0)]['Report Impression'].to_frame().reset_index(drop=True)

        df_res_0.columns = ['True 0, {} 1, {} 1'.format(model_name_1, model_name_2)]
        df_res_1.columns = ['True 1, {} 0, {} 0'.format(model_name_1, model_name_2)]

        df_res_new = pd.concat([df_res,df_res_0,df_res_1], axis=1)

        if out_path:
            mode = 'w' if name == 'Cardiomegaly' else 'a'
            with pd.ExcelWriter(out_path, engine='openpyxl', mode=mode) as writer:
                df_res_new.to_excel(writer, sheet_name=name, index=False, header=True)

def saveDiffData_true(true_path, pred_path, model_name, out_path, annotation='True'):
    """ Save reports with different labels between existing annotation and model
    @param true_path (string): path to csv file containing reports with existing annotations 
    @param pred_path (string): path to csv file containing reports with labels from model
    @param model_name (string): name of model
    @param out_path (string): path to xlsx file for output reports
    @param annotation (string): name of existing annotation
    """
    df_true = pd.read_csv(true_path)
    df_pred = pd.read_csv(pred_path)
    print('df_true shape:', df_true.shape)
    print('df_pred shape:', df_pred.shape)

    for name in CONDITIONS:
        df_diff = df_true[df_true[name] != df_pred[name]]
        d = {}

        d['{} 0, {} 1'.format(annotation, model_name)] = df_diff[df_diff[name] == 0]['Report Impression'].reset_index(drop=True)
        d['{} 1, {} 0'.format(annotation, model_name)] = df_diff[df_diff[name] == 1]['Report Impression'].reset_index(drop=True)

        print('Diff report {}: {} + {}'.format(name, d['{} 0, {} 1'.format(annotation, model_name)].shape[0],d['{} 1, {} 0'.format(annotation, model_name)].shape[0]))

        df_res = pd.DataFrame(d)
        df_res = df_res.fillna('')

        if out_path:
            mode = 'w' if name == 'Cardiomegaly' else 'a'
            with pd.ExcelWriter(out_path, engine='openpyxl', mode=mode) as writer:
                df_res.to_excel(writer, sheet_name=name, index=False, header=True)