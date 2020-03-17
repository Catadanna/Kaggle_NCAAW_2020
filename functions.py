from constants import *

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import csv
import math
import pickle

import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV, TheilSenRegressor, HuberRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier 
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

import xgboost
from xgboost import XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

# Description: Read Data from CSV file into Pandas DataFrame
def read_data(inFile, sep=','):
    df_op = pd.read_csv(filepath_or_buffer=inFile, low_memory=False, encoding='utf-8', sep=sep)
    return df_op

# Description: Write Pandas DataFrame into CSV file
def write_data(df, outFile):
    f = open(outFile+'.csv', 'w')
    r = df.to_csv(index=False, path_or_buf=f)
    f.close()

# Description: Create submission file:    
def print_submission_into_file(y_pred, df_test_id, algo=""):
    l = []
    for myindex in range(y_pred.shape[0]):
        Y0 = y_pred[myindex]
        l.insert(myindex, Y0)
    
    df_pred = pd.DataFrame(pd.Series(l), columns=["Pred"])
    df_result = pd.concat([df_test_id, df_pred], axis=1, sort=False)
     
    f = open('submission'+algo+'.csv', 'w')
    r = df_result.to_csv(index=False, path_or_buf=f)
    f.close()

    return df_result

# Description: Generate string in the format of submission ID
def concat_row(r):
    if r['WTeamID'] < r['LTeamID']:
        res = str(r['Season'])+"_"+str(r['WTeamID'])+"_"+str(r['LTeamID'])
    else:
        res = str(r['Season'])+"_"+str(r['LTeamID'])+"_"+str(r['WTeamID'])
    return res

# Delete leaked from train
def delete_leaked_from_df_train(df_train, df_test):
    # Delete leaked from train
    dft = df_train.loc[:, ['Season','WTeamID','LTeamID']]
    df_train['Concats'] = df_train.apply(concat_row, axis=1)
    df2 = df_test[df_test['ID'].isin(df_train['Concats'].unique())]

    df_train_duplicates = df_train[df_train['Concats'].isin(df_test['ID'].unique())]
    df_train_idx = df_train_duplicates.index.values
    
    df_train = df_train.drop(df_train_idx)
    df_train = df_train.drop('Concats', axis=1)
    
    return df_train

# Convert seed to numeric:
def replace_seed_only(s):
    s = s.replace('W', '')
    s = s.replace('X', '')
    s = s.replace('Y', '')
    s = s.replace('Z', '')
    
    if re.search('(a|b)', s):
        s = s.replace('a', '')
        s = s.replace('b', '')
    else:
        s = s+'0'
     
    return int(s)

# Parse Log Loss       
def log_loss(y_01, y_p):
    n = y_01.shape[0]
    v = np.multiply(y_01, np.log(y_p)) + np.multiply((1-y_01), np.log(1-y_p))
    
    res = -(np.sum(v)/float(n)) 
    return res

# Use aggregation in order to create new columns
def set_aggregation(row, se_agg, se_col, r_col, op_col):
    df_s = se_agg[se_agg[se_col] == row[r_col]]
    df = df_s[df_s['Season']==row['Season']].reset_index(drop=True)
    if df.shape[0] == 0:
        return 0
    else:
        return df.at[0, op_col]
    
# Get value for count features for a team, and replace NaNs withe zero:    
def get_value_for_count(team, team_name, team_count):
    if team in team_count.index:
        return team_count.loc[team, 'Count']
    else:
        return 0
   
def set_WLoc(row):
    if row==1:
        return 2
    elif row==2:
        return 1
    else:
        return 0
    
def write_label(r):
    if r['WTeamID'] < r['LTeamID']:
        return 1
    else:
        return 0
    
def get_labels_df_train(df_train, df_test):
    df_train['Concats'] = df_train.apply(concat_row, axis=1)
    df_train_good = df_train[df_train['Concats'].isin(df_test['ID'].unique())]
    df_train_good['Label'] = df_train_good.apply(write_label, axis=1)
    return df_train_good    
