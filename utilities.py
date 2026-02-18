import os

import pandas as pd
import numpy as np
from functools import reduce
import copy
import matplotlib.pyplot as plt
import seaborn as sns

def ABS_SHAP(df_shap,df):

    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'mediumseagreen','silver')
    
    # Plot it
    
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k1 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k1 = k1.sort_values(by='SHAP_abs',ascending = True)
    return k1



def ensure_directory_exists(directory_path):
    """
    Checks if a directory exists, and if not, creates it.
    
    Args:
        directory_path (str): The path to the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def decode_bytes(val):
    if isinstance(val, bytes):
        try:
            return val.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return val.decode('latin1')
            except UnicodeDecodeError:
                return val.decode('cp1252', errors='replace')
    return val


def read_sas(filename, byte_cols='SCCRIP_ID'):
    df = pd.read_sas(filename)
        
    df[byte_cols] = df[byte_cols].apply(decode_bytes)
    return df
    
def splitBackgroundAndDynamicData(cleanDataFolder, dataFileNames, idColumn='SCCRIP_ID', threshold=0.9):
    dataSet = {'background':[],
               'dynamic':[]
              }
    for f in dataFileNames:
        f_ = cleanDataFolder + f
        ff = f.split('.')[0]
        
        df = read_sas(f_)
        # print(df.dtypes)
        size = df.groupby(idColumn).size()
        size_ = set(size)
        ratio = (size==1).mean()
        
        if (len(size_) == 1) and (list(size_)[0]==1):
            dataSet['background'].append(ff)
        elif (len(size_) != 1) and (ratio>threshold):
            dataSet['background'].append(ff)
            
            dateCol = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
            df = df.sort_values(by=dateCol, ascending=False)            
            df = df.drop_duplicates(subset=idColumn, keep='first')
                       
        else:
            dataSet['dynamic'].append(ff)
            
    return dataSet       

def mergeBackgroundDataset(cleanDataFolder, backgroundDatasetList, idColumn='SCCRIP_ID'):
    dfList = []
    for i in backgroundDatasetList:
        fname = cleanDataFolder + i + '.sas7bdat'
        df_ = read_sas(fname)
        suffix = '_' + i
        df_ = df_.set_index(idColumn).add_suffix(suffix).reset_index()
        
        dfList.append(df_)
    df = reduce(lambda x, y: pd.merge(x, y, on = idColumn, how='outer'), dfList)
    df = df.T.drop_duplicates().T
    df = df.dropna(axis=1, how='all')
    
    df = df.apply(lambda col: pd.to_numeric(col, errors='ignore') 
                      if col.dtypes == object 
                      else col, 
                      axis=0)
    return df


def mergeDynamicDataset(cleanDataFolder, dynamicDatasetList, idColumn='SCCRIP_ID'):
    dfList = []
    for i in dynamicDatasetList:
        fname = cleanDataFolder + i + '.sas7bdat'
        df_ = read_sas(fname)

        suffix = '_' + i
        df_ = df_.set_index(idColumn).add_suffix(suffix).reset_index()
        
        dfList.append(df_)
        
    df = dfList[0]
    dateCol_ = [col for col in df.columns if (df[col].dtype == 'datetime64[ns]' or df[col].dtype == 'datetime64[s]')]
    dateCol = dateCol_[0]
    df = df.rename(columns={dateCol: 'Date'})    
    
    for d in dfList[1:]:

        dDateCol_ = [col for col in d.columns if (d[col].dtype == 'datetime64[ns]' or d[col].dtype == 'datetime64[s]')]
        dDateCol = dDateCol_[0]
        d = d.rename(columns={dDateCol: 'Date'})        
        df = df.merge(d, on=[idColumn, 'Date'], how='outer')
        
    df = df.dropna(axis=1, how='all')        
    
    df = df.apply(lambda col: pd.to_numeric(col, errors='ignore') 
                  if col.dtypes == object 
                  else col, 
                  axis=0)
    return df


def convertDateToInt(df):
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            df[col]=pd.to_numeric(df[col], errors='coerce').astype('Int64')

    return None


def removeUncompleteId(df_background, df_dynamic,idColumn='SCCRIP_ID'):
    idSet = set(df_background[idColumn]).intersection(set(df_dynamic[idColumn]))
    
    df_background = df_background[df_background[idColumn].isin(idSet)]
    df_dynamic = df_dynamic[df_dynamic[idColumn].isin(idSet)]
    
    return df_background, df_dynamic


def convertDateTime(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_datetime(data[col])
            except ValueError:
                pass

def encode_object_columns(df):
    mapping_dicts = {}
    for c in df.columns:
        if c != 'SCCRIP_ID' and df[c].dtype == 'object':
            df[c], uniques = pd.factorize(df[c])
            t_dict = {value: index for index, value in enumerate(uniques)}
            
            mapping_dicts[c] = copy.deepcopy(t_dict)
            
    return mapping_dicts