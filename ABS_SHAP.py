import matplotlib as plt
import seaborn as sns
import pandas as pd
import numpy as np

def ABS_SHAP(df_shap,df,figure_name, max_display=10, figsize=(8,8)):
    #import matplotlib as plt
    # Make a copy of the input data
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
    k2 = k1.tail(max_display)
    colorlist = k2['Sign']
    fig=plt.figure(figsize=figsize)
    sns.set_theme(font_scale=1, style='whitegrid')
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=figsize,legend=False)
    ax.set_xlabel("SHAP Value (GREEN = Positive Impact)")
    plt.tight_layout()
    plt.savefig(figure_name)
    plt.close(fig)
    return k1