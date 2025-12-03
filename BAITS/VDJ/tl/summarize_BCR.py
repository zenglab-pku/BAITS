import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pl.basic_plot import _plot_bar

from .bcr_desc import compute_index 

def stat_clone(df, groupby, Cgene_col, clone_col, plot=True, palette='Set2', xlabel=None, ylabel=None, ylog=False, figsize=(4,3.5) ):
    y_name = 'clone_by_'+groupby
    clone_df = df[[groupby, Cgene_col, clone_col]].drop_duplicates().groupby([groupby, Cgene_col]).size().reset_index(name = y_name)
    
    if plot:
        _plot_bar(clone_df, Cgene_col, y_name, groupby=Cgene_col, palette='Set2', xlabel=None, ylabel=None, ylog=False, figsize=(4,3.5) ) 

    cloneDict = clone_df.set_index([groupby, Cgene_col])['clone_by_'+groupby].to_dict() 
    df[y_name] = df.apply(lambda row: cloneDict.get((row[groupby], row[Cgene_col]), None), axis=1) 

    return df


def aggregate_clone_df(df, group_by, Cgene_col, clone_col, groups, count_basis='location', loc_x_col='X', loc_y_col='Y', Umi_col='UMI'):
    if count_basis == 'location':
        lst = list(set([group_by, Cgene_col, clone_col, loc_x_col, loc_y_col] + groups))
        loc_df = df[lst].drop_duplicates() 
        _Index_compute_count = loc_df[ groups+[clone_col] ].groupby(groups)[clone_col].value_counts().reset_index(name='count') 
        _Index_compute_freq = loc_df[ groups+[clone_col] ].groupby(groups)[clone_col].value_counts(normalize=True).reset_index(name='freq') 
        _Index_compute = pd.merge(_Index_compute_freq, _Index_compute_count, on = groups+[clone_col] ) 
        return _Index_compute
        
    if count_basis == 'UMI': 
        _Index_compute_count = df[ groups+[clone_col] ].groupby(groups)[clone_col].value_counts().reset_index(name='count') 
        _Index_compute_freq = df[ groups+[clone_col] ].groupby(groups)[clone_col].value_counts(normalize=True).reset_index(name='freq') 
        _Index_compute = pd.merge(_Index_compute_freq, _Index_compute_count, on = groups+[clone_col] ) 
        return _Index_compute


def compute_grouped_index(df, group_by, Cgene_col, clone_col, groups, count_basis='location', loc_x_col='X', loc_y_col='Y', Umi_col=None, index='shannon_entropy'):
    if count_basis=='location':
        _Index_compute = aggregate_clone_df(df, group_by, Cgene_col, clone_col, groups, count_basis=count_basis, loc_x_col='X', loc_y_col='Y').copy()
    if count_basis=='UMI':
        _Index_compute = aggregate_clone_df(df, group_by, Cgene_col, clone_col, groups, count_basis=count_basis, Umi_col=Umi_col).copy()

    tmp_df = _Index_compute.groupby(groups)['freq'].apply(lambda x: compute_index(index, x))
    
    if index != 'renyi_entropy':
        tmp_df = tmp_df.reset_index(name=index).dropna(subset=[index])
    else: 
        tmp_df = tmp_df.melt(ignore_index=False, var_name='alpha', value_name = index).reset_index() 
        cols = list(range(0, len(groups))) + list(range(len(groups)+1, len(tmp_df.columns))) 
        tmp_df = tmp_df.iloc[:, cols ]
    
    return tmp_df
















