import numpy as np 
import pandas as pd
import seaborn as sns
from pl.basic_plot import _qc_boxplot_clone, _qc_boxplot_umis, _plot_cdr3_length_freq

def calculate_cdr3_length(df, sample_col, Cgene_col, cdr3_col, cdr3_type='nt', plot=True, figsize=(9,3)):
    clone_length_df = df[[sample_col, Cgene_col, cdr3_col]].drop_duplicates().assign(cdr3_length=lambda df: df[cdr3_col].str.len())[[sample_col, Cgene_col, cdr3_col, 'cdr3_length']]

    if cdr3_type=='nt':
        clone_length_df['cdr3_length'] /= 3 

    clone_length_df['cdr3_length'] = clone_length_df['cdr3_length'].astype(int)
    clone_length_dict = clone_length_df[[cdr3_col, 'cdr3_length']].set_index([cdr3_col])['cdr3_length'].to_dict()
    df['cdr3_length'] = df.apply(lambda row: clone_length_dict.get((row[cdr3_col]), None), axis=1) 

    if plot:
        _plot_cdr3_length_freq(clone_length_df, Cgene_col, cdr3_length='cdr3_length', figsize=figsize)

    return df



def calculate_qc_clones(df, group_by, Cgene_col, clone_col, loc_x_col='X', loc_y_col='Y', plot=True): 
    clone_df = df[[group_by, Cgene_col, clone_col ]].drop_duplicates().groupby([group_by, Cgene_col ]).size().reset_index(name='clone_by_group')  
    clone_spatial_df = df[[group_by, Cgene_col, clone_col, loc_x_col, loc_y_col]].drop_duplicates().groupby([group_by, Cgene_col, loc_x_col, loc_y_col]).size().reset_index(name='clone_by_group_spatialLoc') 
    
    if plot:
        _qc_boxplot_clone( clone_df, clone_spatial_df, Cgene_col, group_by, figsize=(7, 3.5) )
    cloneDict = clone_df.set_index([group_by, Cgene_col])['clone_by_group'].to_dict()
    clonesSpatialDict = clone_spatial_df.set_index([group_by, Cgene_col, loc_x_col, loc_y_col])['clone_by_group_spatialLoc'].to_dict()
    df['clone_by_group'] = df.apply(lambda row: cloneDict.get((row[group_by], row[Cgene_col]), None), axis=1) 
    df['clone_by_group_spatialLoc'] = df.apply(lambda row: clonesSpatialDict.get((row[group_by], row[Cgene_col], row[loc_x_col], row[loc_y_col]), None), axis=1) 
    return df


def calculate_qc_umis(df, group_by, Cgene_col, clone_col, loc_x_col='X', loc_y_col='Y', plot=True, figsize=(7, 3.5)): 
    umis_df = df[[group_by, Cgene_col, clone_col ]].groupby([group_by, Cgene_col ]).size().reset_index(name='umis_by_group')  
    umis_spatial_df = df[[group_by, Cgene_col, clone_col, loc_x_col, loc_y_col]].groupby([group_by, Cgene_col, loc_x_col, loc_y_col]).size().reset_index(name='umis_by_group_spatialLoc') 

    if plot:
        _qc_boxplot_umis( umis_df, umis_spatial_df, Cgene_col, group_by, figsize=(7, 3.5) )
    umisDict = umis_df.set_index([group_by, Cgene_col])['umis_by_group'].to_dict()
    umisSpatialDict = umis_spatial_df.set_index([group_by, Cgene_col, loc_x_col, loc_y_col])['umis_by_group_spatialLoc'].to_dict()
    df['umis_by_group'] = df.apply(lambda row: umisDict.get((row[group_by], row[Cgene_col]), None), axis=1) 
    df['umis_by_group_spatialLoc'] = df.apply(lambda row: umisSpatialDict.get((row[group_by], row[Cgene_col], row[loc_x_col], row[loc_y_col]), None), axis=1) 
    return df


def filter_clones(df, clone_col, min_clone=1): 
    return df[df[clone_col] > min_clone]

def filter_clones_spatial(df, clone_spatial_col, min_clone_spatial=1): 
    return df[df[clone_spatial_col] > min_clone_spatial]

def filter_umi(df, umi_key, min_umi=1): 
    return df[df[umi_key] > min_umi]

def filter_umi_spatial(df, clone_umi_key, min_umi_spatial=1): 
    return df[df[clone_umi_key] > min_umi_spatial]


