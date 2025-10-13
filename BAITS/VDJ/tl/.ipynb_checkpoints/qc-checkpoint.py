import numpy as np 
import pandas as pd
import seaborn as sns
from pl.basic_plot import _qc_boxplot_clone, _qc_boxplot_umis

def calculate_qc_clones(df, group_by, Cgene_key, clone_key,loc_x_key='X', loc_y_key='Y', plot=True): 
    clone_df = df[[group_by, Cgene_key, clone_key ]].drop_duplicates().groupby([group_by, Cgene_key ]).size().reset_index(name='clone_by_group')  
    clone_spatial_df = df[[group_by, Cgene_key, clone_key, loc_x_key, loc_y_key]].drop_duplicates().groupby([group_by, Cgene_key, loc_x_key, loc_y_key]).size().reset_index(name='clone_by_group_spatialLoc') 
    
    if plot:
        _qc_boxplot_clone( clone_df, clone_spatial_df, Cgene_key, group_by, figsize=(7, 3.5) )
    cloneDict = clone_df.set_index([group_by, Cgene_key])['clone_by_group'].to_dict()
    clonesSpatialDict = clone_spatial_df.set_index([group_by, Cgene_key, loc_x_key, loc_y_key])['clone_by_group_spatialLoc'].to_dict()
    df['clone_by_group'] = df.apply(lambda row: cloneDict.get((row[group_by], row[Cgene_key]), None), axis=1) 
    df['clone_by_group_spatialLoc'] = df.apply(lambda row: clonesSpatialDict.get((row[group_by], row[Cgene_key], row[loc_x_key], row[loc_y_key]), None), axis=1) 
    return df


def calculate_qc_umis(df, group_by, Cgene_key, clone_key, loc_x_key='X', loc_y_key='Y', plot=True, figsize=(7, 3.5)): 
    umis_df = df[[group_by, Cgene_key, clone_key ]].groupby([group_by, Cgene_key ]).size().reset_index(name='umis_by_group')  
    umis_spatial_df = df[[group_by, Cgene_key, clone_key, loc_x_key, loc_y_key]].groupby([group_by, Cgene_key, loc_x_key, loc_y_key]).size().reset_index(name='umis_by_group_spatialLoc') 

    if plot:
        _qc_boxplot_umis( umis_df, umis_spatial_df, Cgene_key, group_by, figsize=(7, 3.5) )
    umisDict = umis_df.set_index([group_by, Cgene_key])['umis_by_group'].to_dict()
    umisSpatialDict = umis_spatial_df.set_index([group_by, Cgene_key, loc_x_key, loc_y_key])['umis_by_group_spatialLoc'].to_dict()
    df['umis_by_group'] = df.apply(lambda row: umisDict.get((row[group_by], row[Cgene_key]), None), axis=1) 
    df['umis_by_group_spatialLoc'] = df.apply(lambda row: umisSpatialDict.get((row[group_by], row[Cgene_key], row[loc_x_key], row[loc_y_key]), None), axis=1) 
    return df


def filter_clones(df, clone_key, min_clone=1): 
    return df[df[clone_key] > min_clone]

def filter_clones_spatial(df, clone_spatial_key, min_clone_spatial=1): 
    return df[df[clone_spatial_key] > min_clone_spatial]

def filter_umi(df, umi_key, min_umi=1): 
    return df[df[umi_key] > min_umi]

def filter_umi_spatial(df, clone_umi_key, min_umi_spatial=1): 
    return df[df[clone_umi_key] > min_umi_spatial]


