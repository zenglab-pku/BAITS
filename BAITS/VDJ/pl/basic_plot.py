import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

def _plot_bar(df, x, y, groupby=None, palette='Set2', xlabel=None, ylabel='Clone Richeness', ylog=False, ax=None, figsize=(4,3.5) ):
    plt.figure(figsize=figsize)
    sns.barplot(y=y, x=x, data=df, hue=groupby, palette='Set2', alpha=0.5) 
    sns.stripplot(y=y, x=x, data=df, hue=groupby, dodge=True, jitter=0.2, palette='Set2')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 
    if ylog:
        plt.yscale('log')


def _boxplot(df, x, y, groupby=None, palette='Set2', xlabel=None, ylabel=None, log=False, ax=None ): 

    sns.boxplot(data=df, x=x, y=y, hue=groupby, palette=palette, ax=ax ) 
    
    if ax:
        ax.set_xlabel(xlabel)  # 设置x轴标签
    if ax:
        ax.set_ylabel(ylabel)  # 设置y轴标签
    
    if log:
        plt.yscale('log')

    if groupby is not None:
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def _qc_boxplot_clone(clone_df, clone_spatial_df, Cgene_key, group_by, figsize=(6, 3) ):
    fig, axes = plt.subplots(1, 2, figsize=figsize) 
    _boxplot(df=clone_df, x=Cgene_key, y='clone_by_group', palette='Set2', xlabel=None, ylabel='clone_by_group', log=False, ax=axes[0] ) 
    _boxplot(df=clone_spatial_df, x=group_by, y='clone_by_group_spatialLoc', groupby=Cgene_key, palette='Set2', xlabel=None, ylabel='clone_by_group_spatialLoc', log=True, ax=axes[1]) 
    plt.tight_layout() 
    plt.show() 

def _qc_boxplot_umis(umis_df, umis_spatial_df, Cgene_key, group_by, figsize=(6, 3) ):
    fig, axes = plt.subplots(1, 2, figsize=figsize) 
    _boxplot(df=umis_df, x=Cgene_key, y='umis_by_group', palette='Set2', xlabel=None, ylabel='umis_by_group', log=True, ax=axes[0] ) 
    _boxplot(df=umis_spatial_df, x=group_by, y='umis_by_group_spatialLoc', groupby=Cgene_key, palette='Set2', xlabel=None, ylabel='umis_by_group_spatialLoc', log=True, ax=axes[1]) 
    plt.tight_layout() 
    plt.show() 


def _scatter_plot(df, x, y, groupby=None, palette='Set2', xlabel=None, ylabel=None, x_log=False, y_log=False, ax=None ): 

    sns.scatterplot( data=df, x=x, y=y, hue=groupby, palette=palette, edgecolor='black', ax=ax ) 
    
    if ax:
        ax.set_xlabel(xlabel)  # 设置x轴标签
    if ax:
        ax.set_ylabel(ylabel)  # 设置y轴标签
    
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    if groupby is not None:
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def _plot_xcr(xcr_df, clone_col, loc_x_key, loc_y_key ):
    xcr_df = xcr_df.sort_values(clone_col)
    xcr_df['clone_access'] = xcr_df[clone_col].astype('category').cat.codes 
    clone = list(xcr_df['clone_access'])

    x_min = xcr_df[loc_x_key].min(); x_max = xcr_df[loc_x_key].max()+1
    y_min = xcr_df[loc_y_key].min(); y_max = xcr_df[loc_y_key].max()+1

    x = list(xcr_df[loc_x_key]); y = list(xcr_df[loc_y_key])
    xcr_mat = np.zeros(( x_max+1, y_max+1 ))
    xcr_mat_c = np.zeros(( x_max+1, y_max+1 )) 
    for i in range(len(x)):
        xcr_mat[x[i],y[i]]+=1 
        xcr_mat_c[x[i],y[i]] = clone[i]
    xcr_row, xcr_col = np.where(xcr_mat )
    xcr_size = xcr_mat[xcr_row, xcr_col] * 0.3 / np.percentile(xcr_mat[xcr_mat!=0].flatten(),0.9)
    xcr_size = np.clip(xcr_size,0,1) * 3
    xcr_color = xcr_mat_c[xcr_row, xcr_col]

    fig, ax = plt.subplots(1,1, figsize=((y_max-y_min)*3/10000, (x_max-x_min)*3/10000 ))     
    ax.scatter(xcr_col, xcr_row, s=xcr_size, c=xcr_color, marker='o',cmap='coolwarm')
    ax.invert_yaxis()
    plt.show()




