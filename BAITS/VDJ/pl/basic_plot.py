import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

def _plot_cdr3_length_freq(df, chain_col,  cdr3_length='cdr3_length', figsize=(12,4)): 
    """
    Plot CDR3 length distribution for each chain type.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing CDR3 length information.
    chain_col : str
        Column specifying chain type (e.g. IGH, IGK, IGL).
    cdr3_length : str, default="cdr3_length"
        Column containing CDR3 length values.
    figsize : tuple, default=(12,4)
        Size of the matplotlib figure.

    Returns
    -------
    None
        Displays histogram plots of CDR3 length distributions.
    """
    fig, axes = plt.subplots(1,3,figsize=figsize)
    for i,chain in enumerate(list(set(df[chain_col]))): 
        sns.histplot(df.loc[df[chain_col]==chain, 'cdr3_length'], bins=20, kde=False, linewidth=0.5, ax=axes[i], palette='Pastel1')  
        axes[i].set_title(chain + ' CDR3 length') 
        axes[i].set_xlabel('Length of CDR3s') 
        axes[i].set_ylabel('Frequency') 
    plt.tight_layout() 


def _plot_bar(df, x, y, groupby=None, palette='Set2', xlabel=None, ylabel='Clone Richeness', ylog=False, ax=None, figsize=(4,3.5) ):
    """
    Create barplot with overlaid stripplot for grouped comparisons.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    x : str
        Column used for x-axis categories.
    y : str
        Column used for y-axis values.
    groupby : str, optional
        Column used for hue grouping.
    palette : str, default="Set2"
        Color palette used for plotting.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, default="Clone Richeness"
        Label for y-axis.
    ylog : bool, default=False
        Whether to use logarithmic scaling on y-axis.
    ax : matplotlib.axes.Axes, optional
        Axes object for plotting.
    figsize : tuple, default=(4,3.5)
        Figure size.

    Returns
    -------
    None
        Displays the barplot.
    """
    plt.figure(figsize=figsize)
    sns.barplot(y=y, x=x, data=df, hue=groupby, palette='Set2', alpha=0.5) 
    sns.stripplot(y=y, x=x, data=df, hue=groupby, dodge=True, jitter=0.2, palette='Pastel1')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 
    if ylog:
        plt.yscale('log')


def _boxplot(df, x, y, groupby=None, palette='Set2', xlabel=None, ylabel=None, log=False, ax=None ): 
    """
    Generate a boxplot for distribution visualization.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    x : str
        Column used for x-axis categories.
    y : str
        Column used for y-axis values.
    groupby : str, optional
        Column used for hue grouping.
    palette : str, default="Set2"
        Color palette.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    log : bool, default=False
        Whether to apply log transformation to y-axis.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot.

    Returns
    -------
    None
        Displays the boxplot.
    """
    sns.boxplot(data=df, x=x, y=y, hue=groupby, palette=palette, ax=ax ) 
    
    if ax:
        ax.set_xlabel(xlabel)  # 设置x轴标签
    if ax:
        ax.set_ylabel(ylabel)  # 设置y轴标签
    
    if log:
        plt.yscale('log')

    # if groupby is not None:
    #     plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def _qc_boxplot_clone(clone_df, clone_spatial_df, Cgene_col, group_by, figsize=(6, 3) ):
    """
    Plot QC boxplots for clone counts across groups.

    Parameters
    ----------
    clone_df : pandas.DataFrame
        Clone summary dataframe.
    clone_spatial_df : pandas.DataFrame
        Spatial clone summary dataframe.
    Cgene_col : str
        Column specifying constant gene class.
    group_by : str
        Column used for spatial grouping.
    figsize : tuple, default=(6,3)
        Figure size.

    Returns
    -------
    None
        Displays QC boxplots for clone metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize) 
    _boxplot(df=clone_df, x=Cgene_col, y='clone_by_group', palette='Pastel1', xlabel=None, ylabel='clone_by_group', log=False, ax=axes[0] ) 
    _boxplot(df=clone_spatial_df, x=group_by, y='clone_by_group_spatialLoc', groupby=Cgene_col, palette='Pastel1', xlabel=None, ylabel='clone_by_group_spatialLoc', log=True, ax=axes[1]) 
    plt.tight_layout() 
    plt.show() 

def _qc_boxplot_umis(umis_df, umis_spatial_df, Cgene_col, group_by, figsize=(6, 3) ):
    """
    Plot QC boxplots for UMI counts across groups.

    Parameters
    ----------
    umis_df : pandas.DataFrame
        UMI summary dataframe.
    umis_spatial_df : pandas.DataFrame
        Spatial UMI summary dataframe.
    Cgene_col : str
        Column specifying constant gene class.
    group_by : str
        Column used for spatial grouping.
    figsize : tuple, default=(6,3)
        Figure size.

    Returns
    -------
    None
        Displays QC boxplots for UMI metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize) 
    _boxplot(df=umis_df, x=Cgene_col, y='umis_by_group', palette='Pastel1', xlabel=None, ylabel='umis_by_group', log=True, ax=axes[0] ) 
    _boxplot(df=umis_spatial_df, x=group_by, y='umis_by_group_spatialLoc', groupby=Cgene_col, palette='Pastel1', xlabel=None, ylabel='umis_by_group_spatialLoc', log=True, ax=axes[1]) 
    plt.tight_layout() 
    plt.show() 


def _scatter_plot(df, x, y, groupby=None, palette='Pastel1', xlabel=None, ylabel=None, x_log=False, y_log=False, ax=None ): 
    """
    Generate scatter plot for two variables with optional grouping.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    x : str
        Column for x-axis.
    y : str
        Column for y-axis.
    groupby : str, optional
        Column used for color grouping.
    palette : str, default="Pastel1"
        Color palette.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    x_log : bool, default=False
        Whether to apply log scale on x-axis.
    y_log : bool, default=False
        Whether to apply log scale on y-axis.
    ax : matplotlib.axes.Axes, optional
        Axes object.

    Returns
    -------
    None
        Displays the scatter plot.
    """
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


def _plot_xcr(xcr_df, clone_col, loc_x_col, loc_y_col ):
    """
    Visualize spatial distribution of BCR/TCR clones.

    Each clone is encoded with a unique color and plotted according
    to its spatial coordinates.

    Parameters
    ----------
    xcr_df : pandas.DataFrame
        DataFrame containing clone and spatial coordinate information.
    clone_col : str
        Column specifying clone identity.
    loc_x_key : str
        Column containing x-coordinate.
    loc_y_key : str
        Column containing y-coordinate.

    Returns
    -------
    None
        Displays spatial scatter plot of clone locations.
    """
    xcr_df = xcr_df.sort_values(clone_col)
    xcr_df['clone_access'] = xcr_df[clone_col].astype('category').cat.codes 
    clone = list(xcr_df['clone_access'])

    x_min = xcr_df[loc_x_col].min(); x_max = xcr_df[loc_x_col].max()+1
    y_min = xcr_df[loc_y_col].min(); y_max = xcr_df[loc_y_col].max()+1

    x = list(xcr_df[loc_x_col]); y = list(xcr_df[loc_y_col])
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




