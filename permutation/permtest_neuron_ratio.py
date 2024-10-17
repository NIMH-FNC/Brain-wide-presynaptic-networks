# system & data loading
import os
import pandas as pd
import glob
import pickle
import json
import copy

# computational packages
import numpy as np
import scipy

import scikit_posthocs as sp

import ot
from pomegranate import *

# visualization
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
import IPython
from IPython.display import HTML

# misc
import time
import random
from tqdm import tqdm

# custom functions
from utils import *
from utils_load import load_data

seed = 31213
np.random.seed(seed)
random.seed(seed)

# permutation test -- neuron number ratio

if __name__ == '__main__':

    # create directory to store figures and permutation results
    savepath_plot = './output/visualization/permutation/'
    os.makedirs(savepath_plot, exist_ok=True)
    savepath_permutation = './output/permutation/'
    os.makedirs(savepath_permutation, exist_ok=True)

    
    # load neurons spatial location data
    dir_data = "./output/shortrange_layerprojection_test/*.csv"
    group1, group2, merge_df, IDlist = load_data(dir_data)
    # load neuron class definition
    excit_class, inhib_class, longr_class, other, thal_dict = define_class()

    ######################################################################
    # Figure 3e
    # permutation test - number of presynaptic neurons per brain
    
    available_ID = group1 + group2
    resample_list, rlen = generate_resample(available_ID, group1, group2, nperm=10000)
    T_ratio = np.zeros((rlen, 1))
    ncell_list = []
    
    for ID in available_ID:
        subj_df = merge_df.loc[ID]
        ncell_list.append(cell_count([ID], merge_df, base='all')[0])
    df_ncell = pd.DataFrame(np.asarray(ncell_list), columns = ['ncell'])
    df_ncell['ID'] = available_ID
    
    for b in tqdm(range(rlen)):
        rgroup1 = resample_list[b][0]
        rgroup2 = resample_list[b][1]
        group1_ncell = df_ncell.loc[df_ncell['ID'].isin(rgroup1)]['ncell'].values
        group2_ncell = df_ncell.loc[df_ncell['ID'].isin(rgroup2)]['ncell'].values
        T_ratio[b, 0] = group1_ncell.mean() - group2_ncell.mean()
    
    T_ratio = pd.DataFrame(T_ratio, columns = ['metric'])
    pval = (np.sum(np.abs(T_ratio['metric'].values) > np.abs(T_ratio['metric'].values[0]))+1)/(T_ratio['metric'].shape[0]+1)
        
    print(pval)

    ######################################################################
    # Figure 3f
    # permutation test - fraction of presynaptic neurons in wS1
    
    available_ID = group1 + group2
    resample_list, rlen = generate_resample(available_ID, group1, group2, nperm=10000)
    T_ratio = np.zeros((rlen, 1))
    subj_ratio = []
    
    for ID in available_ID:
        subj_df = merge_df.loc[ID]
        _, _, ncell, groupweight = load_groupROIlist_coord(
            [ID], 
            merge_df, 
            ROIlist=inhib_class+excit_class
        )
        base_ncell = cell_count([ID], merge_df, base='all')
        subj_ratio.append(ncell[0]/base_ncell[0])
        
    df_ratio = pd.DataFrame(np.asarray(subj_ratio), columns = ['ratio'])
    df_ratio['ID'] = available_ID
    
    for b in tqdm(range(rlen)):
        rgroup1 = resample_list[b][0]
        rgroup2 = resample_list[b][1]
        group1_subjRatio = df_ratio.loc[df_ratio['ID'].isin(rgroup1)]['ratio'].values.mean()
        group2_subjRatio = df_ratio.loc[df_ratio['ID'].isin(rgroup2)]['ratio'].values.mean()
        T_ratio[b, 0] = group1_subjRatio - group2_subjRatio
    
    T_ratio = pd.DataFrame(T_ratio, columns = ['metric'])
    pval = (np.sum(np.abs(T_ratio['metric'].values) > np.abs(T_ratio['metric'].values[0]))+1)/(T_ratio['metric'].shape[0]+1)
    
    print(pval)

    # Figure 3h, i
    # Laminar distribution of wS1 glutamatergic presynaptic neurons (L6 versus others)
    # Laminar distribution of wS1 GABAergic presynaptic neurons (L2/3 versus others)
    group1_remove02 = group1.copy()
    group1_remove02.remove('ana02')

    available_ID = group1_remove02 + group2
    resample_list, rlen = generate_resample(available_ID, group1_remove02, group2, nperm=10000)
    T_ratio = np.zeros((rlen, 1))
    
    layer_ncell_list = []
    
    for ID in available_ID:
        layer_ncell = []
        for (idx, l) in enumerate(excit_class):
            subj_df = merge_df.loc[ID]
            cell_number = subj_df.loc[ subj_df['marker'].isin([l]) ].shape[0]
            layer_ncell.append(cell_number)
        layer_ncell_list.append(layer_ncell)
        
    layer_ncell_list = np.vstack(layer_ncell_list)
    df_ratio = pd.DataFrame(layer_ncell_list, columns = excit_class)
    df_ratio['ID'] = available_ID
    
    stats, pval = scipy.stats.kruskal(
        df_ratio.iloc[:,1].values,
        df_ratio.iloc[:,2].values,
        df_ratio.iloc[:,3].values,
        df_ratio.iloc[:,4].values
    )
    print(pval)
    pval_adjust = sp.posthoc_dunn(
        [df_ratio.iloc[:,1].values,
         df_ratio.iloc[:,2].values,
         df_ratio.iloc[:,3].values,
         df_ratio.iloc[:,4].values],
        p_adjust = 'bonferroni'
    )
    pval_adjust.index = excit_class
    pval_adjust.columns = excit_class
    print(pval_adjust)

    available_ID = group1_remove02 + group2
    resample_list, rlen = generate_resample(available_ID, group1_remove02, group2, nperm=10000)
    T_ratio = np.zeros((rlen, 1))
    
    layer_ncell_list = []
    
    for ID in available_ID:
        layer_ncell = []
        for (idx, l) in enumerate(inhib_class):
            subj_df = merge_df.loc[ID]
            cell_number = subj_df.loc[ subj_df['marker'].isin([l]) ].shape[0]
            layer_ncell.append(cell_number)
        layer_ncell_list.append(layer_ncell)
        
    layer_ncell_list = np.vstack(layer_ncell_list)
    df_ratio = pd.DataFrame(layer_ncell_list, columns = inhib_class)
    df_ratio['ID'] = available_ID
    
    stats, pval = scipy.stats.kruskal(
        df_ratio.iloc[:,1].values,
        df_ratio.iloc[:,2].values,
        df_ratio.iloc[:,3].values,
        df_ratio.iloc[:,4].values
    )
    print(pval)
    pval_adjust = sp.posthoc_dunn(
        [df_ratio.iloc[:,1].values,
         df_ratio.iloc[:,2].values,
         df_ratio.iloc[:,3].values,
         df_ratio.iloc[:,4].values],
        p_adjust = 'bonferroni'
    )
    pval_adjust.index = inhib_class
    pval_adjust.columns = inhib_class
    print(pval_adjust)

    # Figure 3h, i permutation test
    layer_list = excit_class+inhib_class[:-1]
    nlayer = len(layer_list)
    fig = pyplot.figure(figsize=(nlayer*2, 2))
    ax_list = [fig.add_subplot(1,nlayer,i+1) for i in range(nlayer)]
    
    for (idx, layer) in enumerate(layer_list):
        available_ID, _, _, _ = load_groupROIlist_coord(IDlist, merge_df, ROIlist=[layer])
        resample_list, rlen = generate_resample(available_ID, group1_remove02, group2, nperm=10000)
    
        T_ratio = np.zeros((rlen, 1))
        subj_ratio = []
        for ID in available_ID:
            subj_df = merge_df.loc[ID]
            _, _, ncell, groupweight = load_groupROIlist_coord([ID], merge_df, ROIlist=[layer])
            base_ncell = cell_count([ID], merge_df, base='local')
            subj_ratio.append(ncell[0]/base_ncell[0])
    
        df_ratio = pd.DataFrame(np.asarray(subj_ratio), columns = ['ratio'])
        df_ratio['ID'] = available_ID
    
        for b in tqdm(range(rlen)):
            rgroup1 = resample_list[b][0]
            rgroup2 = resample_list[b][1]
            group1_subjRatio = df_ratio.loc[df_ratio['ID'].isin(rgroup1)]['ratio'].values.mean()
            group2_subjRatio = df_ratio.loc[df_ratio['ID'].isin(rgroup2)]['ratio'].values.mean()
            T_ratio[b, 0] = group1_subjRatio - group2_subjRatio
    
        T_ratio = pd.DataFrame(T_ratio, columns = ['metric'])
        
        pval = (np.sum(np.abs(T_ratio['metric'].values) > np.abs(T_ratio['metric'].values[0]))+1)/(T_ratio['metric'].shape[0]+1)
        
        sns.histplot(T_ratio['metric'].values, ax=ax_list[idx], bins=30, element='step', alpha=0.3, color='black')
        bottom, top = ax_list[idx].get_ylim()
        baseline = T_ratio['metric'].values[0]
        ax_list[idx].vlines(baseline, bottom, top, colors='r')
        if idx == 0:
            ax_list[idx].text(baseline, top/6, 'observation', c='r', rotation=270, fontsize=12)
        title = '{} \n p-val={:.2e}'.format(layer, pval)
        ax_list[idx].set_title(title, fontsize=10)
        ax_list[idx].set_xticks([])
        if idx != 0:
            ax_list[idx].set_ylabel(None)
        ax_list[idx].set_yticks([])
        ax_list[idx].set_yticklabels([])
    
    fig.subplots_adjust(wspace=0, hspace=0)
    pyplot.tight_layout()
    
    pyplot.savefig(os.path.join(savepath_plot, 'shortr_cellnumber_ratio2shortr_permutation_02remove.png'))
    pyplot.savefig(os.path.join(savepath_plot, 'shortr_cellnumber_ratio2shortr_permutation_02remove.eps'), format='eps')
    pyplot.savefig(os.path.join(savepath_plot, 'shortr_cellnumber_ratio2shortr_permutation_02remove.svg'), format='svg')


    # Figure 4 permutation test on cell ratio (long ranged cell)
    layer_list = ['M1/M2',
               'S2',
               'LR_Sen',
               'Thal_VPM/VPL', 'Thal_PO',
               'Thal_VA/VL',
               'Fissure',
               'cS1',
               'BF']
    nlayer = len(layer_list)
    fig = pyplot.figure(figsize=(nlayer*2, 2))
    ax_list = [fig.add_subplot(1,9,i+1) for i in range(nlayer)]
    
    for (idx, layer) in enumerate(layer_list):
    
        available_ID, _, _, _ = load_groupROIlist_coord(IDlist, merge_df, ROIlist=[layer])
        resample_list, rlen = generate_resample(available_ID, group1, group2, nperm=10000)
    
        T_ratio = np.zeros((rlen, 1))
        subj_ratio = []
    
        for ID in available_ID:
            subj_df = merge_df.loc[ID]
            _, _, ncell, groupweight = load_groupROIlist_coord([ID], merge_df, ROIlist=[layer])
            base_ncell = cell_count([ID], merge_df, base='longr')
            subj_ratio.append(ncell[0]/base_ncell[0])
    
        df_ratio = pd.DataFrame(np.asarray(subj_ratio), columns = ['ratio'])
        df_ratio['ID'] = available_ID
    
        for b in tqdm(range(rlen)):
            rgroup1 = resample_list[b][0]
            rgroup2 = resample_list[b][1]
    
            group1_subjRatio = df_ratio.loc[df_ratio['ID'].isin(rgroup1)]['ratio'].values.mean()
            group2_subjRatio = df_ratio.loc[df_ratio['ID'].isin(rgroup2)]['ratio'].values.mean()
            T_ratio[b, 0] = group1_subjRatio - group2_subjRatio
    
        T_ratio = pd.DataFrame(T_ratio, columns = ['metric'])
        
        pval = (np.sum(np.abs(T_ratio['metric'].values) > np.abs(T_ratio['metric'].values[0]))+1)/(T_ratio['metric'].shape[0]+1)
        
        sns.histplot(T_ratio['metric'].values, ax=ax_list[idx], bins=30, element='step', alpha=0.3, color='black')
        bottom, top = ax_list[idx].get_ylim()
        baseline = T_ratio['metric'].values[0]
        ax_list[idx].vlines(baseline, bottom, top, colors='r')
        if (idx == 0):
            ax_list[idx].text(baseline*0.7,top/6, 'observation', c='r', rotation=270, fontsize=12)
        if (idx == 9):
            ax_list[idx].text(baseline, top/6, 'observation', c='r', rotation=270, fontsize=12)
        title = '{} \n p-val={:.2e}'.format(layer, pval)
        ax_list[idx].set_title(title, fontsize=10)
        ax_list[idx].set_xticks([])
        if (idx != 0):
            ax_list[idx].set_ylabel(None)
        ax_list[idx].set_yticks([])
        ax_list[idx].set_yticklabels([])
    
    fig.subplots_adjust(wspace=0, hspace=0.3)
    pyplot.tight_layout()
    pyplot.savefig(os.path.join(savepath_plot, 'longr_cellnumber_ratio2longr_permutation.png'))
    pyplot.savefig(os.path.join(savepath_plot, 'longr_cellnumber_ratio2longr_permutation.eps'), format='eps')
    pyplot.savefig(os.path.join(savepath_plot, 'longr_cellnumber_ratio2longr_permutation.svg'), format='svg')

    sensory_class = thal_dict['sensory']
    anterior_class = thal_dict['anterior']
    PO_class = thal_dict['PO']
    VPM_class = thal_dict['VPM']
    main_class = thal_dict['main']

    class_list = [sensory_class, anterior_class, PO_class, VPM_class, main_class]
    class_name = ['Sensory', 'Anterior', 'PO', 'VPM', 'Main']
    nclass = len(class_list)
    fig = pyplot.figure(figsize=(nclass*2, 2))
    ax_list = [fig.add_subplot(1,5,i+1) for i in range(nclass)]
    
    for (idx, layers) in enumerate(class_list):
    
        available_ID, _, _, _ = load_groupROIlist_coord(IDlist, merge_df, ROIlist=layers)
        resample_list, rlen = generate_resample(available_ID, group1, group2, nperm=10000)
    
        T_ratio = np.zeros((rlen, 1))
        subj_ratio = []
    
        for ID in available_ID:
            subj_df = merge_df.loc[ID]
            _, _, ncell, groupweight = load_groupROIlist_coord([ID], merge_df, ROIlist=layers)
            base_ncell = cell_count([ID], merge_df, base='longr')
            subj_ratio.append(ncell[0]/base_ncell[0])
    
        df_ratio = pd.DataFrame(np.asarray(subj_ratio), columns = ['ratio'])
        df_ratio['ID'] = available_ID
    
        for b in tqdm(range(rlen)):
            rgroup1 = resample_list[b][0]
            rgroup2 = resample_list[b][1]
    
            group1_subjRatio = df_ratio.loc[df_ratio['ID'].isin(rgroup1)]['ratio'].values.mean()
            group2_subjRatio = df_ratio.loc[df_ratio['ID'].isin(rgroup2)]['ratio'].values.mean()
            T_ratio[b, 0] = group1_subjRatio - group2_subjRatio
    
        T_ratio = pd.DataFrame(T_ratio, columns = ['metric'])
        
        pval = (np.sum(np.abs(T_ratio['metric'].values) > np.abs(T_ratio['metric'].values[0]))+1)/(T_ratio['metric'].shape[0]+1)
        
        sns.histplot(T_ratio['metric'].values, ax=ax_list[idx], bins=30, element='step', alpha=0.3, color='black')
        bottom, top = ax_list[idx].get_ylim()
        baseline = T_ratio['metric'].values[0]
        ax_list[idx].vlines(baseline, bottom, top, colors='r')
        if (idx == 0):
            ax_list[idx].text(baseline,top/6, 'observation', c='r', rotation=270, fontsize=12)
        if (idx == 9):
            ax_list[idx].text(baseline, top/6, 'observation', c='r', rotation=270, fontsize=12)
        title = '{} \n p-val={:.2e}'.format(class_name[idx], pval)
        ax_list[idx].set_title(title, fontsize=10)
        ax_list[idx].set_xticks([])
        if (idx != 0):
            ax_list[idx].set_ylabel(None)
        ax_list[idx].set_yticks([])
        ax_list[idx].set_yticklabels([])

    fig.subplots_adjust(wspace=0, hspace=0.0)
    
    pyplot.tight_layout()
    
    pyplot.savefig(os.path.join(savepath_plot, 'Thal_ratio2longr_permutation.png'))
    pyplot.savefig(os.path.join(savepath_plot, 'Thal_ratio2longr_permutation.eps'), format='eps')
    pyplot.savefig(os.path.join(savepath_plot, 'Thal_ratio2longr_permutation.svg'), format='svg')


    # Figure 10 - permutation test on cell ratio (total cell)
    class_list = [excit_class, inhib_class, longr_class, other]
    class_name = ['Excitatory', 'Inhibitory', 'Long-ranged', 'Other']
    nclass = len(class_list)
    fig = pyplot.figure(figsize=(nclass*2, 2))
    ax_list = [fig.add_subplot(1,4,i+1) for i in range(nclass)]
    
    for (idx, layers) in enumerate(class_list):
    
        available_ID, _, _, _ = load_groupROIlist_coord(IDlist, merge_df, ROIlist=layers)
        resample_list, rlen = generate_resample(available_ID, group1, group2, nperm=10000)
    
        T_ratio = np.zeros((rlen, 1))
        subj_ratio = []
    
        for ID in available_ID:
            subj_df = merge_df.loc[ID]
            _, _, ncell, groupweight = load_groupROIlist_coord([ID], merge_df, ROIlist=layers)
            base_ncell = cell_count([ID], merge_df, base='all')
            subj_ratio.append(ncell[0]/base_ncell[0])
    
        df_ratio = pd.DataFrame(np.asarray(subj_ratio), columns = ['ratio'])
        df_ratio['ID'] = available_ID
    
        for b in tqdm(range(rlen)):
            rgroup1 = resample_list[b][0]
            rgroup2 = resample_list[b][1]
    
            group1_subjRatio = df_ratio.loc[df_ratio['ID'].isin(rgroup1)]['ratio'].values.mean()
            group2_subjRatio = df_ratio.loc[df_ratio['ID'].isin(rgroup2)]['ratio'].values.mean()
            T_ratio[b, 0] = group1_subjRatio - group2_subjRatio
    
        T_ratio = pd.DataFrame(T_ratio, columns = ['metric'])
        
        pval = (np.sum(np.abs(T_ratio['metric'].values) > np.abs(T_ratio['metric'].values[0]))+1)/(T_ratio['metric'].shape[0]+1)
        
        sns.histplot(T_ratio['metric'].values, ax=ax_list[idx], bins=30, element='step', alpha=0.3, color='black')
        bottom, top = ax_list[idx].get_ylim()
        baseline = T_ratio['metric'].values[0]
        ax_list[idx].vlines(baseline, bottom, top, colors='r')
        if (idx == 0):
            ax_list[idx].text(baseline,top/6, 'observation', c='r', rotation=270, fontsize=12)
        if (idx == 9):
            ax_list[idx].text(baseline, top/6, 'observation', c='r', rotation=270, fontsize=12)
        title = '{} \n p-val={:.2e}'.format(class_name[idx], pval)
        ax_list[idx].set_title(title, fontsize=10)
        ax_list[idx].set_xticks([])
        if (idx != 0):
            ax_list[idx].set_ylabel(None)
        ax_list[idx].set_yticks([])
        ax_list[idx].set_yticklabels([])
    

    fig.subplots_adjust(wspace=0, hspace=0.3)
    
    pyplot.tight_layout()
    pyplot.savefig(os.path.join(savepath_plot, 'class_cellnumber_ratio2all_permutation.png'))
    pyplot.savefig(os.path.join(savepath_plot, 'class_cellnumber_ratio2all_permutation.eps'), format='eps')
    pyplot.savefig(os.path.join(savepath_plot, 'class_cellnumber_ratio2all_permutation.svg'), format='svg')
        
    
    