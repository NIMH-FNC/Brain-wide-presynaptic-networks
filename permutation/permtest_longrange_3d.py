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


# reproducibility
random_state = 3121213
np.random.seed(random_state)
random.seed(random_state)

# permutation test - 2-Wasserstein on each type of long-ranged neurons
# Extended Figure 13

if __name__ == '__main__':

    # load neurons spatial location data
    dir_data = "./data/processed/combined/*.csv"
    group1, group2, merge_df, IDlist = load_data(dir_data)
    # load neuron class definition
    excit_class, inhib_class, longr_class, other, thal_dict = define_class()


    available_ID = group1 + group2
    resample_list, rlen = generate_resample(available_ID, group1, group2, nperm=10000)

    layer_list = ['S2', 'M1/M2', 'LR_Sen', 'cS1', 'BF', 'Fissure', 'FissureTotal']
    metric_list = []
    for (idx, layer) in enumerate(layer_list):
        metric = []
        for b in tqdm(range(rlen)):
            rg1 = resample_list[b][0]
            rg2 = resample_list[b][1]
            if layer == 'FissureTotal':
                wass = compute_metric(merge_df, rg1, rg2, ['Fissure','cFissure'], coordinate='atlas', metric='wass')
            else:
                wass = compute_metric(merge_df, rg1, rg2, [layer], coordinate='atlas', metric='wass')
            metric.append( np.concatenate(([b], [wass])) )
        
        column_names = ['index'] + ['wass']
        metric = pd.DataFrame(metric, columns = column_names)
        metric['marker'] = layer
        metric['index'] = metric['index'].astype('int')
        metric_list.append(metric)
            
    metric_df = pd.concat(metric_list)

    for layer in layer_list:
    
        df = metric_df.loc[metric_df['marker'] == layer]
        df = df[ ['index', 'marker', 'wass'] ]
        df = df.rename(columns={'wass':'2-Wasserstein'})
        nmetric = df.shape[1]-2
        metrics = list(df.columns)[2:]
    
        for (idx, m) in enumerate(metrics):
            layer_metric = df[m].values
            pval = (np.sum(layer_metric > layer_metric[0])+1)/(layer_metric.shape[0]+1)
            
            print(layer, 'p-value:', pval)
            print(layer, 'p-value correction:', np.minimum(1, pval*len(layer_list)))

    
    