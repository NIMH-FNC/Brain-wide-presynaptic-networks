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
from utils_load import load_data, load_groupROIlist_coord

seed = 31213
np.random.seed(seed)
random.seed(seed)

# permutation test - Figure 3j wS1 GABAergic presynaptic neurons 

if __name__ == '__main__':

    # define neuron classes
    excit_class, inhib_class, longr_class, other, thal_dict = define_class()
    color_code = ['red', 'green', 'purple', 'orange', 'blue']

    # load neurons spatial location data
    dir_data = "./output/shortrange_layerprojection/*.csv"
    group1, group2, merge_df, IDlist = load_data(dir_data)


    available_ID = group1 + group2
    resample_list, rlen = generate_resample(available_ID, group1, group2, nperm=100)

    metric_list = []
    
    for (idx, layer) in enumerate(inhib_class[:-1]):
        metric = []
        for b in tqdm(range(rlen)):
            rg1 = resample_list[b][0]
            rg2 = resample_list[b][1]
            wass = compute_metric(merge_df, rg1, rg2, [layer], metric='wass')
            metric.append( np.concatenate(([b], [wass])) )
        
        column_names = ['index'] + ['wass']
        metric = pd.DataFrame(metric, columns = column_names)
        metric['marker'] = layer
        metric['index'] = metric['index'].astype('int')
        metric_list.append(metric)
            
    metric_df = pd.concat(metric_list)


    nlayer = len(inhib_class[1:-1])

    for (idx, layer) in enumerate(inhib_class[1:-1]):
        
        df = metric_df.loc[metric_df['marker'] == layer]
        df = df[ ['index', 'marker', 'wass'] ]
        df = df.rename(columns={'wass':'2-Wasserstein'})
        layer_metric = df['2-Wasserstein'].values
        pval = (np.sum(layer_metric > layer_metric[0])+1)/(layer_metric.shape[0]+1)

        print(layer, ', p-value: ', pval)
        


                     

