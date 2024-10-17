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
import sklearn
import open3d as o3d

import itertools
from itertools import permutations, combinations

from scipy import stats
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import distance
from numpy.linalg import eig, inv
from sklearn.neighbors import KernelDensity

import ot
from pomegranate import *

# misc
import time
import random
from tqdm import tqdm

# custom functions
from utils import *
from utils_load import load_data, load_groupROIlist_coord



# visualization
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
import IPython
from IPython.display import HTML

import matplotlib
import matplotlib.offsetbox
from matplotlib.lines import Line2D

class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """
    def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None, 
                 frameon=True, linekw={}, **kwargs):
        if not ax:
            ax = pyplot.gca()
        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0,size],[0,0], **linekw)
        vline1 = Line2D([0,0],[-extent/2.,extent/2.], **linekw)
        vline2 = Line2D([size,size],[-extent/2.,extent/2.], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        # txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False)
        txt = matplotlib.offsetbox.TextArea(label)
        self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar,txt],  
                                 align="center", pad=ppad, sep=sep) 
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad, 
                 borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon,
                 **kwargs)



seed = 31213
np.random.seed(seed)
random.seed(seed)

# Plot generation and permutation test for smaller spatial dispersion of L4 to L2/3 and L5
# Extended figure 11

if __name__ == '__main__':

    # load neuron class definition
    excit_class, inhib_class, longr_class, other, thal_dict = define_class()
    
    color_code = ['red', 'green', 'purple', 'orange', 'blue']

    # create directory to store figures and permutation results
    savepath_plot = './output/visualization/'
    os.makedirs(savepath_plot, exist_ok=True)

    # load neurons spatial location data
    dir_data = "./output/shortrange_layerprojection/*.csv"
    group1, group2, merge_df, IDlist = load_data(dir_data)


    
    fig = pyplot.figure(figsize=(10, 3))
    ax_list = [fig.add_subplot(1,2,i+1) for i in range(2)]
    
    group1_det_trend = []
    group2_det_trend = []
    
    group1_etrend = []
    group2_etrend = []
    for (idx, ID) in enumerate(group1+group2):
        trend = []
        etrend = []
        for (i, layer) in enumerate(excit_class[:-1]):
            subj_df = merge_df.loc[ID]
            xyz = subj_df.loc[subj_df['marker'] == layer][['ppx','ppy']].values
            if xyz.shape[0] > 5:
                kde = stats.gaussian_kde(xyz.T)
                trend.append(np.linalg.det(kde.covariance / (kde.factor**2)))
                
                eigs, _ = np.linalg.eig(kde.covariance / (kde.factor**2))
                etrend.append(np.max(eigs))
                # print(kde.factor)
            else:
                trend.append(np.nan)
                etrend.append(np.nan)
                
        trend = np.array(trend)
        trend = trend/np.nanmax(trend)
        
        etrend = np.array(etrend)
        etrend = etrend/np.nanmax(etrend)
            
        print(ID, np.sign((trend[1]-trend[0])*(trend[2]-trend[1])),
              (trend[1]-trend[0])*(trend[2]-trend[1]), trend)
        
        if ID in group1:
            color = 'r'
            group1_det_trend.append(trend)
            group1_etrend.append(etrend)
        else:
            color = 'b'
            group2_det_trend.append(trend)
            group2_etrend.append(etrend)
            
        ax_list[0].plot(trend, linewidth=1, alpha=0.3, c=color)
        ax_list[1].plot(etrend, linewidth=1, alpha=0.3, c=color)
    
    group1_trend = np.mean(np.vstack((group1_det_trend)), axis=0)
    group2_trend = np.mean(np.vstack((group2_det_trend)), axis=0)
    
    group1_etrend = np.mean(np.vstack((group1_etrend)), axis=0)
    group2_etrend = np.mean(np.vstack((group2_etrend)), axis=0)
    
    ax_list[0].plot(group1_trend, linewidth=5, alpha=0.5, c='r', label='Group 1 mean')
    ax_list[0].plot(group2_trend, linewidth=5, alpha=0.5, c='b', label='Group 2 mean')
    
    ax_list[1].plot(group1_etrend, linewidth=5, alpha=0.5, c='r', label='Group 1 mean')
    ax_list[1].plot(group2_etrend, linewidth=5, alpha=0.5, c='b', label='Group 2 mean')
    
    ax_list[0].set_xticks(np.arange(len(excit_class[:-1])))
    ax_list[0].set_xticklabels(excit_class[:-1])
    
    ax_list[1].set_xticks(np.arange(len(excit_class[:-1])))
    ax_list[1].set_xticklabels(excit_class[:-1])
    
    ax_list[0].legend(loc='lower right')
    ax_list[1].legend(loc='lower right')
    
    ax_list[0].grid()
    ax_list[1].grid()
    # pyplot.show()
    
    pyplot.savefig(os.path.join(savepath_plot, 'spread_trend.png'))
    pyplot.savefig(os.path.join(savepath_plot, 'spread_trend.eps'), format='eps')
    pyplot.savefig(os.path.join(savepath_plot, 'spread_trend.svg'), format='svg')



    # Generate extension Figure 10

    nsubj = len(group1)
    # fig = pyplot.figure(figsize=(8, 2*nsubj))
    # ax_list = [fig.add_subplot(nsubj,3,i+1) for i in range(nsubj)]
    fig, axs = pyplot.subplots(ncols=3, nrows=nsubj, figsize=(8, 2*nsubj))
    
    width = 2000
    
    
    meanlist = np.zeros((len(group1), len(excit_class[:-1])))
    maxlist = np.zeros((len(group1), len(excit_class[:-1])))
    detlist = np.zeros((len(group1), len(excit_class[:-1])))
    
    group1_area_ratio = []
    group1_area = []
    
    for (i, ID) in enumerate(group1):
        
        gmax = []
        gmin = []
        trend_density = []
        
        for (j, layer) in enumerate(excit_class[:-1]):
            
            subj_df = merge_df.loc[ID]
            xyz = subj_df.loc[subj_df['marker'] == layer][['ppx','ppy']].values
            
            if xyz.shape[0] > 5:
                kde = stats.gaussian_kde(xyz.T)
                mean = np.mean(xyz, axis=0)
                eigs, eigvs = np.linalg.eig(kde.covariance / (kde.factor**2))
                
                xx, yy = np.mgrid[-width/2:width/2:50, -width/2:width/2:50]
                density = kde(np.c_[xx.flat, yy.flat].T).reshape(xx.shape)
                gmax.append(np.max(density))
                gmin.append(np.min(density))
                trend_density.append([i,layer,xx,yy, density,xyz, eigs, eigvs, mean, sklearn.metrics.pairwise_distances(xyz)])
                
                # print(kde.factor)
            else:
                trend_density.append([i,layer,np.nan,np.nan,np.nan,xyz, sklearn.metrics.pairwise_distances(xyz)])
            
            pdist = trend_density[j][9]
            pdist = pdist[np.triu_indices(pdist.shape[0],k=1)]
            meanlist[i,j] = np.mean(pdist)
            maxlist[i,j] = np.max(pdist)
            
        area_ratio = []
        raw_area = []
        for (j, layer) in enumerate(excit_class[:-1]):
        
            eigs = trend_density[j][6]
            area = eigs[0]*eigs[1]
            area_ratio.append(area)
            raw_area.append(area)
        area_ratio = [a/area_ratio[1] for a in area_ratio]
        
        group1_area_ratio.append(area_ratio)
        group1_area.append(raw_area)
        
        for (j, layer) in enumerate(excit_class[:-1]):
            detlist[i,j] = area_ratio[j]
            
        for (j, layer) in enumerate(excit_class[:-1]):
            
            gmax = np.max(gmax)
            gmin = np.min(gmin)
            axs[i,j].contour(trend_density[j][2],
                             trend_density[j][3],
                             trend_density[j][4],
                             levels=np.linspace(np.min(gmin), np.max(gmax),20),
                             colors='black', alpha=0.5, linewidths=0.5)
            
            
            axs[i,j].scatter(trend_density[j][5][:,0],
                             trend_density[j][5][:,1],
                             c='r', alpha=0.5, s=0.5)
            
            axs[i,j].quiver(trend_density[j][8][0],trend_density[j][8][1],
                            2*np.sqrt(trend_density[j][6][0])*trend_density[j][7][0,0],
                            2*np.sqrt(trend_density[j][6][0])*trend_density[j][7][1,0],
                            angles='xy', scale_units='xy', scale=1)
            
            axs[i,j].quiver(trend_density[j][8][0],trend_density[j][8][1],
                            2*np.sqrt(trend_density[j][6][1])*trend_density[j][7][0,1],
                            2*np.sqrt(trend_density[j][6][1])*trend_density[j][7][1,1],
                            angles='xy', scale_units='xy', scale=1)
            
            axs[i,j].set_xlim([-1000, 1000])
            axs[i,j].set_ylim([-1000, 1000])
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            if j == 0:
                axs[i,j].text(-950, 800, ID, fontsize=10)
            axs[i,j].text(400, -900, layer, fontsize=8)
            
            pdist = trend_density[j][9]
            pdist = pdist[np.triu_indices(pdist.shape[0],k=1)]
            axs[i,j].text(-950, -910, 'mean = {:.2f}({:.2f})'.format(np.mean(pdist), np.std(pdist)), fontsize=8)
            axs[i,j].text(-950, -790, 'max = {:.2f}'.format(np.max(pdist)), fontsize=8)
            # axs[i,j].text(300, -780, 'det = {:.2f}'.format(area_ratio[j]), fontsize=8)
            
            if i == 0:
                if j == len(excit_class[:-1])-1:
                    ob = AnchoredHScaleBar(size=500, label="500 $\mu m$", loc=1, frameon=False,
                           pad=0.6,sep=4, linekw=dict(color="crimson"),) 
                    axs[i,j].add_artist(ob)
    
    pyplot.subplots_adjust(wspace=0, hspace=0)
    pyplot.savefig(os.path.join(savepath_plot, 'group1_spread.png'))
    pyplot.savefig(os.path.join(savepath_plot, 'group1_spread.eps'), format='eps')
    pyplot.savefig(os.path.join(savepath_plot, 'group1_spread.svg'), format='svg')
    
    nsubj = len(group2)
    fig, axs = pyplot.subplots(ncols=3, nrows=nsubj, figsize=(8, 2*nsubj))
    
    width = 2000
    
    meanlist = np.zeros((len(group1), len(excit_class[:-1])))
    maxlist = np.zeros((len(group1), len(excit_class[:-1])))
    detlist = np.zeros((len(group1), len(excit_class[:-1])))
    
    group2_area_ratio = []
    group2_area = []
    
    for (i, ID) in enumerate(group2):
        
        gmax = []
        gmin = []
        trend_density = []
        
        for (j, layer) in enumerate(excit_class[:-1]):
            
            subj_df = merge_df.loc[ID]
            xyz = subj_df.loc[subj_df['marker'] == layer][['ppx','ppy']].values
            
            if xyz.shape[0] > 5:
                kde = stats.gaussian_kde(xyz.T)
                mean = np.mean(xyz, axis=0)
                eigs, eigvs = np.linalg.eig(kde.covariance / (kde.factor**2))
                
                xx, yy = np.mgrid[-width/2:width/2:50, -width/2:width/2:50]
                density = kde(np.c_[xx.flat, yy.flat].T).reshape(xx.shape)
                gmax.append(np.max(density))
                gmin.append(np.min(density))
                trend_density.append([i,layer,xx,yy, density,xyz, eigs, eigvs, mean, sklearn.metrics.pairwise_distances(xyz)])
                
                # print(kde.factor)
            else:
                trend_density.append([i,layer,np.nan,np.nan,np.nan,xyz, sklearn.metrics.pairwise_distances(xyz)])
            
            pdist = trend_density[j][9]
            pdist = pdist[np.triu_indices(pdist.shape[0],k=1)]
            meanlist[i,j] = np.mean(pdist)
            maxlist[i,j] = np.max(pdist)
            
        area_ratio = []
        raw_area = []
        for (j, layer) in enumerate(excit_class[:-1]):
        
            eigs = trend_density[j][6]
            area = eigs[0]*eigs[1]
            area_ratio.append(area)
            raw_area.append(area)
        area_ratio = [a/area_ratio[1] for a in area_ratio]
        
        group2_area_ratio.append(area_ratio)
        group2_area.append(raw_area)
        
        for (j, layer) in enumerate(excit_class[:-1]):
            detlist[i,j] = area_ratio[j]
            
        
        for (j, layer) in enumerate(excit_class[:-1]):
            
            gmax = np.max(gmax)
            gmin = np.min(gmin)
            axs[i,j].contour(trend_density[j][2],
                             trend_density[j][3],
                             trend_density[j][4],
                             levels=np.linspace(np.min(gmin), np.max(gmax),20),
                             colors='black', alpha=0.5, linewidths=0.5)
            
            
            axs[i,j].scatter(trend_density[j][5][:,0],
                             trend_density[j][5][:,1],
                             c='r', alpha=0.5, s=0.5)
            
            axs[i,j].quiver(trend_density[j][8][0],trend_density[j][8][1],
                            2*np.sqrt(trend_density[j][6][0])*trend_density[j][7][0,0],
                            2*np.sqrt(trend_density[j][6][0])*trend_density[j][7][1,0],
                            angles='xy', scale_units='xy', scale=1)
            
            axs[i,j].quiver(trend_density[j][8][0],trend_density[j][8][1],
                            2*np.sqrt(trend_density[j][6][1])*trend_density[j][7][0,1],
                            2*np.sqrt(trend_density[j][6][1])*trend_density[j][7][1,1],
                            angles='xy', scale_units='xy', scale=1)
            
            axs[i,j].set_xlim([-1000, 1000])
            axs[i,j].set_ylim([-1000, 1000])
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            if j == 0:
                axs[i,j].text(-950, 800, ID, fontsize=10)
            axs[i,j].text(400, -900, layer, fontsize=8)
            
            pdist = trend_density[j][9]
            pdist = pdist[np.triu_indices(pdist.shape[0],k=1)]
            axs[i,j].text(-950, -910, 'mean = {:.2f}'.format(np.mean(pdist)), fontsize=8)
            axs[i,j].text(-950, -790, 'max = {:.2f}'.format(np.max(pdist)), fontsize=8)
            
            if i == 0:
                if j == len(excit_class[:-1])-1:
                    ob = AnchoredHScaleBar(size=500, label="500 $\mu m$", loc=1, frameon=False,
                           pad=0.6,sep=4, linekw=dict(color="crimson"),) 
                    axs[i,j].add_artist(ob)
                    
            

    pyplot.subplots_adjust(wspace=0, hspace=0)
    pyplot.savefig(os.path.join(savepath_plot, 'group2_spread.png'))
    pyplot.savefig(os.path.join(savepath_plot, 'group2_spread.eps'), format='eps')
    pyplot.savefig(os.path.join(savepath_plot, 'group2_spread.svg'), format='svg')
    
    full_area_ratio = group1_area_ratio + group2_area_ratio
    full_area_ratio = np.vstack(full_area_ratio)

    t_statistic, p_value = stats.ttest_1samp(full_area_ratio[:,0], 1)
    p_value_one_tailed = p_value / 2
    
    # Output results
    print(f"T-statistic: {t_statistic}")
    print(f"P-value (one-tailed): {p_value_one_tailed}")
    
    t_statistic, p_value = stats.ttest_1samp(full_area_ratio[:,2], 1)
    p_value_one_tailed = p_value / 2
    
    # Output results
    print(f"T-statistic: {t_statistic}")
    print(f"P-value (one-tailed): {p_value_one_tailed}")
    