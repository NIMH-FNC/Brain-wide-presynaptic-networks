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
import open3d as o3d

import itertools
from itertools import permutations, combinations

from scipy import stats
from scipy.spatial import distance
from numpy.linalg import eig, inv

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


# Compute normal vector given the point cloud of surface
def compute_normalvec(
    contour_df, 
    contour_name,
    *,
    radius_ratio=5,
    nb_normal=50,
    nface=10000,
    mesh_colorscale='Blues',
    meshpt_color='blue'
    
):

    # estimate normal vectors given the point clounds of boundary vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.abs(contour_df[['x','y','z']].values))
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(nb_normal)
    
    # construct triangular meshes using the boundary vertices
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = radius_ratio * avg_dist
    # using ball pivoting algorithm to estimate the mesh
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
    
    # alternatively, we can also use the poisson algorithm to estimate the mesh
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)
    
    # mesh clearning
    dec_mesh = mesh.simplify_quadric_decimation(nface)
    
    # generate go.Mesh3d object for visualization
    face = np.asarray(dec_mesh.triangles)
    vertex = np.asarray(dec_mesh.vertices)
    go_mesh = go.Mesh3d(
        x=np.abs(vertex[:,0]),
        y=np.abs(vertex[:,1]),
        z=np.abs(vertex[:,2]),
        colorbar_title=contour_name,
        # i, j and k give the vertices of triangles
        i = face[:,0],
        j = face[:,1],
        k = face[:,2],
        colorscale = mesh_colorscale,
        opacity=0.15,
        showscale=False,
        hoverinfo='skip',
        name='{} surface'.format(contour_name)
        )
    
    go_meshpt = go.Scatter3d(
        x=np.abs(contour_df['x'].values),
        y=np.abs(contour_df['y'].values),
        z=np.abs(contour_df['z'].values),
        mode='markers', marker=dict(size=1, opacity=0.1),
        name='{} surface vertices'.format(contour_name),
        marker_color=meshpt_color
        )
    
    # normal vector estimate for every manually marked vertex in the cortical surface
    normalvec = np.asarray(pcd.normals)
    print('Number of normal vectors:', normalvec.shape)

    return vertex, normalvec, go_meshpt, dec_mesh


def estimate_plane(coord):
    
    xmin = np.min(coord[:,0])
    xmax = np.max(coord[:,0])
    ymin = np.min(coord[:,1])
    ymax = np.max(coord[:,1])
    zmin = np.min(coord[:,2])
    zmax = np.max(coord[:,2])
    
    xgrid = np.linspace(xmin, xmax, 20)
    ygrid = np.linspace(ymin, ymax, 20)

    X,Y = np.meshgrid(xgrid, ygrid)
        
    XX = X.flatten()
    YY = Y.flatten()
    
    A = np.c_[coord[:,0], coord[:,1], np.ones(coord.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, coord[:,2])
    Z = C[0]*X + C[1]*Y + C[2]
    
    print('C = ', C)

    p = np.asarray([np.mean(X), np.mean(Y), np.mean(Z)])
    planecenter = p
    
    print('planecenter = ', planecenter)

    A = np.array([X[0,0], Y[0,0], Z[0,0]])
    B = np.array([X[1,1], Y[1,1], Z[1,1]])
    C = np.array([X[5,2], Y[5,2], Z[5,2]])
    AB = (B-A)/np.linalg.norm(B-A)
    AC = (C-A)/np.linalg.norm(C-A)
    
    print('AB, AC = ', AB, AC)

    nn = np.cross(AB, AC)
    
    print('nn = ', nn)
    
    nn = nn/np.linalg.norm(nn)        
    normal = nn
    
    print('normal = ', normal)

    proj_coord = np.zeros_like(coord)
    proj_length = np.zeros(coord.shape[0])
    for k in range(coord.shape[0]):
        q = coord[k,:]
        proj_coord[k,:] = q - np.dot(q - p, nn) * nn
        proj_length[k] = np.sign(np.dot(q - p, nn))*np.linalg.norm(np.dot(q - p, nn) * nn)

    
    return xgrid, ygrid, Z, planecenter, normal, proj_coord


def tilting(coord, center, nn, ref_direction):
    
    coord -= np.tile(center, (coord.shape[0],1))
    
    nn = nn/np.linalg.norm(nn)  
    n0 = ref_direction
    theta = np.arccos(np.dot(nn, n0))
    sgn = np.sign(np.cross(nn, n0))
    theta = np.mod(theta * (-1)**(sgn[2] < 0), 2*np.pi);
    ct = np.cos(theta)
    st = np.sin(theta)

    u = np.cross(nn, n0)
    u = u/np.linalg.norm(u)
    R = np.array([[ct + u[0]**2*(1-ct), u[0]*u[1]*(1-ct), u[1]*st],
                  [u[0]*u[1]*(1-ct), ct + u[1]**2*(1-ct), -u[0]*st],
                  [-u[1]*st, u[0]*st, ct]])
    
#     assert np.abs(np.dot(np.dot(R, nn), n0)-1) < 1e-8

    print(np.abs(np.dot(np.dot(R, nn), n0)-1) )

    coord = (R@(coord.T)).T
    
    coord += np.tile(center, (coord.shape[0],1))
    
    return coord


def generate_resample(IDlist, group1, group2, nperm=1000, show_baseline=False):

    nsubj = len(IDlist)
    g1 = [ID for ID in IDlist if ID in group1]
    g2 = [ID for ID in IDlist if ID in group2]
    full_list = g1 + g2
    
    sample_list = [random.sample( full_list, len(g1) ) for r in range(nperm)]
    resample_list = [[samples, list(np.setdiff1d(full_list, samples))] for samples in sample_list]
    resample_list.insert(0, [g1, g2])
    rlen = len(resample_list)

    if show_baseline:
        print('First (Observation) combination: \n'
              + str(resample_list[0][0]) + ' \n'
              + str(resample_list[0][1]) )

    return resample_list, rlen

def compute_metric(
    merge_df,
    rgroup1, rgroup2,
    layerlist,
    coordinate='planeprojection',
    metric='wass',
    epsilon=0.1,
    center_shift=False
):
    # epsilon only useful for sinkhorn type metrics

    if coordinate == 'atlas':
        dimension = 3
        coordinate_column = ['atlas_x', 'atlas_y', 'atlas_z']
    elif coordinate == 'planeprojection':
        dimension = 2
        coordinate_column = ['ppx', 'ppy']
    else:
        raise ValueError('Unnknown coordinates')
    
    # compute the relative weight of each subject in a group
    weight_group1 = []
    weight_group2 = []
    
    group1_pcoord = np.empty((0, dimension), float)
    group2_pcoord = np.empty((0, dimension), float)
    
    group1_sampleweight = np.array([])
    group2_sampleweight = np.array([])
    
    for (j, ID) in enumerate(rgroup1):
        subj_df = merge_df.loc[ID]
        layer_coord = subj_df.loc[subj_df['marker'].isin(layerlist)][coordinate_column].values
        if layer_coord.shape[0] > 0:
            if center_shift:
                layer_coord = layer_coord - np.mean(layer_coord, axis=0)
            group1_pcoord = np.append(group1_pcoord, layer_coord, axis=0)
            weight = 1/(layer_coord.shape[0])
            group1_sampleweight = np.append(group1_sampleweight, np.ones(layer_coord.shape[0])*weight)
    for (j, ID) in enumerate(rgroup2):
        subj_df = merge_df.loc[ID]
        layer_coord = subj_df.loc[subj_df['marker'].isin(layerlist)][coordinate_column].values
        if layer_coord.shape[0] > 0:
            if center_shift:
                layer_coord = layer_coord - np.mean(layer_coord, axis=0)
            group2_pcoord = np.append(group2_pcoord, layer_coord, axis=0)
            weight = 1/(layer_coord.shape[0])
            group2_sampleweight = np.append(group2_sampleweight, np.ones(layer_coord.shape[0])*weight)
            
    assert group1_pcoord.shape[0] > 0
    assert group2_pcoord.shape[0] > 0
    assert np.sum(group1_sampleweight <= 0) == 0
    assert np.sum(group2_sampleweight <= 0) == 0
    
    n1 = group1_pcoord.shape[0]
    n2 = group2_pcoord.shape[0]
        
    # transform the distribution into a probabilistic distribution
    xw = group1_sampleweight / np.sum(group1_sampleweight)
    yw = group2_sampleweight / np.sum(group2_sampleweight)
    
    if metric == 'wass':
        M = ot.dist(group1_pcoord, group2_pcoord)
        M /= M.max()
        Gs = ot.emd2(xw, yw, M)
    elif metric == 'sinkhorn':
        M = ot.dist(group1_pcoord, group2_pcoord)
        M /= M.max()
        Gs = ot.sinkhorn2(xw, yw, M, epsilon)
    elif metric == 'empirical_sinkhorn':
        Gs = ot.bregman.empirical_sinkhorn2(group1_pcoord, group2_pcoord, a=xw, b=yw, reg=epsilon)
    elif metric == 'JS':
        
        gmm_p = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, dimension, group1_pcoord, xw)
        gmm_q = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, dimension, group2_pcoord, yw )
        
        x = gmm_p.sample(10**4)
        y = gmm_q.sample(10**4)
        log_p_x = gmm_p.log_probability(x)
        log_p_y = gmm_p.log_probability(y)
        log_q_x = gmm_q.log_probability(x)
        log_q_y = gmm_q.log_probability(y)
        log_mix_x = np.logaddexp(log_p_x, log_q_x)
        log_mix_y = np.logaddexp(log_p_y, log_q_y)
        
        Gs = (log_p_x.mean()-(log_mix_x.mean()-np.log(2)) + log_q_y.mean()-(log_mix_y.mean()-np.log(2))) / 2
        
    return Gs

def define_class():

    excit_class = ['Layer II/III', 'Layer IV', 'Layer V', 'Layer VI']
    inhib_class = ['Layer I', 'GABA-II/III', 'GABA-IV', 'GABA-V', 'GABA-VI']
    longr_class = ['M1/M2',
                   'S2',
                   'LR_Sen',
                   'Thal_VPM/VPL', 'Thal_PO',
                   'Thal_VA/VL', 'Thal_VM', 'Thal_AM',
                   'Fissure', 'cFissure',
                   'cS1',
                   'BF', 'Striatum',
                   'Insular',
                   'Retrosplenial',
                   'Orbital','cOrbital',
                   'Thal_Other', 'Thal_LD/LP',
                   'LR_Others',
                   'Amygdala',
                   'DorsalRaphe']
    
    other = ['Insular', 'Retrosplenial',
             'Orbital', 'cOrbital',
             'Thal_Other', 'LR_Others',
             'Amygdala', 'DorsalRaphe']
    
    thal_dict = {
        'sensory':['Thal_VPM/VPL', 'Thal_PO'],
        'anterior':['Thal_VA/VL', 'Thal_VM', 'THAL_AM'],
        'PO':['Thal_PO'],
        'VPM':['Thal_VPM/VPL'],
        'main':['Thal_VPM/VPL', 'Thal_PO', 'Thal_VA/VL', 'Thal_VM', 'THAL_AM', 'Thal_LD/LP'],
        'all':['Thal_VPM/VPL', 'Thal_PO', 'Thal_VA/VL', 'Thal_VM', 'THAL_AM', 'Thal_LD/LP', 'Thal_Other']
        }

    return excit_class, inhib_class, longr_class, other, thal_dict


def cell_count(IDlist, merge_df, base='all'):

    excit_class, inhib_class, longr_class, other, thal_dict = define_class()
    
    base_ncell = []
    for (i, ID) in enumerate(IDlist):
        subjcell = merge_df.loc[ID]
        if base == 'all':
            cell_number = subjcell.loc[ subjcell['marker'].isin(excit_class+inhib_class+longr_class) ].shape[0]
        elif base == 'local':
            cell_number = subjcell.loc[ subjcell['marker'].isin(excit_class+inhib_class) ].shape[0]
        elif base == 'longr':
            cell_number = subjcell.loc[ subjcell['marker'].isin(longr_class) ].shape[0]
        elif base == 'other':
            cell_number = subjcell.loc[ subjcell['marker'].isin(other) ].shape[0]
        elif base == 'thal':
            cell_number = subjcell.loc[ subjcell['marker'].isin(thal_dict['all']) ].shape[0]
        elif base == 'thal_main':
            cell_number = subjcell.loc[ subjcell['marker'].isin(thal_dict['main']) ].shape[0]
        elif base == 'thal_sensory':
            cell_number = subjcell.loc[ subjcell['marker'].isin(thal_dict['sensory']) ].shape[0]
        elif base == 'thal_anterior':
            cell_number = subjcell.loc[ subjcell['marker'].isin(thal_dict['anterior']) ].shape[0]
        elif base == 'thal_PO':
            cell_number = subjcell.loc[ subjcell['marker'].isin(thal_dict['PO']) ].shape[0]
        elif base == 'thal_VPM':
            cell_number = subjcell.loc[ subjcell['marker'].isin(thal_dict['VPM']) ].shape[0]
        else:
            raise ValueError('unknown base')
        base_ncell.append(cell_number)
        
    return base_ncell
    