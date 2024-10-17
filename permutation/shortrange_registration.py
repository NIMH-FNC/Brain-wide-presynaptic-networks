# system & data loading
import os
import pandas as pd
import glob
import pickle

# computational packages
import numpy as np
import scipy
import open3d as o3d

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
import copy

# custom functions
from utils import compute_normalvec, estimate_plane, tilting
from utils_load import load_data

seed = 31213
np.random.seed(seed)
random.seed(seed)

# perform fine scale rigid registration

if __name__ == '__main__':

    # define excitatory and inhibitory class
    excit_class = ['Layer II/III', 'Layer IV', 'Layer V', 'Layer VI']
    inhib_class = ['Layer I', 'GABA-II/III', 'GABA-IV', 'GABA-V', 'GABA-VI']
    
    # create directory to store registration output
    savepath_registration = './output/shortrange_registration'
    os.makedirs(savepath_registration, exist_ok=True)
    savepath_planeprojection = './output/shortrange_layerprojection'
    os.makedirs(savepath_planeprojection, exist_ok=True)
    

    contour_name = 'Cortex'
    # load spatial coordinates of manually marked boundary vertices
    _path_to_data_contour = './data/contour/{}_contour.csv'.format(contour_name)
    contour_df = pd.read_csv(_path_to_data_contour, index_col=0)
    
    contour_name = 'Cortex'
    radius_ratio = 5
    nb_normal = 50
    nface = 10000
    
    # load spatial coordinates of manually marked boundary vertices
    _path_to_data_contour = './data/contour/{}_contour.csv'.format(contour_name)
    contour_df = pd.read_csv(_path_to_data_contour, index_col=0)
    
    # compute normal vector given the point cloud of cortical surface
    # (used for visualization)
    vertex, cortex_normal, cortex_mesh, cortex_meshpt = compute_normalvec(
        contour_df,
        contour_name,
        radius_ratio=radius_ratio,
        nb_normal=nb_normal,
        nface=nface
    )

    # load neurons spatial location data
    pathlist_data = "./data/processed/combined/*.csv"
    group1, group2, merge_df, IDlist = load_data(pathlist_data)


    # estimate the reference injection site and the corresponding normal vector
    injection_estimate = []
    for ID in group1 + group2:
        subj_df = merge_df.loc[ID]
        
        center_mass = subj_df.loc[subj_df['marker']=='Layer II/III'][['atlas_x', 'atlas_y', 'atlas_z']].values
        center_mass = np.mean(center_mass, axis=0)[None,:]
        injection_estimate.append(center_mass)
        
    injection_estimate = np.concatenate(injection_estimate)
    ref_injection = np.mean(injection_estimate, axis=0)[None,:]
    
    ref_nearest = np.argmin( np.sum((np.abs(vertex) - ref_injection)**2, axis=1) )
    ref_neighbor = np.argsort( np.sum((np.abs(vertex) - np.abs(vertex[ref_nearest,:]) )**2, axis=1) )[:20]
    
    _, _, _, _, ref_normal, _ = estimate_plane(np.abs(vertex[ref_neighbor,:]))



    # Given the reference injection site and the corresponding normal vector,
    # we perform rigid alignment for individual subject.
    for ID in group1+group2:
        print(ID)
        subj_df = merge_df.loc[ID] 
        
        for layer in excit_class+inhib_class:
            layer_df = subj_df.loc[subj_df['marker']==layer]
            atlas_xyz = layer_df[['atlas_x', 'atlas_y', 'atlas_z']].values
            
        center_mass = subj_df.loc[subj_df['marker']=='Layer II/III'][['atlas_x', 'atlas_y', 'atlas_z']].values
        center_mass = np.mean(center_mass, axis=0)[None,:]
    
        surf_nearest_vertex = np.argmin( np.sum((np.abs(vertex) - center_mass)**2, axis=1) )
        surf_nearest = np.abs(vertex[surf_nearest_vertex,:][None,:])
        surf_neighbor = np.argsort( np.sum((np.abs(vertex) - np.abs(vertex[surf_nearest_vertex,:]) )**2, axis=1) )[:20]
        
        _, _, _, _, mean_surf_normal, _ = estimate_plane(np.abs(vertex[surf_neighbor,:]))
        
        coord = subj_df.loc[subj_df['marker'].isin(excit_class+inhib_class)][['atlas_x', 'atlas_y', 'atlas_z']].values
        
        ## shift to reference point
        coord += np.tile(ref_injection-center_mass, (coord.shape[0],1))
        assert np.sum(np.isnan(coord)) == 0
        
        if np.linalg.norm(mean_surf_normal - ref_normal) < 1e-8:
            tilt_coord = coord
        else:
            tilt_coord = tilting(coord, ref_injection, mean_surf_normal, ref_normal)
        assert np.sum(np.isnan(tilt_coord)) == 0
        
    
        registered_df = subj_df.loc[subj_df['marker'].isin(excit_class+inhib_class)].copy().reset_index()
        registered_df['rx'] = tilt_coord[:,0]
        registered_df['ry'] = tilt_coord[:,1]
        registered_df['rz'] = tilt_coord[:,2]
        
        registered_df.to_csv(os.path.join(savepath_registration, ID+'-registered.csv'))

    pathlist_data = os.path.join(savepath_registration, '*-registered.csv')
    # reload neurons spatial location data for registration results
    group1, group2, merge_df = load_data(pathlist_data)

    # compute layer center by averaging
    layer_center = []

    for layer in excit_class+inhib_class:
        
        layer_xyz = []
        for ID in group1+group2:
            subj_df = merge_df.loc[ID]
            layer_df = subj_df.loc[subj_df['marker']==layer]
            reg_xyz = layer_df[['rx', 'ry', 'rz']].values
            layer_xyz.append(reg_xyz)
        layer_xyz = np.concatenate(layer_xyz)
        layer_center.append(np.mean(layer_xyz, axis=0))


    # compute layer projection and plane projection for  individual subject
    for ID in group1+group2:
        subj_df = merge_df.loc[ID] 
        layerprojected_df = []
        
        for (i,layer) in enumerate(excit_class+inhib_class):
            layer_df = copy.deepcopy(subj_df.loc[subj_df['marker']==layer])
            rxyz = layer_df[['rx', 'ry', 'rz']].values
            
            proj_coord = np.zeros_like(rxyz)
            proj_length = np.zeros(rxyz.shape[0])
            nn = ref_normal
            p = layer_center[i]
            for k in range(rxyz.shape[0]):
                q = rxyz[k,:]
                proj_coord[k,:] = q - np.dot(q - p, nn) * nn
                proj_length[k] = np.sign(np.dot(q - p, nn))*np.linalg.norm(np.dot(q - p, nn) * nn)
                
                
            n0 = np.array([0,0,1])
            ct = np.dot(nn, n0)/np.linalg.norm(nn)
            st = np.sqrt(1 - ct**2)
            u = np.cross(nn, n0)/np.linalg.norm(nn)
            R = np.array([[ct + u[0]**2*(1-ct), u[0]*u[1]*(1-ct), u[1]*st],
                          [u[0]*u[1]*(1-ct), ct + u[1]**2*(1-ct), -u[0]*st],
                          [-u[1]*st, u[0]*st, ct]])
    
        
            layer_df['px'] = proj_coord[:,0]
            layer_df['py'] = proj_coord[:,1]
            layer_df['pz'] = proj_coord[:,2]
    
            Mproj_coord = np.zeros_like(proj_coord)
            for k in range(proj_coord.shape[0]):
                Mproj_coord[k,:] = R@(proj_coord[k,:] - p)
            
            layer_df['ppx'] = Mproj_coord[:,0]
            layer_df['ppy'] = Mproj_coord[:,1]
            layer_df['ppz'] = Mproj_coord[:,2]
            
            layerprojected_df.append(layer_df)
    
        layerprojected_df = pd.concat(layerprojected_df)
        layerprojected_df.to_csv(os.path.join(savepath_planeprojection, ID+'-layerprojection.csv'))