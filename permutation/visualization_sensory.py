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

# reproducibility
random_state = 3121213
np.random.seed(random_state)
random.seed(random_state)

# visualization of Thal sensory neurons


if __name__ == '__main__':

    # load neurons spatial location data
    dir_data = "./data/processed/combined/*.csv"
    group1, group2, merge_df, IDlist = load_data(dir_data)
    # load neuron class definition
    excit_class, inhib_class, longr_class, other, thal_dict = define_class()

    
    POM_contour = pd.read_csv('./data/contour/POM_contour.csv')
    
    # estimate POM surface
    radius_ratio = 5
    nb_normal = 20
    nface = 5000
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.abs(POM_contour[['x','y','z']].values))
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(nb_normal)
    
    POM_normal = np.asarray(pcd.normals)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = radius_ratio * avg_dist
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    dec_mesh = mesh.simplify_quadric_decimation(nface)

    face = np.asarray(dec_mesh.triangles)
    vertex = np.abs(np.asarray(dec_mesh.vertices))
    
    POM_vertex = np.abs(np.asarray(dec_mesh.vertices))
    POM_mesh_transparent = go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:,0],
            y=vertex[:,1],
            z=vertex[:,2],
            colorbar_title='POM_mesh',
    
            # i, j and k give the vertices of triangles
            i = face[:,0],
            j = face[:,1],
            k = face[:,2],
            colorscale ='Greys', color='Grey', opacity=0.3, showscale=False, hoverinfo='skip',
            name='POM_mesh', showlegend=True
        )


    VPM_contour = pd.read_csv('./data/contour/VPM_contour.csv')
    
    # estimate VPM surface
    radius_ratio = 25
    nb_normal = 20
    nface = 1000
    
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.abs(VPM_contour[['x','y','z']].values))
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(nb_normal)
    
    VPM_normal = np.asarray(pcd.normals)
    
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = radius_ratio * avg_dist
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    dec_mesh = mesh.simplify_quadric_decimation(nface)

    face = np.asarray(dec_mesh.triangles)
    vertex = np.abs(np.asarray(dec_mesh.vertices))
    VPM_vertex = np.abs(np.asarray(dec_mesh.vertices))
    
    VPM_mesh_transparent = go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:,0],
            y=vertex[:,1],
            z=vertex[:,2],
            colorbar_title='POM_mesh',
    
            # i, j and k give the vertices of triangles
            i = face[:,0],
            j = face[:,1],
            k = face[:,2],
            colorscale ='Greys', color='Grey', opacity=0.1, showscale=False, hoverinfo='skip',
            name='VPM_mesh', showlegend=True
        )

    ROIlist = 'sensory'

    filepath = os.path.join('./output/visualization', '{}-group.html'.format(ROIlist))
    available_ID, coordinate, ncell, groupweight = load_groupROIlist_coord(IDlist, merge_df, ROIlist=thal_dict[ROIlist])
    
    group1_cell = np.empty((0,3))
    group2_cell = np.empty((0,3))
    for ID in available_ID:
        celldf = merge_df.loc[ID]
        atlas_xyz = celldf.loc[celldf['marker'].isin(thal_dict[ROIlist])][['atlas_x', 'atlas_y', 'atlas_z']].values
        if ID in group1:
            label = 'group 1'
            group1_cell = np.append(group1_cell, atlas_xyz, axis=0)
        else:
            label = 'group 2'
            group2_cell = np.append(group2_cell, atlas_xyz, axis=0)
            
    group1_cell = go.Scatter3d(x=group1_cell[:,0], y=group1_cell[:,1], z=group1_cell[:,2],
                               mode='markers', marker=dict(size=5, opacity=1), marker_color='black', name='Group 1')
    group2_cell = go.Scatter3d(x=group2_cell[:,0], y=group2_cell[:,1], z=group2_cell[:,2],
                               mode='markers', marker=dict(size=5, opacity=1), marker_color='magenta', name='Group 2')
    
    group_cell_list = [group1_cell, group2_cell]
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    
    ax_style = dict(showbackground = False,
                    backgroundcolor="rgb(240, 240, 240)",
                    showgrid=True,
                    zeroline=False,
                    showticklabels=False)
    
    MESH_list = [POM_mesh_transparent, VPM_mesh_transparent]
    fig = go.Figure(data= group_cell_list + MESH_list , layout=layout)
    
    fig.update_layout(scene=dict(xaxis=ax_style, 
                                 yaxis=ax_style, 
                                 zaxis=ax_style,
                                 # camera_eye=dict(x=1.85, y=1.85, z=1)
                                ),
                     legend={'itemsizing': 'constant', 'x':0.8, 'y':0.8})
    
    fig.write_html(filepath)
    