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
from scipy import stats
from scipy.spatial import distance
import sklearn
from sklearn import mixture

import random
import itertools
from itertools import permutations, combinations

import open3d as o3d

# visualization
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import plotly
import plotly.io as pio
import plotly.graph_objects as go
pio.orca.config.use_xvfb = True
import IPython
from IPython.display import HTML
from IPython.display import IFrame

import seaborn as sns
from textwrap import wrap

# import ot
# import ot.plot

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

def load_coord(root, name='Secondary motor area-L'):
    coord = []
    for child in root:
    # print(child.tag, child.attrib)
        if 'contour' in child.tag:
            if child.attrib['name'] == name:
                # print(child.tag, child.attrib['name'])
                for element in child:
                    if 'point' in element.tag:
                        # print(element.tag)
                        # print(element.attrib['x'], element.attrib['y'], element.attrib['z'])
                        if 'L' in name:
                            coord.append(np.array([float(element.attrib['x']),
                                                   float(element.attrib['y']),
                                                   float(element.attrib['z'])]
                                                 )
                                        )
                        else:
                            coord.append(np.array([float(element.attrib['x']),
                                                   float(element.attrib['y']),
                                                   float(element.attrib['z'])]
                                             )
                                    )
    return coord


def compute_signed_distance(scene_SM_L, scene_wall_L, scene_SM_R, scene_wall_R, points, eps=-40):
    
    interior_label = []
    position_label = []

    distance_df = pd.DataFrame( columns=['distance', 'region'] )
    
    for i in range(points.shape[0]):
        query_point = o3d.core.Tensor([points[i,:]], dtype=o3d.core.Dtype.Float32)

        # if distance < 0, it means the query_point is interior of the watertight surface, positive otherwise
        distance_SM_R = scene_SM_R.compute_signed_distance(query_point)
        distance_SM_R = distance_SM_R.numpy()
        # print(i, 'SM_R interior test', distance_SM_R)
        
        # if it is inside SM_R
        if distance_SM_R < eps:
            interior_label.append(True)
            position_label.append('SM_R')
            distance_df = pd.concat([ distance_df, pd.DataFrame({'distance':distance_SM_R, 'region':'SM_R'})], ignore_index=True)
        # if it is outside SM_R and if it is inside SM_L
        else:
            distance_SM_L = scene_SM_L.compute_signed_distance(query_point)
            distance_SM_L = distance_SM_L.numpy()
            # print('SM_L interior test', distance_SM_L)
            if distance_SM_L < eps:
                interior_label.append(True)
                position_label.append('SM_L')
                distance_df = pd.concat([ distance_df, pd.DataFrame({'distance':distance_SM_L, 'region':'SM_L'})], ignore_index=True)

            # if it is also outside SM_L
            else:

                # if it is > 5700, it is on the left hemisphere
                if points[i,0] > 5700:

                    # print('Its on left hemisphere')
                    PM_L_dist = np.min(np.sqrt(np.sum( (PM_L - np.tile(points[i,:], (len(PM_L),1)) )**2, axis=1) ))
                    # find the shortest distance between contour points in PM_L
                    # print('PM_L_dist', PM_L_dist)
                    
                    interior_label.append(False)
                    position_label.append('PM_L')
                    distance_df = pd.concat([ distance_df, pd.DataFrame({'distance':[PM_L_dist], 'region':'PM_L'})], ignore_index=True)
               
                else:
                    PM_R_dist = np.min(np.sqrt(np.sum((PM_R - np.tile(points[i,:], (len(PM_R),1)))**2, axis=1)))
                    # print('PM_R_dist', PM_R_dist)
                    
                    interior_label.append(False)
                    position_label.append('PM_R')
                    distance_df = pd.concat([ distance_df, pd.DataFrame({'distance':[PM_R_dist], 'region':'PM_R'})], ignore_index=True)
                    
    points_trace = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2],
                                mode='markers', marker=dict(size=3, 
                                                            opacity=1, 
                                                            color=np.array(interior_label)*1.0),
                                name='Interior/exterior'
                               )

    signed_distance = []
    for i in range(len(position_label)):
        query_point = o3d.core.Tensor([points[i,:]], dtype=o3d.core.Dtype.Float32)
        if points[i,0] < 5700:
            distance = scene_wall_R.compute_signed_distance(query_point)
        else:
            distance = scene_wall_L.compute_signed_distance(query_point)
        if interior_label[i]:
            signed_distance.append(np.abs(distance.numpy()[0]))
        else:
            signed_distance.append(-np.abs(distance.numpy()[0]))

    return interior_label, points_trace, position_label, signed_distance, distance_df
            
def plot_interiorlabel(background_trace, point_trace, plots_dir='./output/visualization/M1M2/', filename='_test.html'):

    layout = go.Layout(margin=dict(l=0,
                               r=0,
                               b=0,
                               t=0))

    ax_style = dict(showbackground = False,
                    backgroundcolor="rgb(240, 240, 240)",
                    showgrid=True,
                    zeroline=False,
                    showticklabels=True)
    
    fig = go.Figure(data=background_trace + [point_trace], layout=layout)
    
    fig.update_layout(scene=dict(xaxis=ax_style, 
                                 yaxis=ax_style, 
                                 zaxis=ax_style,
                                ),
                     legend={'itemsizing': 'constant'})

    output_path = os.path.join(plots_dir, filename)
    fig.write_html(output_path)
    IFRAME_command = "IFrame(src='{}', width=800, height=800)".format(output_path)
    return IFRAME_command

def compute_weighted_histogram(distance_list, IDlist, nbins=30):

    combined_min = 1e10
    combined_max = -1e10

    nsample = len(distance_list)
    for dist in distance_list:
        if np.min(dist) < combined_min:
            combined_min = np.min(dist)
        if np.max(dist) > combined_max:
            combined_max = np.max(dist)
    
    bin_edges = np.linspace(combined_min, combined_max, num=nbins+1)
    histcount_dflist = []
    for i, dist in enumerate(distance_list):
        hist_counts, dist_bin = np.histogram(dist, bins=bin_edges)
        # print(hist_counts)
        normalized_hist_counts = hist_counts/np.sum(hist_counts)

        histcount_df = pd.DataFrame({'distance': dist_bin[:-1],
                                     'counts':normalized_hist_counts,
                                     'subject':IDlist[i]})
        histcount_dflist.append(histcount_df)

    return histcount_dflist


def plot_histogram(df_list, exclude_outliner=0):
    ncell_order = np.argsort([df.shape[0] for df in df_list])

    if len(df_list[exclude_outliner:]) > 1:
         df_stack = pd.concat(df_list[exclude_outliner:])
    else:
        df_stack = df_list[0]

    fig, ax = pyplot.subplots(figsize=(7,4))
    sns.barplot(data=df_stack, x='distance', y='counts', ax=ax, errorbar=None)
    xticks = ax.get_xticks()
    xlabel = ax.get_xticklabels()
    xlabel = [np.around(float(x.get_text()), decimals=2) for x in xlabel]

    nbins = len(xlabel)
    skip = int(nbins/20)
    
    ax.set_xticks(xticks[::skip])
    ax.set_xticklabels(xlabel[::skip], rotation=90)
    bottom, top = ax.get_ylim()
    pyplot.show()

def resampling(samples, num_samples=None, random_seed=0):

    num_distributions = len(samples)
    
    probability = [ np.ones_like(dist)/len(dist) for dist in samples ]
    probability = np.concatenate(probability) / num_distributions

    if num_samples is None:
        num_samples = len(probability)
    
    # print(num_samples, sum(probability))

    np.random.seed(random_state)

    resamples = np.random.choice(np.concatenate(samples),
                                 p=probability,
                                 size=num_samples)

    return resamples, probability


def compute_weighted_distance(distance_list, IDlist, exclude_outliner=0, version='resample', num_samples=None, verbose=True, random_seed=0):
    if verbose:
        print([len(dist) for dist in distance_list])
    ncell_order = np.argsort([len(dist) for dist in distance_list])
    selected_idx = ncell_order[exclude_outliner:]
    if len(selected_idx) > 1:
        distance_list = [distance_list[i] for i in selected_idx]
        selected_subject = [IDlist[i] for i in selected_idx]
        if verbose:
            print(selected_subject)
            print(selected_idx)
            print([len(dist) for dist in distance_list])
    else:
        raise ValueError("Too many subjects were excluded")

    if version=='reweight':
        # Calculate effective sample size for each distribution
        effective_sample_sizes = [len(dist) / np.var(dist) for dist in distance_list]
    
        # Desired effective sample size for combined distribution
        desired_effective_sample_size = np.mean(effective_sample_sizes)
    
        # Calculate weights for each distribution
        weights = [desired_effective_sample_size / ess for ess in effective_sample_sizes]
    
        # Reweight the data for each distribution
        reweighted_distributions = [np.array(dist) * weight for dist, weight in zip(distance_list, weights)]
        
        # Combine the reweighted distributions
        combined_distribution = np.concatenate(reweighted_distributions)
        if verbose:
            print(weights)

    elif version=='resample':
        combined_distribution, probability = resampling(
            distance_list, num_samples=num_samples, random_seed=random_state
        )
    return combined_distribution



if __name__ == '__main__':


    # setup directory for images
    savedir_plot = './output/visualization/'
    os.makedirs(savedir_plot, exist_ok=True)

    # Contour vertex loading
    import xml.etree.ElementTree as ET
    tree = ET.parse('./data/cltr_1205_Ana16_rabSC_874_0116_CountsByAna004_output -M1_M2.xml')
    root = tree.getroot()    
    PM_L = load_coord(root, name='Primary motor area-L')
    PM_R = load_coord(root, name='Primary motor area-R')
    SM_L = load_coord(root, name='Secondary motor area-L')
    SM_R = load_coord(root, name='Secondary motor area-R')
    
    overlap_L = []
    dist_list = []
    for i in range(len(PM_L)):
        dist = np.min(np.sum((SM_L - np.tile(PM_L[i], (len(SM_L),1)))**2, axis=1))
        dist_list.append(dist)
        if dist <= np.sqrt(900000):
            overlap_L.append(PM_L[i])
    overlap_L = np.vstack(overlap_L)
    
    overlap_R = []
    dist_list = []
    for i in range(len(PM_R)):
        dist = np.min(np.sum((SM_R - np.tile(PM_R[i], (len(SM_R),1)))**2, axis=1))
        dist_list.append(dist)
        if dist <= np.sqrt(900000):
            overlap_R.append(PM_R[i])
    overlap_R = np.vstack(overlap_R)
    
    # Wall estimation (left)
    radius_ratio = 10
    nb_normal = 50
    nface = 15000
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(overlap_L)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(nb_normal)
    
    wall_normal = np.asarray(pcd.normals)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = radius_ratio * avg_dist
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
    mesh = o3d.geometry.TriangleMesh.remove_degenerate_triangles(mesh)
    
    L_wall_mesh = mesh.merge_close_vertices(eps=100)
    face = np.asarray(L_wall_mesh.triangles)
    vertex = np.asarray(L_wall_mesh.vertices)
    
    wall_vertex_L = np.asarray(L_wall_mesh.vertices)
    wall_mesh_L = go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:,0],
            y=vertex[:,1],
            z=vertex[:,2],
            colorbar_title='Left wall points',
    
            # i, j and k give the vertices of triangles
            i = face[:,0],
            j = face[:,1],
            k = face[:,2],
            colorscale ='Greys', opacity=0.5, showscale=False, hoverinfo='skip',
            name='L-wall', showlegend=True
        )
    
    wall_meshpt_L = go.Scatter3d(x=overlap_L[:,0],
                                 y=overlap_L[:,1],
                                 z=overlap_L[:,2],
                                 mode='markers', marker=dict(size=1, opacity=0.1),
                                 name='wall surface vertices',
                                 marker_color='black')

    # Wall estimation (right)
    radius_ratio = 15
    nb_normal = 50
    nface = 15000
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(overlap_R)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(nb_normal)
    
    wall_normal = np.asarray(pcd.normals)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = radius_ratio * avg_dist
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
    mesh = o3d.geometry.TriangleMesh.remove_degenerate_triangles(mesh)
    R_wall_mesh = mesh.merge_close_vertices(eps=50)
    
    face = np.asarray(R_wall_mesh.triangles)
    vertex = np.asarray(R_wall_mesh.vertices)
    wall_vertex_R = np.asarray(R_wall_mesh.vertices)
    wall_mesh_R = go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:,0],
            y=vertex[:,1],
            z=vertex[:,2],
            colorbar_title='Left wall points',
    
            # i, j and k give the vertices of triangles
            i = face[:,0],
            j = face[:,1],
            k = face[:,2],
            colorscale ='Greys', opacity=0.5, showscale=False, hoverinfo='skip',
            name='R-wall', showlegend=True
        )
    
    wall_meshpt_R = go.Scatter3d(x=overlap_R[:,0],
                              y=overlap_R[:,1],
                              z=overlap_R[:,2],
                              mode='markers', marker=dict(size=1, opacity=0.1),
                              name='wall surface vertices',
                              marker_color='black')

    
    ## compute shortest distance towards the wall

    # generate watertight mesh surface: SM_L
    radius_ratio = 25
    nb_normal = 20
    nface = 1000

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(SM_L)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(nb_normal)
    
    wall_normal = np.asarray(pcd.normals)
    
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = radius_ratio * avg_dist
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    SM_L_mesh = mesh.simplify_quadric_decimation(nface)
    face = np.asarray(SM_L_mesh.triangles)
    vertex = np.asarray(SM_L_mesh.vertices)
    
    SM_vertex_L = np.asarray(SM_L_mesh.vertices)
    SM_mesh_L = go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:,0],
            y=vertex[:,1],
            z=vertex[:,2],
            colorbar_title='SM-L points',
    
            # i, j and k give the vertices of triangles
            i = face[:,0],
            j = face[:,1],
            k = face[:,2],
            colorscale ='Greys', opacity=0.5, showscale=False, hoverinfo='skip',
            name='SM_mesh-L'
        )
    
    SM_mesh_L_transparent = go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:,0],
            y=vertex[:,1],
            z=vertex[:,2],
            colorbar_title='SM-L points',
    
            # i, j and k give the vertices of triangles
            i = face[:,0],
            j = face[:,1],
            k = face[:,2],
            colorscale ='Greys', color='Grey', opacity=0.2, showscale=False, hoverinfo='skip',
            name='SM_mesh_L', showlegend=True
        )

    # generate watertight mesh surface: SM_R
    radius_ratio = 25
    nb_normal = 10
    nface = 1500

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(SM_R)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(nb_normal)
    
    wall_normal = np.asarray(pcd.normals)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = radius_ratio * avg_dist
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    SM_R_mesh = mesh.simplify_quadric_decimation(nface)
    face = np.asarray(SM_R_mesh.triangles)
    vertex = np.asarray(SM_R_mesh.vertices)
    
    SM_vertex_R = np.asarray(SM_R_mesh.vertices)
    SM_mesh_R = go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:,0],
            y=vertex[:,1],
            z=vertex[:,2],
            colorbar_title='SM-R points',
    
            # i, j and k give the vertices of triangles
            i = face[:,0],
            j = face[:,1],
            k = face[:,2],
            colorscale ='Greys', opacity=0.5, showscale=False, hoverinfo='skip',
            name='SM_mesh-R'
        )
    
    SM_mesh_R_transparent = go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:,0],
            y=vertex[:,1],
            z=vertex[:,2],
            colorbar_title='SM-R points',
    
            # i, j and k give the vertices of triangles
            i = face[:,0],
            j = face[:,1],
            k = face[:,2],
            colorscale ='Greys', color='Grey', opacity=0.1, showscale=False, hoverinfo='skip',
            name='SM_mesh_R', showlegend=True
        )

    # generate watertight mesh surface: PM_L
    radius_ratio = 45
    nb_normal = 20
    nface = 2500
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(PM_L)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(nb_normal)
    
    wall_normal = np.asarray(pcd.normals)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = radius_ratio * avg_dist
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    PM_L_mesh = mesh.simplify_quadric_decimation(nface)
    face = np.asarray(PM_L_mesh.triangles)
    vertex = np.asarray(PM_L_mesh.vertices)
    
    PM_vertex_L = np.asarray(PM_L_mesh.vertices)
    PM_mesh_L = go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:,0],
            y=vertex[:,1],
            z=vertex[:,2],
            colorbar_title='PM-L points',
    
            # i, j and k give the vertices of triangles
            i = face[:,0],
            j = face[:,1],
            k = face[:,2],
            colorscale ='Greys', opacity=0.5, showscale=False, hoverinfo='skip',
            name='PM_mesh-L'
        )
    
    PM_mesh_L_transparent = go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:,0],
            y=vertex[:,1],
            z=vertex[:,2],
            colorbar_title='PM-L points',
    
            # i, j and k give the vertices of triangles
            i = face[:,0],
            j = face[:,1],
            k = face[:,2],
            colorscale ='Greys', color='Grey', opacity=0.2, showscale=False, hoverinfo='skip',
            name='PM_mesh_L', showlegend=True
        )

    # generate watertight mesh surface: PM_R
    radius_ratio = 45
    nb_normal = 20
    nface = 2500
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(PM_R)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(nb_normal)
    
    wall_normal = np.asarray(pcd.normals)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = radius_ratio * avg_dist

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    PM_R_mesh = mesh.simplify_quadric_decimation(nface)
    face = np.asarray(PM_R_mesh.triangles)
    vertex = np.asarray(PM_R_mesh.vertices)
    
    PM_vertex_R = np.asarray(PM_R_mesh.vertices)
    PM_mesh_R = go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:,0],
            y=vertex[:,1],
            z=vertex[:,2],
            colorbar_title='PM-R points',
    
            # i, j and k give the vertices of triangles
            i = face[:,0],
            j = face[:,1],
            k = face[:,2],
            colorscale ='Greys', opacity=0.5, showscale=False, hoverinfo='skip',
            name='PM_mesh-R'
        )
    
    PM_mesh_R_transparent = go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:,0],
            y=vertex[:,1],
            z=vertex[:,2],
            colorbar_title='PM-R points',
    
            # i, j and k give the vertices of triangles
            i = face[:,0],
            j = face[:,1],
            k = face[:,2],
            colorscale ='Greys', color='Grey', opacity=0.2, showscale=False, hoverinfo='skip',
            name='PM_mesh_R', showlegend=True
        )

    SM_R_mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(SM_R_mesh)
    SM_L_mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(SM_L_mesh)
    print(SM_R_mesh_legacy, SM_L_mesh_legacy)
    
    scene_SM_L = o3d.t.geometry.RaycastingScene()
    _ = scene_SM_L.add_triangles(SM_L_mesh_legacy)
    scene_SM_R = o3d.t.geometry.RaycastingScene()
    _ = scene_SM_R.add_triangles(SM_R_mesh_legacy)

    R_wall_mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(R_wall_mesh)
    L_wall_mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(L_wall_mesh)
    print(R_wall_mesh_legacy, L_wall_mesh_legacy)
    
    scene_wall_L = o3d.t.geometry.RaycastingScene()
    _ = scene_wall_L.add_triangles(L_wall_mesh_legacy)
    scene_wall_R = o3d.t.geometry.RaycastingScene()
    _ = scene_wall_R.add_triangles(R_wall_mesh_legacy)


    # espilon (fine-tuning the wall location)
    # after manually inspection for each subjects
    eps_list = [40,0,100,-80,-10,-120,-120,-40,40,0,0,
                0,-40,0,0,0,-40,0,0,0,0,-40]
    
    M1M2_count_df = pd.DataFrame(columns=['subject', 'total', 'SM_L', 'SM_R', 'PM_L', 'PM_R'])



    # load neurons spatial location data
    dir_data = "./data/processed/combined/*.csv"
    group1, group2, merge_df, IDlist = load_data(dir_data)

    group1_ID = [IDlist.index(ID) for ID in group1]
    group2_ID = [IDlist.index(ID) for ID in group2]
    
    ROIlist = ['M1/M2']
    available_ID, _, _, _ = load_groupROIlist_coord(IDlist, merge_df, ROIlist=ROIlist)
    
    cell_list = []
    point_list = []
    for ID in available_ID:
        celldf = merge_df.loc[ID]
        atlas_xyz = celldf.loc[celldf['marker'].isin(ROIlist)][['atlas_x', 'atlas_y', 'atlas_z']].values
    
        atlas_xyz[:,1] *= -1
        atlas_xyz[:,2] *= -1
        point_list.append(atlas_xyz)
        
        if ID in group1: symbol = 'circle'
        else: symbol = 'cross'
        subj_cell = go.Scatter3d(x=atlas_xyz[:,0], y=atlas_xyz[:,1], z=atlas_xyz[:,2],
                                 mode='markers', marker=dict(size=3, opacity=1), name=ID, marker_symbol=symbol)
        cell_list.append(subj_cell)
    
    
    for i in range(len(point_list)):
        
        (
            interior_label,
            points_trace,
            position_label,
            signed_distance,
            distance_df
        ) = compute_signed_distance(
            scene_SM_L,
            scene_wall_L,
            scene_SM_R, 
            scene_wall_R, 
            point_list[i], 
            eps=eps_list[i]
        )
        
        print(IDlist[i], point_list[i].shape[0])
        print('SM_L :', len(distance_df.loc[distance_df['region']=='SM_L']),
              'SM_R :', len(distance_df.loc[distance_df['region']=='SM_R']),
              'PM_L :', len(distance_df.loc[distance_df['region']=='PM_L']),
              'PM_R :', len(distance_df.loc[distance_df['region']=='PM_R']),)
    
        M1M2_count_df = pd.concat(
            [M1M2_count_df, pd.DataFrame({
                'subject': IDlist[i],
                'total': [distance_df.shape[0]],
                'SM_L' : [len(distance_df.loc[distance_df['region']=='SM_L'])],
                'SM_R' : [len(distance_df.loc[distance_df['region']=='SM_R'])],
                'PM_L' : [len(distance_df.loc[distance_df['region']=='PM_L'])],
                'PM_R' : [len(distance_df.loc[distance_df['region']=='PM_R'])]
                })
             ], ignore_index=True
        )


    nperm = 10000
    available_ID = group1_ID + group2_ID
    resample_list, rlen = generate_resample(
        available_ID, 
        group1_ID, 
        group2_ID, 
        nperm=nperm   
    )

    
    trace_list = []
    position_list = []
    dist_list = []
    
    for i in range(len(point_list)):
    
        (
            interior_label,
            points_trace, 
            position_label, 
            signed_distance, _
        )=  compute_signed_distance(
            scene_SM_L, 
            scene_wall_L,                                                                        
            scene_SM_R, 
            scene_wall_R,                                                      
            point_list[i], 
            eps=eps_list[i]
        )
        
        trace_list.append(points_trace)
        position_list.append(position_label)
        dist_list.append(signed_distance)

    metric = []
    for b in tqdm(range(rlen)):
        rg1 = resample_list[b][0]
        rg2 = resample_list[b][1]
        rdist_1 = [dist_list[k] for k in rg1]
        rdist_2 = [dist_list[k] for k in rg2]
        group1_dist = compute_weighted_distance(
            rdist_1,
            rg1,
            exclude_outliner=4, num_samples=10000,
            version='resample', verbose=False, random_seed=random_state+1
        )
        group2_dist = compute_weighted_distance(
            rdist_2,
            rg2,
            exclude_outliner=4, num_samples=10000,
            version='resample', verbose=False, random_seed=random_state+1
        )
        
        T_observe = np.abs(np.mean(group1_dist) - np.mean(group2_dist))
    
        if b == 0:
            group1_observe = group1_dist
            group2_observe = group2_dist
        
        metric.append(T_observe)
    
    pval = (np.sum(metric > metric[0])+1)/(len(metric)+1)
    print('Wall distance, M1/M2, p-value:', pval)
    
    group1_observe_df = pd.DataFrame(group1_observe, columns=['distance'])
    group1_observe_df['group'] = 'Group 1'
    group2_observe_df = pd.DataFrame(group2_observe, columns=['distance'])
    group2_observe_df['group'] = 'Group 2'
    observe_df = pd.concat([group1_observe_df, group2_observe_df], ignore_index=True)


    fig, ax = pyplot.subplots(figsize=(7,3), ncols=2)
    sns.histplot(metric, ax=ax[0], bins=30, element='step', alpha=0.3, color='black')
    bottom, top = ax[0].get_ylim()
    baseline = metric[0]
    ax[0].vlines(baseline, bottom, top, colors='r')
    ax[0].text(baseline*0.95, top/6, 'observation', c='r', rotation=270, fontsize=12)
    title = 'Permutation test \n p-val={:.2e}'.format(pval)
    ax[0].set_title(title, fontsize=10)
    ax[0].set_xticks([])
    
    sns.histplot(observe_df, x='distance',hue='group', bins=30, ax=ax[1], color='r', alpha=0.4, stat='proportion', kde=False)
    ax[1].set_title('Estimated observation \n (weighted)')
    # ax[1].legend()
    # ax[1].set_yticks([])
    # ax[1].set_ylabel('Count')
    ax[1].set_xlabel('Signed distance')
    
    pyplot.tight_layout()
    
    
    pyplot.savefig(os.path.join(savedir_plot, 'M1M2_shortest_distance.png'))
    pyplot.savefig(os.path.join(savedir_plot, 'M1M2_shortest_distance.eps'), format='eps')
    pyplot.savefig(os.path.join(savedir_plot, 'M1M2_shortest_distance.svg'), format='svg')
    

    ROIlist = ['M1/M2']
    available_ID, _, _, _ = load_groupROIlist_coord(IDlist, merge_df, ROIlist=ROIlist)
    
    g1_point_list = []
    g2_point_list = []
    for ID in available_ID:
        celldf = merge_df.loc[ID]
        atlas_xyz = celldf.loc[celldf['marker'].isin(ROIlist)][['atlas_x', 'atlas_y', 'atlas_z']].values
    
        atlas_xyz[:,1] *= -1
        atlas_xyz[:,2] *= -1
    
        if ID in group1:
            g1_point_list.append(atlas_xyz)
        else:
            g2_point_list.append(atlas_xyz)
    g1_point_list = np.vstack(g1_point_list)
    g2_point_list = np.vstack(g2_point_list)
    
    g1_cell = go.Scatter3d(
        x=g1_point_list[:,0], y=g1_point_list[:,1], z=g1_point_list[:,2],
        mode='markers', marker=dict(size=3, opacity=1, color='black'),
        name='Group 1', marker_symbol='circle'
    )
        
    
    g2_cell = go.Scatter3d(
        x=g2_point_list[:,0], y=g2_point_list[:,1], z=g2_point_list[:,2],
        mode='markers', marker=dict(size=3, opacity=1, color='magenta'), 
        name='Group 2', marker_symbol='circle'
    )
                                 
    group_cell_list = [g1_cell, g2_cell]
    
    
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    
    ax_style = dict(showbackground = False,
                    backgroundcolor="rgb(240, 240, 240)",
                    showgrid=True,
                    zeroline=False,
                    showticklabels=True)
    
    # [wall_mesh_R] + [wall_mesh_L]
    # data_R+data_L
    MESH_list = [PM_mesh_R_transparent, PM_mesh_L_transparent, SM_mesh_R_transparent, SM_mesh_L_transparent]
    fig = go.Figure(data= group_cell_list + MESH_list + [wall_mesh_R] + [wall_mesh_L], layout=layout)
    
    fig.update_layout(
        scene=dict(xaxis=ax_style, yaxis=ax_style, zaxis=ax_style,
                   # camera_eye=dict(x=1.85, y=1.85, z=1)
                   ),
        legend={'itemsizing': 'constant', 'x':0.99, 'y':0.85}
    )
    
    fig.write_html(os.path.join(savedir_plot, "Contour_M1M2-group.html"))
    