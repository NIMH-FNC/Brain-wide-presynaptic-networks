# system & data loading
import os
import pandas as pd
import glob
import pickle

# computational packages
import numpy as np

# misc
import time
import random
from tqdm import tqdm


def _ID2filepath(cell_df, ID):
    # ID should be of the form 'anaXX'
    return cell_df.loc[cell_df['ID']==ID]['filename'].values[0]

def load_groupROIlist_coord(IDlist, merge_df, ROIlist):
        
    coordinate = []
    ncell = []
    groupweight = []
    available_ID = []
        
    for (i, ID) in enumerate(IDlist):
        subjcell = merge_df.loc[ID]
        cell_number = subjcell.loc[ subjcell['marker'].isin(ROIlist) ].shape[0]
        if cell_number > 0:
            ncell.append( cell_number )
            available_ID.append(ID)
            
    ratio = [np.max(ncell)/k for k in ncell]
            
    for (i, ID) in enumerate(available_ID):
        celldf = merge_df.loc[ID]
        atlas_xyz = celldf.loc[celldf['marker'].isin(ROIlist)][['atlas_x', 'atlas_y', 'atlas_z']].values
        coordinate.append( atlas_xyz )
        groupweight.append( np.ones(ncell[i])*ratio[i] )

    coordinate = np.vstack((coordinate))
    groupweight = np.hstack((groupweight))
    return available_ID, coordinate, ncell, groupweight


def load_data(dir_data):

    # dir_data = "./data/processed/combined/*.csv"

    csv_list = glob.glob(dir_data)
    
    group1 = ['ana02', 'ana04', 'ana07', 'ana08', 'ana09',
              'ana10', 'ana19', 'ana20', 'ana22', 'ana24', 'ana26']
    group2 = ['ana11', 'ana12', 'ana13', 'ana14', 'ana15',
              'ana16', 'ana18', 'ana21', 'ana25', 'ana27', 'ana28']
    
    cell_df = []
    for file in csv_list:
        for g in group1:
            if g in file.lower():
                cell_df.append([g, 1, file])
        for g in group2:
            if g in file.lower():
                cell_df.append([g, 2, file])
    
    cell_df = pd.DataFrame(cell_df, columns=['ID', 'group', 'filename'])
    print('Total csv files loaded: {}'.format(cell_df.shape[0]))
    
    pieces = dict()
    IDlist = []
    for ID in (group1 + group2):
        try:
            pieces[ID] = pd.read_csv( _ID2filepath(cell_df, ID), index_col=[0] )
            if len(pieces[ID]) > 0:
                IDlist.append(ID)
        except:
            pass
    merge_df = pd.concat(pieces)

    return group1, group2, merge_df, IDlist
