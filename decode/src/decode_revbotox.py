# 1786_ArchT_Session3
# 1786_ArchT_Session5
# 1786_ArchT_Session7

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib import pyplot as plt
from scipy.io import savemat
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from pymatreader import read_mat

from util import normalize, change_binsize


data_dir = Path.cwd() / "data/Rev_Botox"
output_dir = Path.cwd() / "outputs/Rev_Botox"
output_dir.mkdir(exist_ok=True, parents=True)

dataset = [
    {'subject': "Mouse1786_ArchT", 'sessions': ["10182023_Mouse1786_ArchT_Session3_BotoxFOV1_Div1_Plane01_vars.mat", "10192023_Mouse1786_ArchT_Session5_BotoxFOV1_Div1_Plane01_vars.mat", "10212023_Mouse1786_ArchT_Session7_BotoxFOV1_Div1_Plane01_vars.mat"], 'mapping': ["Mouse1786_ArchT_Session3_5.mat", "Mouse1786_ArchT_Session5_7.mat"]},
    {'subject': "Mouse1786_ArchT", 'sessions': ["10182023_Mouse1786_ArchT_Session4_BotoxFOV2_Div1_Plane01_vars.mat", "10192023_Mouse1786_ArchT_Session6_BotoxFOV2_Div1_Plane01_vars.mat", "10212023_Mouse1786_ArchT_Session8_BotoxFOV2_Div1_Plane01_vars.mat"], 'mapping': ["Mouse1786_ArchT_Session4_6.mat", "Mouse1786_ArchT_Session6_8.mat"]},
    {'subject': "Mouse1788_Halo", 'sessions': ["10182023_Mouse1788_Halo_Session4_BotoxFOV2_Div1_Plane01_vars.mat", "10192023_Mouse1788_Halo_Session6_BotoxFOV2_Div1_Plane01_vars.mat", "10212023_Mouse1788_Halo_Session8_BotoxFOV2_Div1_Plane01_vars.mat"], 'mapping': ["Mouse1788_Halo_Session4_6.mat", "Mouse1788_Halo_Session6_8.mat"]}
]


np.random.seed(0)
alphas = np.logspace(-1, 18, 10, endpoint=True)  # l2 penalty grid
cv = None  # None for GCV


def decoder(X, y, Z=None, **kwargs):
    """Fit decoder
    param Z: all cells 
    """
    if Z is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        ridge_Z = None
        rsq_Z_training = None
        rsq_Z_test = None
    else:
        Z_train, Z_test, X_train, X_test, y_train, y_test = train_test_split(Z, X, y)
        ridge_Z = RidgeCV(**kwargs).fit(Z_train, y_train)
        rsq_Z_training = ridge_Z.score(Z_train, y_train)
        rsq_Z_test = ridge_Z.score(Z_test, y_test)

    ridge_X = RidgeCV(**kwargs).fit(X_train, y_train)
    rsq_X_training = ridge_X.score(X_train, y_train)
    rsq_X_test = ridge_X.score(X_test, y_test)
    return ridge_X, rsq_X_training, rsq_X_test, rsq_Z_training, rsq_Z_test, ridge_Z


def get_mapping(mappings):
    cell_map1 = read_mat(data_dir/mappings[0])
    cell_map2 = read_mat(data_dir/mappings[1])
    mapping1 = cell_map1['roiMatchData']['allSessionMapping'][()] - 1  # translate matlab index
    mapping2 = cell_map2['roiMatchData']['allSessionMapping'][()] - 1  # translate matlab index
    return mapping1, mapping2


def get_session(sess, window=15, check_nan=True):
    # 15 bins = 15 * 33.3ms = 500ms
    path = sess['path']
    path_group = sess['group']
    
    data = read_mat(path)
    group = read_mat(path_group)

    spont = group['groups']['spontAll']

    trial_names = data['fTrial'].keys()
    trials = data['fTrial'].values()
    iscell = data['iscell'][()][:, 0] == 1
    caTrialsMask = data['caTrialsMask']  # spontaneous
    caTrialsMaskLED = data['caTrialsMaskLED']
    caTrialsMaskLEDCL = data['caTrialsMaskLEDCL']
    
    lSpks = []
    lSpeed = []
    lWhisk1 = []
    lStim = []
    lSound = []
    spontaneous = []
    drug = []
    tnames = []
    for trial, tname, mask, led, ledcl in zip(trials, trial_names, caTrialsMask, caTrialsMaskLED, caTrialsMaskLEDCL):
        if tname not in spont:
            print(f'{tname} excluded')
            continue  # exclude specified trials

        tnames.append(tname)
        fSpks = trial['fSpks'][()]
        fSpeed = trial['fSpeed'][()]
        fWhisk1 = trial['fWhisk1'][()]
        fStim = trial['fStim'][()]
        fSound = trial['fSound'][()]
        assert fSpks.shape[0] == fSpeed.shape[0]
        assert np.all(np.isfinite(fSpeed))
        assert fSpks.shape[0] == fWhisk1.shape[0]
        if check_nan:
            valid = np.isfinite(fSpeed) & np.isfinite(fWhisk1)
        else:
            valid = np.ones_like(fSpeed, dtype=bool)
        fSpks = change_binsize(fSpks[valid, :], window, 0)
        fSpeed = change_binsize(fSpeed[valid], window, 0)
        fWhisk1 = change_binsize(fWhisk1[valid], window, 0)
        fStim = change_binsize(fStim[valid], window, 0)
        fSound = change_binsize(fSound[valid], window, 0)

        lSpks.append(fSpks)
        lSpeed.append(fSpeed)
        lWhisk1.append(fWhisk1)
        lStim.append(fStim)
        lSound.append(fSound)

        spontaneous.append(0)
        drug1 = int(led)
        drug2 = int(ledcl)
        if drug1 > 0:
            if drug2 > 0:
                raise ValueError('Positive Drug 1 and Drug 2')
            d = 1
        elif drug2 > 0:
            d = 2
        else:
            d = 0
        drug.append(d)

    return lSpks, lSpeed, lWhisk1, lStim, lSound, iscell, spontaneous, drug, tnames


def adapt(path):
    subdir = path.replace("_vars.mat", "_Analysed")
    group_file = path.replace("_vars.mat", "_groups.mat")
    return {'path': data_dir/ subdir / path, 'group': data_dir/ subdir / group_file}


def get_data(data):
    subject = data['subject']
    sessions = data['sessions']

    cell_map1, cell_map2 = get_mapping(data['mapping'])
    cell_map = [(pair[0], *np.squeeze(cell_map2[cell_map2[:, 0] == pair[1]])) for pair in cell_map1]
    cell_map = np.array([m for m in cell_map if len(m) == 3])

    fSpks0, fSpeed0, fWhisk0, fStim0, fSound0, iscell0, spontaneous0, _, tnames0 = get_session(adapt(sessions[0]))
    fSpks1, fSpeed1, fWhisk1, fStim1, fSound1, iscell1, spontaneous1, _, tnames1 = get_session(adapt(sessions[1]))
    fSpks2, fSpeed2, fWhisk2, fStim2, fSound2, iscell2, spontaneous2, _, tnames2 = get_session(adapt(sessions[2]))

    def make_data(cell_map, iscell, fSpks, spontaneous, fSpeed, fWhisk, tnames):
        return {
            'cell_map': cell_map,
            'spk': [t[:, iscell] for t, s in zip(fSpks, spontaneous) if s == 0],
            'speed': [t for t, s in zip(fSpeed, spontaneous) if s == 0],
            'whisk': [t for t, s in zip(fWhisk, spontaneous) if s == 0],
            'tnames': [t for t, s in zip(tnames, spontaneous) if s == 0],
        }
    
    data0 = make_data(cell_map[:, 0], iscell0, fSpks0, spontaneous0, fSpeed0, fWhisk0, tnames0)
    data1 = make_data(cell_map[:, 1], iscell1, fSpks1, spontaneous1, fSpeed1, fWhisk1, tnames1)
    data2 = make_data(cell_map[:, 2], iscell2, fSpks2, spontaneous2, fSpeed2, fWhisk2, tnames2)

    fSpks0 = np.row_stack([t for t, s in zip(fSpks0, spontaneous0) if s == 0])
    fSpeed0 = np.concatenate([t for t, s in zip(fSpeed0, spontaneous0) if s == 0])
    fWhisk0 = np.concatenate([t for t, s in zip(fWhisk0, spontaneous0) if s == 0])

    fSpks1 = np.row_stack([t for t, s in zip(fSpks1, spontaneous1) if s == 0])
    fSpeed1 = np.concatenate([t for t, s in zip(fSpeed1, spontaneous1) if s == 0])
    fWhisk1 = np.concatenate([t for t, s in zip(fWhisk1, spontaneous1) if s == 0])

    fSpks2 = np.row_stack([t for t, s in zip(fSpks2, spontaneous2) if s == 0])
    fSpeed2 = np.concatenate([t for t, s in zip(fSpeed2, spontaneous2) if s == 0])
    fWhisk2 = np.concatenate([t for t, s in zip(fWhisk2, spontaneous2) if s == 0])
    
    cSpks0 = fSpks0[:, iscell0]
    cSpks1 = fSpks1[:, iscell1]
    cSpks2 = fSpks2[:, iscell2]
    
    # don't have to normalize firing rate or map cells
    X0 = C0 = cSpks0
    X1 = C1 = cSpks1
    X2 = C2 = cSpks2

    # mapped cells
    X0 = C0[:, cell_map[:, 0]]
    X1 = C1[:, cell_map[:, 1]]
    X2 = C2[:, cell_map[:, 2]]

    # normalize firing rates
    zscore_func0 = normalize(C0, zscore=True, return_func=True)
    zscore_func1 = normalize(C1, zscore=True, return_func=True)
    zscore_func2 = normalize(C2, zscore=True, return_func=True)
    C0 = zscore_func0(C0)
    C1 = zscore_func1(C1)
    C2 = zscore_func2(C2)
    
    X0 = normalize(X0, True)
    X1 = normalize(X1, True)
    X2 = normalize(X2, True)
    
    data0['zscore'] = zscore_func0
    data1['zscore'] = zscore_func1
    data2['zscore'] = zscore_func2

    # np.savetxt(f'outputs/{pair}_mapped_cell.csv', cell_map + 1, delimiter=',', fmt='%d')  # save mapped cells

    return subject, sessions, C0, X0, fSpeed0, fWhisk0, C1, X1, fSpeed1, fWhisk1, C2, X2, fSpeed2, fWhisk2, data0, data1, data2


# %%  Across day
def decode_crossday(data):
    r2s = pd.DataFrame(columns=['pair', 'dv', 'R2_1', 'R2_2', 'R2_1_2', 'R2_2_1'])

    subject, sessions, C0, X0, fSpeed0, fWhisk0, C1, X1, fSpeed1, fWhisk1, C2, X2, fSpeed2, fWhisk2, data0, data1, data2 = get_data(data)

    # %%
    for dv in ['speed']:
        if dv == 'speed':
            y0 = fSpeed0
            y1 = fSpeed1
            y2 = fSpeed2
        else:
            y0 = fWhisk0
            y1 = fWhisk1
            y2 = fWhisk2
       
        # zscore behavioral variables if across days
        behav_normalizer0 = normalize(y0, True, return_func=True)
        behav_normalizer1 = normalize(y1, True, return_func=True)
        behav_normalizer2 = normalize(y2, True, return_func=True)

        y0 = behav_normalizer0(y0)
        y1 = behav_normalizer1(y1)
        y2 = behav_normalizer2(y2)

        ridge0, rsq_0_training, rsq_0_test, rsq_0_training_cell, rsq_0_test_cell, _ = decoder(X0, y0, C0, alphas=alphas, cv=cv, fit_intercept=False)
        ridge1, rsq_1_training, rsq_1_test, rsq_1_training_cell, rsq_1_test_cell, _ = decoder(X1, y1, C1, alphas=alphas, cv=cv, fit_intercept=False)
        ridge2, rsq_2_training, rsq_2_test, rsq_2_training_cell, rsq_2_test_cell, _ = decoder(X2, y2, C2, alphas=alphas, cv=cv, fit_intercept=False)

        table = [
            [subject, dv, sessions[0], 'training', rsq_0_training_cell],
            [subject, dv, sessions[0], 'test', rsq_0_test_cell],
            [subject, dv, sessions[0], sessions[1], ridge0.score(X1, y1)],
            [subject, dv, sessions[0], sessions[2], ridge0.score(X2, y2)],
            [subject, dv, sessions[1], 'training', rsq_1_training_cell],
            [subject, dv, sessions[1], 'test', rsq_1_test_cell],
            [subject, dv, sessions[1], sessions[0], ridge1.score(X0, y0)],
            [subject, dv, sessions[1], sessions[2], ridge1.score(X2, y2)],
            [subject, dv, sessions[2], 'training', rsq_2_training_cell],
            [subject, dv, sessions[2], 'test', rsq_2_test_cell],
            [subject, dv, sessions[2], sessions[0], ridge2.score(X0, y0)],
            [subject, dv, sessions[2], sessions[1], ridge2.score(X1, y1)],            
        ]

    df = pd.DataFrame(table, columns=['mouse', 'dv', 'model', 'set', 'r2'])

    return df


if __name__ == '__main__':
    dfs = []
    r2s = []

    with ProcessPoolExecutor() as pool:
        dfs = list(pool.map(decode_crossday, dataset))

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_dir / "decode_bnorm.csv", index=False)  # 

