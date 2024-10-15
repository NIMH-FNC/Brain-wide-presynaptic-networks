#%% Imports
import math

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib import pyplot as plt
from scipy.io import savemat
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

from util_old import change_binsize, get_mapping, get_session, load_yml, normalize

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

def get_data(data):
    data1 = {}
    data2 = {}

    if 'pair' in data:
        # chronic and botox
        pair = data['pair']
        mouse, day1, day2 = pair.split('_')
        mouse = mouse.removeprefix('Mouse')

        cell_map = get_mapping(data['map'])
        fSpks1, fSpeed1, fWhisk1, fStim1, fSound1, iscell1, spontaneous1, _, tnames1 = get_session(data['day1'])
        fSpks2, fSpeed2, fWhisk2, fStim2, fSound2, iscell2, spontaneous2, _, tnames2 = get_session(data['day2'])
        
        data1['cell_map'] = cell_map[:, 0]
        data1['spk'] = [t[:, iscell1] for t, s in zip(fSpks1, spontaneous1) if s == 0]
        data1['speed'] = [t for t, s in zip(fSpeed1, spontaneous1) if s == 0]
        data1['whisk'] = [t for t, s in zip(fWhisk1, spontaneous1) if s == 0]
        data1['tnames'] = [t for t, s in zip(tnames1, spontaneous1) if s == 0]
        
        data2['cell_map'] = cell_map[:, 1]
        data2['spk'] = [t[:, iscell2] for t, s in zip(fSpks2, spontaneous2) if s == 0]
        data2['speed'] = [t for t, s in zip(fSpeed2, spontaneous2) if s == 0]
        data2['whisk'] = [t for t, s in zip(fWhisk2, spontaneous2) if s == 0]
        data2['tnames'] = [t for t, s in zip(tnames2, spontaneous1) if s == 0]

        fSpks0 = None
        fSpeed0 = None
        fWhisk0 = None

        fSpks1 = np.row_stack([t for t, s in zip(fSpks1, spontaneous1) if s == 0])
        fSpeed1 = np.concatenate([t for t, s in zip(fSpeed1, spontaneous1) if s == 0])
        fWhisk1 = np.concatenate([t for t, s in zip(fWhisk1, spontaneous1) if s == 0])
        # fStim1 = np.concatenate([t for t, s in zip(fStim1, spontaneous1) if s == 0])
        # fSound1 = np.concatenate([t for t, s in zip(fSound1, spontaneous1) if s == 0])

        fSpks2 = np.row_stack([t for t, s in zip(fSpks2, spontaneous2) if s == 0])
        fSpeed2 = np.concatenate([t for t, s in zip(fSpeed2, spontaneous2) if s == 0])
        fWhisk2 = np.concatenate([t for t, s in zip(fWhisk2, spontaneous2) if s == 0])
        # fStim2 = np.concatenate([t for t, s in zip(fStim2, spontaneous2) if s == 0])
        # fSound2 = np.concatenate([t for t, s in zip(fSound2, spontaneous2) if s == 0])

        cSpks1 = fSpks1[:, iscell1]
        cSpks2 = fSpks2[:, iscell2]
    else:
        # neuromod
        pre, mouse, *_ = data['path'].split('_')
        pair = data['path'].split('/')[-1]
        day1 = 'pre'
        day2 = 'post' 
        mouse = mouse.removeprefix('Mouse')
        # print(mouse, day1, day2)
        fSpks, fSpeed, fWhisk, fStim, fSound, iscell, spontaneous, drug, tnames = get_session(data)

        # Baseline
        fSpks0 = np.row_stack([t for t, s, d in zip(fSpks, spontaneous, drug) if s == 0 and d == 0])
        fSpeed0 = np.concatenate([t for t, s, d in zip(fSpeed, spontaneous, drug) if s == 0 and d == 0])
        fWhisk0 = np.concatenate([t for t, s, d in zip(fWhisk, spontaneous, drug) if s == 0 and d == 0])
        # fStim0 = np.concatenate([t for t, s, d in zip(fStim, spontaneous, drug) if s == 0 and d == 0])
        # fSound0 = np.concatenate([t for t, s, d in zip(fSound, spontaneous, drug) if s == 0 and d == 0])
        
        # Drug 1
        fSpks1 = np.row_stack([t for t, s, d in zip(fSpks, spontaneous, drug) if s == 0 and d == 1])
        fSpeed1 = np.concatenate([t for t, s, d in zip(fSpeed, spontaneous, drug) if s == 0 and d == 1])
        fWhisk1 = np.concatenate([t for t, s, d in zip(fWhisk, spontaneous, drug) if s == 0 and d == 1])
        # fStim1 = np.concatenate([t for t, s, d in zip(fStim, spontaneous, drug) if s == 0 and d == 1])
        # fSound1 = np.concatenate([t for t, s, d in zip(fSound, spontaneous, drug) if s == 0 and d == 1])

        cSpks0 = fSpks0[:, iscell]
        cSpks1 = fSpks1[:, iscell]

        # Drug 2
        if 2 in drug:
            fSpks2 = np.row_stack([t for t, s, d in zip(fSpks, spontaneous, drug) if s == 0 and d == 2])
            fSpeed2 = np.concatenate([t for t, s, d in zip(fSpeed, spontaneous, drug) if s == 0 and d == 2])
            fWhisk2 = np.concatenate([t for t, s, d in zip(fWhisk, spontaneous, drug) if s == 0 and d == 2])
            cSpks2 = fSpks2[:, iscell]
        else:
        # fStim2 = np.concatenate([t for t, s, d in zip(fStim, spontaneous, drug) if s == 0 and d == 2])
        # fSound2 = np.concatenate([t for t, s, d in zip(fSound, spontaneous, drug) if s == 0 and d == 2])
            fSpeed2 = None
            fWhisk2 = None
            cSpks2 = None
        
        # print(data['path'], drug)
        cell_map = None
    # %%
    zscore1 = None
    zscore2 = None

    C1 = cSpks1  # all cells
    C2 = cSpks2
    if cell_map is None:
        # don't have to normalize firing rate or map cells
        X1 = C1
        X2 = C2
        X0 = C0 = cSpks0
    else:
        # mapped cells
        X1 = C1[:, cell_map[:, 0]]
        X2 = C2[:, cell_map[:, 1]]
        # normalize firing rates
        zscore1 = normalize(C1, True, return_func=True)
        zscore2 = normalize(C2, True, return_func=True)
        C1 = zscore1(C1)
        C2 = zscore2(C2)
        X1 = normalize(X1, True)
        X2 = normalize(X2, True)
        X0 = C0 = None
    
    data1['zscore'] = zscore1
    data2['zscore'] = zscore2

    # np.savetxt(f'outputs/{pair}_mapped_cell.csv', cell_map + 1, delimiter=',', fmt='%d')  # save mapped cells

    return mouse, pair, day1, day2, C1, X1, fSpeed1, fWhisk1, C2, X2, fSpeed2, fWhisk2, C0, X0, fSpeed0, fWhisk0, data1, data2


# %%  Across day
def decode_crossday(data):
    df = pd.DataFrame(
        columns=['mouse', 'pair', 'cell', 'dv', 'model', 'training', 'within', 'cross'])
    r2s = pd.DataFrame(columns=['pair', 'dv', 'R2_1', 'R2_2', 'R2_1_2', 'R2_2_1'])

    mouse, pair, day1, day2, C1, X1, fSpeed1, fWhisk1, C2, X2, fSpeed2, fWhisk2, C0, X0, fSpeed0, fWhisk0, data1, data2 = get_data(data)
    # %%
    for dv in ['speed', 'whisk']:
        if dv == 'speed':
            y1 = fSpeed1
            y2 = fSpeed2
        else:
            y1 = fWhisk1
            y2 = fWhisk2
       
        # zscore behavioral variables if across days
        zscore1 = normalize(y1, True, return_func=True)
        zscore2 = normalize(y2, True, return_func=True)
        y1 = zscore1(y1)
        y2 = zscore2(y2)
        # y1 = normalize(y1, True)
        # y2 = normalize(y2, True)
        
        #
        ridge1, rsq_1_training, rsq_1_test, rsq_1_training_cell, rsq_1_test_cell, ridge_all = decoder(X1, y1, C1, alphas=alphas, cv=cv, fit_intercept=False)
        ys = []
        yhats = []
        for X, y in zip(data1['spk'], data1[dv]):
            ys.append(zscore1(y))
            yhats.append(ridge_all.predict(data1['zscore'](X)))
        ys = np.column_stack(ys)
        yhats = np.column_stack(yhats)
        savemat(f'outputs/{pair}_Day1_{dv}_decode.mat', {'true_trace': ys, 'predicted_trace': yhats, 'weight': ridge_all.coef_, 'trials': data1['tnames']})

        df = df.append(
            {
                'mouse': mouse,
                'pair': pair, 
                'cell': 'all',
                'dv': dv,
                'model': day1,
                'training': rsq_1_training_cell,
                'within': rsq_1_test_cell,
                'cross': None
            },
            ignore_index=True)

        df = df.append(
            {
                'mouse': mouse,
                'pair': pair,
                'cell': 'mapped',
                'dv': dv,
                'model': day1,
                'training': rsq_1_training,
                'within': rsq_1_test,
                'cross': ridge1.score(X2, y2)
            },
            ignore_index=True)

        ridge2, rsq_2_training, rsq_2_test, rsq_2_training_cell, rsq_2_test_cell, ridge_all = decoder(X2, y2, C2, alphas=alphas, cv=cv, fit_intercept=False)
        ys = []
        yhats = []
        for X, y in zip(data2['spk'], data2[dv]):
            ys.append(zscore2(y))
            yhats.append(ridge_all.predict(data2['zscore'](X)))
        ys = np.column_stack(ys)
        yhats = np.column_stack(yhats)
        savemat(f'outputs/{pair}_Day2_{dv}_decode.mat', {'true_trace': ys, 'predicted_trace': yhats, 'weight': ridge_all.coef_, 'trials': data2['tnames']})

        df = df.append(
            {
                'mouse': mouse,
                'pair': pair, 
                'cell': 'all',
                'dv': dv,
                'model': day2,
                'training': rsq_2_training_cell,
                'within': rsq_2_test_cell,
                'cross': None
            },
            ignore_index=True)

        df = df.append(
            {
                'mouse': mouse,
                'pair': pair, 
                'cell': 'mapped',
                'dv': dv,
                'model': day2,
                'training': rsq_2_training,
                'within': rsq_2_test,
                'cross': ridge2.score(X1, y1)
            },
            ignore_index=True)
    
        r2s = r2s.append(
            {
                'pair': pair,
                'dv': dv,
                'R1': rsq_1_training,
                'R2': rsq_2_training,
                'R2_1': rsq_1_test,
                'R2_2': rsq_2_test,
                'R2_1_2': ridge1.score(X2, y2),
                'R2_2_1': ridge2.score(X1, y1)
            },
            ignore_index=True
        )
    return df, r2s


def decode_drug(data):
    df = pd.DataFrame(
        columns=['mouse', 'session', 'dv', 'model', 'training', 'baseline', 'drug1', 'drug2'])
    # r2s = pd.DataFrame(columns=['session', 'dv', 'R2_1', 'R2_2', 'R2_1_2', 'R2_2_1'])
    mouse, pair, day1, day2, C1, X1, fSpeed1, fWhisk1, C2, X2, fSpeed2, fWhisk2, C0, X0, fSpeed0, fWhisk0, data1, data2 = get_data(data)
    # %%
    for dv in ['speed', 'whisk']:
        if dv == 'speed':
            y0 = fSpeed0
            y1 = fSpeed1
            y2 = fSpeed2
        else:
            y0 = fWhisk0
            y1 = fWhisk1
            y2 = fWhisk2
        
        y0 = normalize(y0, True)
        y1 = normalize(y1, True)
       
        ridge0, rsq_0_training, rsq_0_test, _, _ = decoder(X0, y0, None, alphas=alphas, cv=cv, fit_intercept=True)
        ridge1, rsq_1_training, rsq_1_test, _, _ = decoder(X1, y1, None, alphas=alphas, cv=cv, fit_intercept=True)
        
        df = df.append(
            {
                'mouse': mouse,
                'session': pair, 
                'dv': dv,
                'model': 'baseline',
                'training': rsq_0_training,
                'baseline': rsq_0_test,
                'drug1': ridge0.score(X1, y1),
                'drug2': np.nan if C2 is None else ridge0.score(X2, y2)
            },
            ignore_index=True)
        
        df = df.append(
            {
                'mouse': mouse,
                'session': pair, 
                'dv': dv,
                'model': 'drug1',
                'training': rsq_1_training,
                'baseline': ridge1.score(X0, y0),
                'drug1': rsq_1_test,
                'drug2': np.nan if C2 is None else ridge1.score(X2, y2)
            },
            ignore_index=True)

        if C2 is not None:
            # print('C2')
            y2 = normalize(y2, True)

            ridge2, rsq_2_training, rsq_2_test, _, _ = decoder(X2, y2, None, alphas=alphas, cv=cv, fit_intercept=True)
            df = df.append(
                {
                    'mouse': mouse,
                    'session': pair, 
                    'dv': dv,
                    'model': 'drug2',
                    'training': rsq_2_training,
                    'baseline': ridge2.score(X0, y0),
                    'drug1': ridge2.score(X1, y1),
                    'drug2': rsq_2_test
                },
                ignore_index=True)

    return df, None


def plot(name, df):
    fig, axs = plt.subplots(2, 4, figsize=(5 * 4, 5 * 2))
    dv = df['dv']

    for k, behavior in enumerate(['speed', 'whisk']):
        rsq1 = df['R2_1'][dv == behavior]
        rsq2 = df['R2_2'][dv == behavior]

        ax = axs[k, 0]
        ax.axis('equal')
        x = df['R2_1_2'][dv == behavior]
        y = df['R2_2_1'][dv == behavior]
        ax.scatter(x, y)
        ax.axline((0, 0), slope=1., ls='--', color='k')
        ax.set_xlabel('Day 1 / Pre Drug')
        ax.set_ylabel('Day 2 / Post Drug')
        ax.set_title(f'{behavior} {name} raw')

        ax = axs[k, 1]
        ax.axis('equal')
        x = df['R2_1_2'][dv == behavior]
        y = df['R2_2_1'][dv == behavior]
        p = (x >=0) & (y>=0)
        x = x[p]
        y = y[p]
        ax.scatter(x, y)
        ax.axline((0, 0), slope=1., ls='--', color='k')
        ax.set_xlabel('Day 1 / Pre Drug')
        ax.set_ylabel('Day 2 / Post Drug')
        ax.set_title(f'{behavior} {name} raw (positive)')

        ax = axs[k, 2]
        ax.axis('equal')
        x = df['R2_1_2'][dv == behavior] / np.maximum(rsq1, rsq2)
        y = df['R2_2_1'][dv == behavior] / np.maximum(rsq1, rsq2)
        ax.scatter(x, y)
        ax.axline((0, 0), slope=1., ls='--', color='k')
        ax.set_xlabel('Day 1 / Pre Drug')
        ax.set_ylabel('Day 2 / Post Drug')
        ax.set_title(f'{behavior} {name} ratio')

        ax = axs[k, 3]
        ax.axis('equal')
        x = df['R2_1_2'][dv == behavior] / np.maximum(rsq1, rsq2)
        y = df['R2_2_1'][dv == behavior] / np.maximum(rsq1, rsq2)
        p = (x >=0) & (y>=0)
        x = x[p]
        y = y[p]
        ax.scatter(x, y)
        ax.axline((0, 0), slope=1., ls='--', color='k')
        ax.set_xlabel('Day 1 / Pre Drug')
        ax.set_ylabel('Day 2 / Post Drug')
        ax.set_title(f'{behavior} {name} ratio (positive)')

    plt.savefig(f'decode_rsq_{name}.pdf')
    plt.close()


if __name__ == '__main__':
    dataset = load_yml('data/Chronic/chronic.yml')
    dfs = []
    r2s = []
    for d in dataset:
        df, r2 = decode_crossday(d)
        dfs.append(df)
        r2s.append(r2)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv('chronic_decode.csv', index=False)

    merged_df = pd.concat(r2s, ignore_index=True)
    merged_df.to_csv('chronic_decode_r2.csv', index=False)

    dataset = load_yml('data/Botox/botox.yml')
    dfs = []
    r2s = []
    for d in dataset:
        df, r2 = decode_crossday(d)
        dfs.append(df)
        r2s.append(r2)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv('botox_decode.csv', index=False)

    merged_df = pd.concat(r2s, ignore_index=True)
    merged_df.to_csv('botox_decode_r2.csv', index=False)

    dataset = load_yml('data/Neuromod/neuromod.yml')
    dfs = []
    # r2s = []
    for d in dataset:
        df, r2 = decode_drug(d)
        dfs.append(df)
        # r2s.append(r2)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv('neuromod_decode_bnorm.csv', index=False)

    merged_df = pd.concat(r2s, ignore_index=True)
    merged_df.to_csv('neuromod_decode_r2.csv')

    # Plot    
    df = pd.read_csv('chronic_decode_r2.csv')
    plot('chronic', df)
    df = pd.read_csv('botox_decode_r2.csv')
    plot('botox', df)
    df = pd.read_csv('neuromod_decode_r2.csv')
    plot('neuromod', df)

    df = pd.concat([
        pd.read_csv('chronic_decode_r2.csv'),
        pd.read_csv('botox_decode_r2.csv'),
        pd.read_csv('neuromod_decode_r2.csv')])
    plot('all', df)