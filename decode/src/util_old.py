import math
import warnings

import mat73
import numpy as np
import yaml
from scipy.io import loadmat


def impute(x):
    mask = np.isnan(x)
    x_copy = np.copy(x)
    x_copy[mask] = np.nanmean(x)
    return x_copy


def load_yml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def normalize(x, zscore=False, eps=1e-8, return_func=False):
    if zscore:
        m = x.mean(0, keepdims=True)
        s = x.std(0, keepdims=True)
        if np.count_nonzero(s) == 0:
            warnings.warn('zero std')
        def func(x):
            return (x - m) / (s + eps)
        
        if return_func:
            return func
        else:
            return func(x)
    else:
        s = np.max(x, axis=0, keepdims=True)
        return x / s

    
def change_binsize(x, window, axis, reduce=np.mean):
    if window == 0:
        return x
    L = x.shape[axis]
    s = np.arange(window, L, step=window)
    xs = np.split(x, s, axis)
    assert xs[0].shape[axis] == window, xs[0].shape[axis]
    return np.concatenate([reduce(w, axis=axis, keepdims=True) for w in xs],
                          axis=axis)


def get_mapping(path):
    cell_map = loadmat(path, squeeze_me=True)
    mapping = cell_map['roiMatchData']['allSessionMapping'][()] - 1  # translate matlab index
    return mapping


def get_session(sess, window=15, check_nan=True):
    # 15 bins = 15 * 33.3ms = 500ms
    path = sess['path']
    exclude = sess['exclude']
    
    try:
        data = loadmat(path, squeeze_me=True)
        trials = data['fTrial'].tolist()
        iscell = data['iscell'][()][:, 0] == 1
        caTrialsMask = data['caTrialsMask'].tolist()  # spontaneous
        caTrialsMaskLED = data['caTrialsMaskLED'].tolist()  # For Neuromod package, 0 = Drug NO, 1 = Drug 1 YES
        caTrialsMaskLEDCL = data['caTrialsMaskLEDCL'].tolist()  # For Neuromod package, 0 = Drug NO, 1 = Drug 2 YES
        trial_names = data['fTrial'].dtype.names
    except:  
        # matfile v7.3
        data = mat73.loadmat(path)
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
        if tname in exclude:
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

        spontaneous.append(int(mask))
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


def get_session_newcase(sess, window=15, check_nan=True):
    # 15 bins = 15 * 33.3ms = 500ms
    path = sess['path']
    exclude = sess['exclude']

    print(path[:-9])
    
    try:
        data = loadmat(path, squeeze_me=True)
        trials = data['fTrial'].tolist()
        iscell = data['iscell'][()][:, 0] == 1
        caTrialsMask = data['caTrialsMask'].tolist()  # spontaneous
        caTrialsMaskLED = data['caTrialsMaskLED'].tolist()  # For Neuromod package, 0 = Drug NO, 1 = Drug 1 YES
        caTrialsMaskLEDCL = data['caTrialsMaskLEDCL'].tolist()  # For Neuromod package, 0 = Drug NO, 1 = Drug 2 YES
        trial_names = data['fTrial'].dtype.names
    except:  
        # matfile v7.3
        data = mat73.loadmat(path)
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
        if tname in exclude:
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

        spontaneous.append(int(mask))
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

