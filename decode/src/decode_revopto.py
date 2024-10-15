from concurrent.futures import ProcessPoolExecutor
import itertools
from pathlib import Path

from absl import app, flags
from pymatreader import read_mat
import numpy as np
import polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

from util import change_binsize, normalize


alphas = np.logspace(-1, 18, 10, endpoint=True)  # l2 penalty grid
cv = None  # None for GCV

flags.DEFINE_string("data_dir", "data/Rev_CtrlM1", None)
flags.DEFINE_string("output_dir", "outputs/Rev_CtrlM1", None)
FLAGS = flags.FLAGS


def run_decoder(X, y, **kwargs):
    """Fit decoder
    param X: cell subset
    param y: behavior
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    ridge_X = RidgeCV(**kwargs).fit(X_train, y_train)
    rsq_X_training = ridge_X.score(X_train, y_train)
    rsq_X_test = ridge_X.score(X_test, y_test)

    return ridge_X, rsq_X_training, rsq_X_test


def get_files(session_dir):
    """
    Given session folder, return paths to variable and group files
    """
    session_name = str(session_dir.stem)

    var_filename = session_name.replace("_Analysed", "_vars.mat")
    group_filename = session_name.replace("_Analysed", "_groups.mat")
    rebound_filename = session_name.replace("_Analysed", "_trig0_reb.mat")

    return (
        session_dir / var_filename,
        session_dir / group_filename,
        session_dir.parent / "Reb" / rebound_filename,
    )


def get_session_data(session_dir, window=15, check_nan=True):
    var_path, group_path, rebound_path = get_files(session_dir)

    data = read_mat(var_path)
    group = read_mat(group_path)
    rebound = read_mat(rebound_path)

    spon_tnames = group["groups"]["spontAll"]
    led_tnames = group["groups"]["spontLed"]

    trial_names = data["fTrial"].keys()
    trials = data["fTrial"].values()
    iscell = data["iscell"][()][:, 0] == 1

    rebound = rebound["reb"]
    rebound = (rebound[:, -1] == 1) | (rebound[:, -2] == 1)
    rebound[np.isnan(rebound)] = False
    rebound = rebound[iscell]

    # spont
    sSpks = []
    sSpeed = []
    sWhisk = []

    # led
    lSpks = []
    lSpeed = []
    lWhisk = []

    for trial, name in zip(trials, trial_names):
        if (name not in spon_tnames) and (name not in led_tnames):
            continue

        fSpks = trial["fSpks"][()]
        fSpeed = trial["fSpeed"][()]
        fWhisk1 = trial["fWhisk1"][()]
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

        fSpks = fSpks[:, iscell]

        if name in spon_tnames:
            sSpks.append(fSpks)
            sSpeed.append(fSpeed)
            sWhisk.append(fWhisk1)
        elif name in led_tnames:
            lSpks.append(fSpks)
            lSpeed.append(fSpeed)
            lWhisk.append(fWhisk1)

    sSpks = np.vstack(sSpks)
    sSpeed = np.concatenate(sSpeed)
    sWhisk = np.concatenate(sWhisk)

    lSpks = np.vstack(lSpks)
    lSpeed = np.concatenate(lSpeed)
    lWhisk = np.concatenate(lWhisk)

    return sSpks, sSpeed, sWhisk, lSpks, lSpeed, lWhisk, rebound


def run_session(session_dir):
    table = None
    try:
        sSpks, sSpeed, sWhisk, lSpks, lSpeed, lWhisk, rebound = get_session_data(
            session_dir
        )

        table = []
        for dv, nnorm, bnorm, with_rebound in itertools.product(
            ["speed", "whisk"], [False, True], [False, True], [False, True]
        ):
            X1 = sSpks
            X2 = lSpks

            if not with_rebound:
                X1 = X1[:, np.logical_not(rebound)]
                X2 = X2[:, np.logical_not(rebound)]

            if nnorm:
                X1 = normalize(X1, True)
                X2 = normalize(X2, True)

            if dv == "speed":
                y1 = sSpeed
                y2 = lSpeed
            else:
                y1 = sWhisk
                y2 = lWhisk

            # zscore behavioral variables if across
            if bnorm:
                y1 = normalize(y1, True)
                y2 = normalize(y2, True)

            #
            ridge1, rsq_training_1, rsq_test_1 = run_decoder(
                X1, y1, alphas=alphas, cv=cv, fit_intercept=not bnorm
            )
            ridge2, rsq_training_2, rsq_test_2 = run_decoder(
                X2, y2, alphas=alphas, cv=cv, fit_intercept=not bnorm
            )

            table.extend(
                [
                    [
                        str(session_dir.stem),
                        dv,
                        "spontAll",
                        "training",
                        rsq_training_1,
                        nnorm,
                        bnorm,
                        with_rebound,
                    ],
                    [
                        str(session_dir.stem),
                        dv,
                        "spontAll",
                        "test",
                        rsq_test_1,
                        nnorm,
                        bnorm,
                        with_rebound,
                    ],
                    [
                        str(session_dir.stem),
                        dv,
                        "spontAll",
                        "spontLed",
                        ridge1.score(X2, y2),
                        nnorm,
                        bnorm,
                        with_rebound,
                    ],
                    [
                        str(session_dir.stem),
                        dv,
                        "spontLed",
                        "training",
                        rsq_training_2,
                        nnorm,
                        bnorm,
                        with_rebound,
                    ],
                    [
                        str(session_dir.stem),
                        dv,
                        "spontLed",
                        "test",
                        rsq_test_2,
                        nnorm,
                        bnorm,
                        with_rebound,
                    ],
                    [
                        str(session_dir.stem),
                        dv,
                        "spontLed",
                        "spontAll",
                        ridge2.score(X1, y1),
                        nnorm,
                        bnorm,
                        with_rebound,
                    ],
                ]
            )
    except Exception as e:
        print(e)

    table = pl.DataFrame(
        table,
        schema={
            "session": pl.Utf8,
            "behavior": pl.Utf8,
            "model": pl.Utf8,
            "target": pl.Utf8,
            "r2": pl.Float32,
            "normed_neural": pl.Boolean,
            "normed_behavior": pl.Boolean,
            "with_rebound": pl.Boolean,
        },
    )
    table.write_csv(session_dir / "decoding.csv")
    return table


def main(_) -> None:
    np.random.seed(0)
    data_dir = Path.cwd() / FLAGS.data_dir
    output_dir = Path.cwd() / FLAGS.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    session_dirs = [path for path in data_dir.iterdir() if path.is_dir()]

    with ProcessPoolExecutor(8) as pool:
        tables = list(pool.map(run_session, session_dirs))

    combined_table: pl.DataFrame = pl.concat(tables)
    combined_table.write_csv(output_dir / "decoding.csv")


if __name__ == "__main__":
    app.run(main)
