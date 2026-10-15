import io
import subprocess

import numpy as np
import pandas as pd


def committed(path):
    return pd.read_csv(io.BytesIO(subprocess.check_output(["git", "show", f"HEAD:{path}"])), low_memory=False)


def noise_only(path):
    old, new = committed(path), pd.read_csv(path, low_memory=False)
    if list(old.columns) != list(new.columns) or len(old) != len(new):
        return False
    num = old.select_dtypes("number").columns
    if not old.drop(columns=num).equals(new.drop(columns=num)):
        return False
    return np.allclose(old[num].to_numpy(float), new[num].to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True)


for path in subprocess.check_output(["git", "diff", "--name-only", "--", "data/*.csv"], text=True).split():
    if noise_only(path):
        subprocess.run(["git", "checkout", "--", path], check=True)
        print(f"{path}: floating-point noise only, kept the committed file")
