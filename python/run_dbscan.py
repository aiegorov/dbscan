"""Tests of DBSCAN C++ modules from Python."""
import os
from pathlib import Path

import numpy as np

from cpp import py_dbscan

def read_pcs_from_numpy(path: str):
    for numpy_f in os.listdir(path):
        print(f"Reading from file {numpy_f}")
        batch = np.load(str(Path(path) / numpy_f))
        for pc in batch:
            yield pc


if __name__ == "__main__":

    path = "/home/andrii/data/recordings/2023-03-13/particles_np/"
    eps = 0.5
    dbscan = py_dbscan.DBSCAN(eps, 10)
    max_n_ponts = 30000

    for _ in range(1):
        pc = next(read_pcs_from_numpy(path))
        print("Hist:")

        res = dbscan.fit_predict(pc[:, :2])
        print("\n")
