#!/usr/bin/env python3

import os
import sys
import tempfile
import zipfile

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from elderberries.benchmarks2018.problem import (
    Benchmarks2018Problem,
)
from elderberries.benchmarks2018.solutions import (
    fingerprinter_by_name
)

def arr_to_bytes(arr):
    f = tempfile.SpooledTemporaryFile()
    np.save(f, arr, allow_pickle=False)
    f.seek(0)
    b = f.read()
    f.close()
    return b

def obj_arr_to_txt(arr):
    return '\n'.join(list(arr))+'\n'

def csr_to_bytes(arr):
    f = tempfile.SpooledTemporaryFile()
    save_npz(f, arr)
    f.seek(0)
    b = f.read()
    f.close()
    return b

def save(arch, arr, name):
    if isinstance(arr, np.ndarray) and arr.dtype != np.object:
        arch.writestr(name + ".npy", arr_to_bytes(arr))
    elif isinstance(arr, np.ndarray) and arr.dtype == np.object:
        assert isinstance(arr[0], str)
        arch.writestr(name + ".txt", obj_arr_to_txt(arr))
    elif isinstance(arr, csr_matrix):
        arch.writestr(name + ".npz", csr_to_bytes(arr))
    else:
        raise TypeError()

def to_pki(arr):
    return 9. - arr

def export(target_uid, dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    problem = Benchmarks2018Problem(
        threshold=None,
        ic50_conversion_strategy="all_relations_half_ic50",
        fit_ic50=False
    )
    arch = zipfile.ZipFile(os.path.join(dirname, f"{target_uid}.zip"), mode='x', compression=zipfile.ZIP_DEFLATED)
    dataset = problem.get_dataset(target_uid)
    save(
        arch=arch,
        arr=problem.get_split("cv", dataset).data["groups"],
        name=f"{target_uid}/cv_groups",
    )
    save(
        arch=arch,
        arr=problem.get_split("bac", dataset).data["groups"],
        name=f"{target_uid}/bac_groups",
    )
    uid = dataset.data["uid"]
    save(arch=arch, arr=uid, name=f"{target_uid}/chembl_id")
    save(arch=arch, arr=dataset.data["smiles"], name=f"{target_uid}/smiles")
    save(arch=arch, arr=to_pki(dataset.data["value"]), name=f"{target_uid}/value")
    for fingerprint in fingerprinter_by_name:
        fp = fingerprinter_by_name[fingerprint](source=dataset)
        assert (fp.data["uid"] == uid).all()
        save(
            arch=arch,
            arr=fp.data[("fingerprint", "data")],
            name=f"{target_uid}/fingerprints/"+fingerprint,
        )
        save(
            arch=arch,
            arr=fp.data[("fingerprint", "keys")],
            name=f"{target_uid}/fingerprints/"+fingerprint+"_keys",
        )
    arch.close()

if __name__ == "__main__":
    targets = sys.argv[1:]
    for i, target in enumerate(targets):
        print(f"Exporting target: {target} ({i+1}/{len(targets)})...")
        export(target, ".")
