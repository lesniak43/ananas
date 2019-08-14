import numpy as np
import tqdm
from collections import Counter
import subprocess
import tempfile
import os
from scipy.sparse import csr_matrix


def is_smiles_correct(arr_smiles):
    mask = np.zeros(len(arr_smiles), dtype=np.bool)
    uids = np.arange(0, len(arr_smiles)).astype(np.str)

    temp_dir = tempfile.mkdtemp()
    smiles_filename = os.path.join(temp_dir, "smiles.smi")
    output_filename = os.path.join(temp_dir, "output.smi")
    _dump_smi(arr_smiles, smiles_filename, uids)
    subprocess.run(["obabel", smiles_filename, "-O", output_filename, "-e"])
    _, correct_uids = _load_smi(output_filename)

    mask[correct_uids.astype(np.int)] = True
    return mask

def molprint2d_count_fingerprinter(arr_smiles):
    temp_dir = tempfile.mkdtemp()
    smiles_filename = os.path.join(temp_dir, "smiles.smi")
    output_filename = os.path.join(temp_dir, "output.mpd")

    _dump_smi(arr_smiles, smiles_filename, arr_uids=None)
    subprocess.run(["obabel", smiles_filename, "-O", output_filename])
    fingerprint, columns, _ = _load_mpd(output_filename)

    os.remove(smiles_filename)
    os.remove(output_filename)

    return fingerprint, columns


def _dump_smi(arr_smiles, filename, arr_uids=None):
    assert len(arr_smiles.shape) == 1
    assert (arr_uids is None) or (arr_uids.shape == arr_smiles.shape)
    with open(filename, 'w') as f_out:
        if arr_uids is None:
            arr_uids = np.arange(0, len(arr_smiles)).astype(np.str)
        for uid, smiles in zip(arr_uids, arr_smiles):
            f_out.write(smiles + "\t" + uid + "\n")

def _load_smi(filename):
    uids = []
    l_smiles = []
    with open(filename, 'r') as f_in:
        for line in tqdm.tqdm(f_in):
            line = line.rstrip().split()
            l_smiles.append(line[0])
            uids.append(line[1])
    arr_uids = np.array(uids)
    arr_smiles = np.array(l_smiles)
    return arr_smiles, arr_uids

def _load_mpd(output_filename):
    """
        return csr_matrix (data), numpy.array (columns), numpy.array (rows)
    """

    data = []
    row_idx = []
    col_idx = []

    h_axis = []
    v_axis = []
    _curr_h_axis_idx = 0
    _h_axis_idx = {}

    with open(output_filename, 'r') as f_in:
        for i, line in tqdm.tqdm(enumerate(f_in)):
            line = line.rstrip()
            line = line.split('\t')
            v_axis.append(line[0])
            for key, value in Counter(line[1:]).items():
                row_idx.append(i)
                if not key in _h_axis_idx:
                    _h_axis_idx[key] = _curr_h_axis_idx
                    _curr_h_axis_idx += 1
                col_idx.append(_h_axis_idx[key])
                data.append(value)

    h_axis = np.array([tup[0] for tup in sorted(_h_axis_idx.items(), key=lambda x:x[1])], dtype=np.str)
    v_axis = np.array(v_axis, dtype=np.str)
    arr = csr_matrix((data, (row_idx, col_idx)), shape=(len(v_axis), len(h_axis)))

    return arr, h_axis, v_axis

def spectrophores(mols, accuracy=20, resolution=3.0):
    from copy import deepcopy
    import os
    import subprocess
    import tempfile
    import numpy as np
    from rdkit.Chem.rdmolfiles import SDWriter

    assert accuracy in (1, 2, 5, 10, 15, 20, 30, 36, 45, 60)
    assert isinstance(resolution, float)
    assert resolution > 0.

    _mols = []
    names = np.array([str(i) for i in range(len(mols))])
    for m, name in zip(mols, names):
        if m.GetNumConformers() == 1:
            _mols.append(deepcopy(m))
            _mols[-1].SetProp("_Name", name)
        elif m.GetNumConformers() == 0:
            _mols.append(None)
        else:
            raise ValueError("every molecule must have at most 1 conformer")
    mols = np.array(_mols, dtype=np.object)
    del _mols

    temp_dir = tempfile.mkdtemp()
    fname = os.path.join(temp_dir, "mols.sdf")
    writer = SDWriter(fname)
    mask_nan = np.array([m is None for m in mols], dtype=np.bool)
    [writer.write(m, confId=0) for m in mols[np.logical_not(mask_nan)]]

    result = subprocess.run(["obspectrophore", "-i", fname, "-a", str(accuracy), "-r", str(resolution)], stdout=subprocess.PIPE).stdout.decode("utf-8").split('\n')

    assert not any(['\t' in line for line in result[:11]])
    assert not '\t' in result[:-1]

    arr = np.zeros((len(mols), 48), dtype=np.float32)
    out_names = []
    for i, line in enumerate(result[11:-1]):
        row = line.split('\t')
        assert len(row) == 50
        out_names.append(row[0])
        assert row[-1] == ""
        arr[i,:] = np.array([float(s) for s in row[1:-1]])
    arr[mask_nan] = np.nan

    assert all([a==b for a, b in zip(names[np.logical_not(mask_nan)], out_names)])

    os.remove(fname)

    return arr, np.array(["{:02d}".format(i+1) for i in range(48)], dtype=np.str)
