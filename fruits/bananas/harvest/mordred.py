from mordred import Calculator, is_missing
from mordred import descriptors as _descriptors 
import numpy as np
import tqdm

descriptors = Calculator(
    _descriptors,
    ignore_3D=False,
    version="1.1.0"
).descriptors
descriptors2d = [d for d in descriptors if not d.require_3D]
descriptors3d = [d for d in descriptors if d.require_3D]

def mordred_fingerprint2d(mols):
    result = np.zeros((len(mols), len(descriptors2d)), dtype=np.float32)
    calc = Calculator(descriptors2d)
    for i, m in enumerate(tqdm.tqdm(mols)):
        for j, v in enumerate(calc(m)):
            result[i,j] = v if not is_missing(v) else np.nan
    header = np.array([str(d) for d in descriptors2d])
    return result, header

def mordred_fingerprint3d(mols):
    result = np.zeros((len(mols), len(descriptors3d)), dtype=np.float32)
    calc = Calculator(descriptors3d)
    for i, m in enumerate(tqdm.tqdm(mols)):
        if m.GetNumConformers() == 1:
            for j, v in enumerate(calc(m)):
                result[i,j] = v if not is_missing(v) else np.nan
        elif m.GetNumConformers() == 0:
            result[i] = np.nan
        else:
            raise ValueError("every molecule must have at most 1 conformer")
    header = np.array([str(d) for d in descriptors3d])
    return result, header
