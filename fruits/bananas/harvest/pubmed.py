import json
import urllib.request
import urllib.error

def query(uids):
    """
    Notes
    -----
    https://www.nlm.nih.gov/bsd/licensee/elements_descriptions.html
    https://nsaunders.wordpress.com/2014/09/24/
        pubmed-publication-date-what-is-it-exactly/
    """
    uids = [str(s) for s in uids]
    assert all([s.isnumeric() for s in uids]), "every uid must be numeric str"
    assert not isinstance(uids, str), "'uids' cannot be a single uid str"
    url = (
        "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        "esummary.fcgi?db=pubmed&id={}&retmode=json"
    ).format(','.join(uids))
    try:
        result = json.load(urllib.request.urlopen(url))['result']
        del result["uids"]
        ok, errors = {}, []
        for k, v in result.items():
            if "error" in v:
                errors.append(k)
            else:
                ok[k] = v
        return ok, errors
    except (urllib.error.URLError, KeyError):
        return None, None

import time
import numpy as np
from tqdm import tqdm

def query2(uids, batch_size=400, n_tries=5, timeout=5):
    """
        query: verbose, batched, unique uids, retry on failure
    """
    uids = np.unique(uids)
    all_ok, all_errors = {}, []
    for i in tqdm.trange(0, len(uids), batch_size):
        _uids = uids[i:i+batch_size]
        for _ in range(n_tries):
            ok, errors = query(_uids)
            if ok is not None:
                break
            print("query failed, waiting {} seconds until next try...".format(timeout))
            time.sleep(timeout)
        if ok is None:
            raise RuntimeError("query failed {} times, aborting...".format(n_tries))
        all_ok.update(ok)
        all_errors += errors
    return all_ok, all_errors
