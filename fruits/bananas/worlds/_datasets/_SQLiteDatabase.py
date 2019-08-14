import os
import sqlite3

import numpy as np
import tqdm

import mandalka

@mandalka.node
class SQLiteDatabase:
    def __init__(self, *, filename):
        self.path = os.path.join(
            os.getenv("BANANAS_EXTERNAL_DATA_PATH"),
            filename,
        )
        assert os.path.isfile(self.path), "Database doesn't exist"

    def query(self, query, *args):
        conn = sqlite3.connect(self.path)
        for r in conn.cursor().execute(query, args):
            yield r
        conn.close()

    def query2(self, query, *args, dtypes, defaults):
        assert len(dtypes) == len(defaults)
        results = [list() for _ in range(len(dtypes))]
        for r in tqdm.tqdm(self.query(query, *args)):
            for i, x in enumerate(r):
                results[i].append(defaults[i] if x is None else x)
        results = [np.array(l, dtype=dtypes[i]) for i, l in enumerate(results)]
        idx = np.lexsort(list(reversed(results)))
        results = [arr[idx] for arr in results]
        return results
