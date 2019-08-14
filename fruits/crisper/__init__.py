import logging
import os
import shutil

import mandalka

def true_path(path):
    return os.path.abspath(os.path.realpath(path))

def safe_path_join(*args, dirname):
    assert dirname == true_path(dirname), "change 'dirname' to absolute path"
    path = true_path(os.path.join(*[dirname] + list(args)))
    assert path.startswith(dirname), "dirname: {}\npath: {}\nYou are a bad person!".format(dirname, path)
    return path

DATA_PATH = true_path(os.environ["CRISPER_DATA_PATH"])
RESULTS_PATH = safe_path_join("results", dirname=DATA_PATH)
CACHE_PATH = safe_path_join("cache", dirname=DATA_PATH)
TRASH_PATH = safe_path_join("trash", dirname=DATA_PATH)
FINGERPRINTS_PATH = safe_path_join("fingerprints", dirname=DATA_PATH)
HISTORY_PATH = safe_path_join("history", dirname=DATA_PATH)

def my_callback(node):
    fname = safe_path_join(mandalka.unique_id(node), dirname=HISTORY_PATH)
    try:
        with open(fname, 'x') as f_out:
            descr = mandalka.describe(node, depth=1)
            uids = ' '.join([mandalka.unique_id(n) for n in mandalka.inputs(node)])
            try:
                uid = mandalka.get_evaluation_stack(-1)
            except IndexError:
                uid = ""
            f_out.write(descr + '\n' + uids + '\n' + uid + '\n')
    except FileExistsError:
        pass
mandalka.config(evaluate_callback=my_callback)

def moderately_safe_rmtree(dirname):
    assert shutil.rmtree.avoids_symlink_attacks, "use Linux"
    assert dirname == true_path(dirname), "change 'dirname' to absolute path"
    assert dirname.startswith(DATA_PATH), "dirname should be inside CRISPER_DATA_PATH"
    print("rm -r {}".format(dirname))
    shutil.rmtree(dirname)

try:
    MODE = os.environ["CRISPER_MODE"]
    print("crisper running in mode: {}".format(MODE))
except KeyError:
    MODE = None


LOGGER = logging.getLogger("crisper")
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(ch)
del ch

def get_logger(node):
    logger = logging.getLogger(
        '.'.join(['crisper', node.__class__.__name__, mandalka.unique_id(node)]))
    return logger

for path in (DATA_PATH, RESULTS_PATH, CACHE_PATH, TRASH_PATH, FINGERPRINTS_PATH, HISTORY_PATH):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
del path


from ._evaluate import evaluate
