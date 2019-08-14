# ananas

Machine Learning meets Biochemistry.

Tested on Linux. Expect bugs.

## Installation

Required python packages:
`numpy numba scipy tqdm scikit-learn rdkit openbabel mordred matplotlib`

Create three separate directories for storing ananas database, results of the experiments, and external data.
Set the following environment variables as full paths to the newly created directories:
`CRISPER_DATA_PATH` `ANANAS_RESULTS_PATH` `BANANAS_EXTERNAL_DATA_PATH`

Download ChEMBL 24.1 database (sqlite), extract the archive, and put the file `chembl_24.db` in the `BANANAS_EXTERNAL_DATA_PATH` directory.

## Examples

### Export datasets

`cd fruits/elderberries/benchmarks2018`
`./export_zip.py TARGET_ID`

Running this script will create a ZIP archive `TARGET_ID.zip` in the current directory.

Replace `TARGET_ID` with ChEMBL target ID, e.g. `CHEMBL214` for serotonin 1a (5-HT1a) receptor.

### Example experiment

`cd fruits/elderberries/benchmarks2018`
`./run.py`

You can run multiple instances of the script in parallel, e.g.:

`for i in {1..43}; do ./run.py & done`

Monitor the progress:

`env CRISPER_MODE=MONITOR ./run.py`

Results will be stored in the `ANANAS_RESULTS_PATH` directory.

Modify the set of targets:

Edit line 108 in `fruits/elderberries/benchmarks2018/problem.py`.

Modify the set of trained models:

Edit dictionaries `SOLUTIONS_C` and `SOLUTIONS_R` (classification and regression respectively) in `fruits/elderberries/benchmarks2018/solutions.py`.

Modify the set of run benchmarks:

Edit the list `SUMMARIES` in `fruits/elderberries/benchmarks2018/run.py`

Sometimes it might be necessary to manually unlock the database. Stop all running scripts and:
* remove `CRISPER_DATA_PATH/cache` directory
* remove `CRISPER_DATA_PATH/fingerprints/cache` directory
* remove `CRISPER_DATA_PATH/fingerprints/pending` directory
