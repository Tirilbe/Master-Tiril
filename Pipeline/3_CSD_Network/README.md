# CSD Network Step

Folder: `Pipeline/3_CSD_Network`

This folder contains the scripts that convert wTO edge lists into C, S, D, and combined CSD networks.

## Files

1. `FindCSD.py`
2. `CreateNetwork.py`

## `FindCSD.py`

Run from this folder:

```bash
python FindCSD.py
```

`FindCSD.py` uses hardcoded input files and output names. The checked-in version currently reads:

- `results/wTO/wTO_rna_normal_edgelist_1_1.txt`
- `results/wTO/wTO_rna_cancer_edgelist_1_1.txt`

and writes the following files to `results/networks/`:

- `AllValues_*.txt`
- `UseableCValues_*.txt`
- `UseableSValues_*.txt`
- `UseableDValues_*.txt`

If you want to run it for proteome or another beta/output suffix, update `file1`, `file2`, and the output filenames in `FindCSD.py` before running.

## `CreateNetwork.py`

Run from this folder:

```bash
python CreateNetwork.py
```

`CreateNetwork.py` also uses hardcoded input files and parameters. The main settings to check before running are:

- `selSize`
- `noSels`
- `cValueFile`
- `sValueFile`
- `dValueFile`
- `AllValues_*.txt` input inside the script

The checked-in version currently reads the proteome `Useable*Values_proteome_WTO_1.txt` and `AllValues_proteome_WTO_1.txt` files from `results/networks/`, then writes the final selected networks to `results/CSD/`.

If you want RNA networks or another beta/output suffix, update the hardcoded file names in `CreateNetwork.py` before running.
