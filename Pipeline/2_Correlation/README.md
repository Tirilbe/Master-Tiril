# Correlation Step

Folder: `Pipeline/2_Correlation`

This folder contains the correlation and wTO generation step used before CSD network construction.

## Files

1. `FindCorrOnly.cpp`
2. `wTO_calculation.r`

## `FindCorrOnly.cpp`

`FindCorrOnly.cpp` uses hardcoded input and output paths in the file itself:

- `expDataFile`
- `outFile`
- `sampleSize`
- `numberOfGenes`

Update these values for the dataset you want to process, then compile and run from this folder:

```bash
g++ -O3 -std=c++17 -o FindCorrOnly.exe FindCorrOnly.cpp
.\FindCorrOnly.exe
```

Run this once for each matrix you need, for example RNA normal, RNA cancer, proteome normal, and proteome cancer, and update the hardcoded paths before each run.

Each run writes one `RhoAndVar_*.txt` file to `results/Corr/`.

## `wTO_calculation.r`

Run this script after the relevant `RhoAndVar_*.txt` files exist:

```bash
Rscript wTO_calculation.r
```

The script uses `BETA = 1` by default. To run with another beta value:

```bash
Rscript wTO_calculation.r 6
```

The checked-in version currently reads these files automatically:

- `results/Corr/RhoAndVar_proteome_normal_FINAL_FINAL.txt`
- `results/Corr/RhoAndVar_proteome_cancer_FINAL_FINAL.txt`

and writes wTO edge lists to `results/wTO/`.

If you want to run it for RNA instead of proteome, update the hardcoded input and output filenames in `wTO_calculation.r` before running.
