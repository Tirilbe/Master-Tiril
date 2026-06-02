# Cross-Omics network analyses of PDAC

This repository includes all scripts and notebooks used in my master thesis. This pipeline is divided into 7 steps and should be run in order from step 1 to step 7. `FindCSD.py` and `CreateNetwork.py` are scripts develope by Voigt et al. (https://github.com/andre-voigt/CSD), whereas `FindCorrOnly.cpp` is a modified script based on `FindCorrAndVar.cpp`. 

## Data

All data were obtained from the CPTAC PDAC cohort through the LinkedOmics database:

http://www.linkedomics.org/data_download/CPTAC-PDAC/

## Requirements

- Python 3.10.19
- R 4.5.2

Recommended setup:

1. Activate the environment from `environment.yml`.
2. Start in the `Pipeline/` folder.
3. Run the steps in numerical order.
4. Check that the output from one step exists before moving to the next step.

## Short summary of the order

1. Preprocess data
2. Calculate correlations
3. Create CSD networks
4. Run network analysis
5. Run functional analysis (optional)
6. Compare RNA and protein
7. Run survival analysis

## Step 1: Data Preprocessing

Folder: `Pipeline/1_Data_Preprocessing`

Run these notebooks in this order:

1. `data_process_transcriptome.ipynb`
2. `data_process_proteome.ipynb`
3. `multiomics_harmonization.ipynb`

This step creates processed RNA and proteomics data in `data/processed/` for the rest of the pipeline.

`cell_composition_analysis.py` — visualizes tissue composition before and after tumor purity filtering and can be run independently after step 1.

## Step 2: Correlation

Folder: `Pipeline/2_Correlation`

Run the files in this folder to calculate correlations:

1. `FindCorrOnly.cpp` 
1. `wTO_calculation.r`

This step creates correlation results that are used in the later network steps.

## Step 3: CSD Network

Folder: `Pipeline/3_CSD_Network`

Run these files in this order:

1. `FindCSD.py`
2. `CreateNetwork.py`

This step creates the C, S, D, and CSD networks used in the network analysis.

## Step 4: Network Analysis

Folder: `Pipeline/4_Network_Analysis`

Run these notebooks in this order:

1. `network_and_gcc.ipynb`
2. `topology_and_hubs.ipynb`
3. `modules.ipynb`

This step creates GCC files, topology and hub results, modules, and eigengenes.

## Step 5: Functional Analysis

Folder: `Pipeline/5_Functional_Analysis`

Run:

1. `functional_enrichment.ipynb`

This step is optional, but useful if you want GO enrichment for the modules.

## Step 6: Cross-Modal Comparison

Folder: `Pipeline/6_Cross_Modal_Comparison`

Run:

1. `cross_modal_overlap.ipynb`

This step compares RNA and protein results across GCCs, modules, and genes.

## Step 7: Survival Analysis

Folder: `Pipeline/7_Survival_Analysis`

Run:

1. `survival_analysis.ipynb`

This step connects the most important modules and genes to survival data.


## Tips

- Run each notebook with `Run All` from the top.
- Do not skip steps, because later steps depend on files from earlier steps.
- Most outputs are written to `results/` and `data/processed/`.
- Further detalis on input and output is described in each individual notebook.