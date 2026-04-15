"""
Utility functions for RNA-seq data processing.
Used by notebooks/Processing/01a_data_process_RNA_counts.ipynb
"""

import gzip
import os
import re
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def load_gtex(gtex_meta, gtex_gct):
    # 1. Load metadata to find pancreas samples
    meta = pd.read_csv(gtex_meta, sep="\t", low_memory=False)
    pancreas_ids = meta.loc[meta["SMTS"] == "Pancreas", "SAMPID"].tolist()
    if not pancreas_ids:
        raise ValueError("Ingen pancreas-samples funnet i GTEx metadata.")
    print(f"Found {len(pancreas_ids)} pancreas samples in GTEx metadata.")

    # 2. Read GTEx expression data (skipping first two rows) (only header)
    header = pd.read_csv(gtex_gct, sep="\t", skiprows=2, nrows=0)
    header.columns = header.columns.str.strip()
    all_cols = header.columns.tolist()

    # 3. Identify the gene_id column
    gene_col = "Name" if "Name" in all_cols else all_cols[0]

    # 4. Select only gene_id + pancreas samples
    keep_cols = [gene_col] + [c for c in all_cols if c in pancreas_ids]
    print(f"Loading only {len(keep_cols) - 1} pancreas columns")

    # 5. Load only selected columns
    expr = pd.read_csv(gtex_gct, sep="\t", skiprows=2, usecols=keep_cols)
    print(f"Loaded GTEx subset with shape: {expr.shape}")

    expr = expr.rename(columns={gene_col: "gene_id"})
    return expr


def load_tcga_tpm(filepath, value_col="tpm_unstranded"):
    """
    Load a TCGA STAR gene counts file into a pandas DataFrame.
    Default: tpm_unstranded.
    """
    df = pd.read_csv(filepath, sep="\t", comment="#", header=0)

    # Drop QC rows (N_unmapped, N_multimapping, etc.)
    first_col = df.columns[0]
    df = df[~df[first_col].astype(str).str.startswith(("N_", "__"))]

    df = df[[first_col, value_col]].rename(columns={first_col: "gene_id"})
    return df


def process_tcga_data(counts_dir, sample_sheet, value_col="tpm_unstranded"):
    all_samples = []
    input_dir = Path(counts_dir)

    # Load sample sheet and build mapping: File Name -> Sample ID
    sample_meta = pd.read_csv(sample_sheet, sep="\t")
    sample_meta.columns = sample_meta.columns.str.strip()
    file_to_sample = dict(
        zip(
            sample_meta["File Name"].astype(str).str.strip(),
            sample_meta["Sample ID"].astype(str).str.strip(),
        )
    )
    print(f"Loaded {len(file_to_sample)} sample mappings from {Path(sample_sheet).name}")

    # Loop through all .tsv files containing STAR counts
    for fpath in input_dir.rglob("*.tsv"):
        if "star_gene_counts" in fpath.name:
            sample_id = file_to_sample.get(fpath.name)
            if sample_id is None:
                print(f"WARNING: '{fpath.name}' not found in sample sheet. Using UUID fallback.")
                sample_id = fpath.name.split(".")[0]
            df = load_tcga_tpm(fpath, value_col=value_col)
            df = df.rename(columns={value_col: sample_id})
            all_samples.append(df)

    if not all_samples:
        raise ValueError(f"No STAR counts files found in {input_dir}.")
    print(f"Found {len(all_samples)} STAR counts files in {input_dir}.")

    # Merge all samples on gene_id
    tcga = reduce(
        lambda left, right: pd.merge(left, right, on="gene_id", how="outer"),
        all_samples,
    )
    print(f"Merged {len(all_samples)} samples into a single DataFrame with shape: {tcga.shape}")
    return tcga


# ---------------------------------------------------------------------------
# Reusable processing functions
# ---------------------------------------------------------------------------

def clean_df(df):
    if "gene_id" not in df.columns:
        raise ValueError("Input DataFrame must contain 'gene_id' column.")

    print(f"Original data shape: {df.shape}")

    # Remove version numbers from gene IDs
    df["gene_id"] = df["gene_id"].astype(str).str.split(".").str[0]

    # Remove duplicates
    before = df.shape[0]
    df = df.drop_duplicates(subset="gene_id", keep="first")
    print(f"Removed {before - df.shape[0]} duplicate genes.")
    print(f"Data shape after removing duplicates: {df.shape}")

    return df.sort_values("gene_id").reset_index(drop=True)


def load_protein_coding_genes(gtf_path, out_file=None):
    gene_ids = set()
    records = []

    gene_id_pattern = re.compile(r'gene_id "([^"]+)"')
    gene_name_pattern = re.compile(r'gene_name "([^"]+)"')
    biotype_pattern = re.compile(r'gene_type "([^"]+)"|gene_biotype "([^"]+)"')

    file_handle = gzip.open(gtf_path, "rt") if str(gtf_path).endswith(".gz") else open(gtf_path, "r")

    with file_handle as file:
        for line in file:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[2] != "gene":
                continue
            info = fields[8]

            biotype_match = biotype_pattern.search(info)
            if not biotype_match or biotype_match.group(1) != "protein_coding":
                continue

            gene_id_match = gene_id_pattern.search(info)
            if not gene_id_match:
                continue
            gene_id = gene_id_match.group(1).split(".")[0]

            gene_name_match = gene_name_pattern.search(info)
            gene_name = gene_name_match.group(1) if gene_name_match else None

            gene_ids.add(gene_id)
            records.append((gene_id, gene_name))

    df = pd.DataFrame(records, columns=["gene_id", "gene_name"]).drop_duplicates()
    print(f"Extracted {len(gene_ids)} protein-coding genes from {Path(gtf_path)}")

    if out_file:
        df.to_csv(out_file, index=False)
        print(f"Saved gene ID to name mapping to {out_file}")

    return gene_ids, df


def filter_protein_coding(df, pc_genes, id_col="gene_id"):
    """Filter a DataFrame to only protein-coding genes."""
    if id_col not in df.columns:
        raise ValueError(f"Input DataFrame must contain '{id_col}' column.")
    df_filt = df[df[id_col].astype(str).isin(pc_genes)].copy()
    df_filt = df_filt.sort_values(by=id_col).reset_index(drop=True)
    print(f"Filtered to protein-coding genes: {df_filt.shape}")
    return df_filt


def filter_low_expression_counts(counts_df, min_cpm=1, min_fraction=0.2):
    """Filter lowly expressed genes using CPM threshold."""
    gene_ids = counts_df["gene_id"]
    expr = counts_df.drop(columns=["gene_id"])

    lib_size = expr.sum(axis=0)
    cpm = expr.div(lib_size, axis=1) * 1e6
    min_samples = int(np.ceil(min_fraction * cpm.shape[1]))
    keep_mask = (cpm > min_cpm).sum(axis=1) >= min_samples

    filtered_df = counts_df.loc[keep_mask].reset_index(drop=True)
    print(f"Original genes: {counts_df.shape[0]}")
    print(f"Genes retained: {filtered_df.shape}")
    print(f"Minimum samples required: {min_samples}")
    return filtered_df


def harmonize_datasets(gtex_df, tcga_df, proteome_genes, out_dir=None):
    id_col = "gene_id"

    if isinstance(proteome_genes, (str, Path)):
        proteome_genes = pd.read_csv(proteome_genes, header=None)[0].tolist()

    common_genes = set(gtex_df[id_col]) & set(tcga_df[id_col]) & set(proteome_genes)
    print(f"Number of common protein-coding genes: {len(common_genes)}")

    gtex_filt = gtex_df[gtex_df[id_col].isin(common_genes)]
    tcga_filt = tcga_df[tcga_df[id_col].isin(common_genes)]

    if out_dir is not None:
        out_path = Path(out_dir) / "genelist/common_genes.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for gene in common_genes:
                f.write(str(gene) + "\n")

    gtex_filt = gtex_filt.sort_values(by=id_col).reset_index(drop=True)
    tcga_filt = tcga_filt.sort_values(by=id_col).reset_index(drop=True)
    print(f"Shapes after filtering to common genes — GTEx: {gtex_filt.shape}, TCGA: {tcga_filt.shape}")
    return gtex_filt, tcga_filt


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def prepare_findcorrvar_input(
    df_path: str | os.PathLike,
    out: str | os.PathLike = "ExpData.txt",
    float_format: str = "{:.6f}",
):
    """Convert a gene expression CSV to findcorrvar-compatible format."""
    df = pd.read_csv(df_path)

    if "gene_id" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "gene_id"})

    genes = df["gene_id"].astype(str).values
    values = df.drop(columns=["gene_id"]).apply(pd.to_numeric, errors="coerce")

    with open(out, "w") as f:
        for gene, row in zip(genes, values.to_numpy()):
            formatted_values = " ".join(float_format.format(v) for v in row)
            f.write(f"{gene} {formatted_values}\n")
