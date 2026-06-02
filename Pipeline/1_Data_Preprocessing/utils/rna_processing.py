"""
Utility functions for RNA-seq data processing.
Used by notebooks/Processing/data_process_transcriptome.ipynb
"""

from pathlib import Path

import pandas as pd


def _prepare_symbol_index(df):
    prepared = df.copy()
    prepared.index = prepared.index.astype(str).str.strip()
    prepared = prepared.loc[prepared.index.notna()]
    prepared = prepared.loc[prepared.index != ""]
    prepared = prepared[~prepared.index.duplicated(keep="first")]
    return prepared


# ---------------------------------------------------------------------------
# Reusable processing functions
# ---------------------------------------------------------------------------

def load_rna_expression_matrices(tumor_path, normal_path, sep="\t"):
    """Load CPTAC tumor and normal RNA matrices and print a short summary."""
    tumor_df = pd.read_csv(tumor_path, sep=sep, index_col=0)
    normal_df = pd.read_csv(normal_path, sep=sep, index_col=0)

    print("Loading data...")
    print(f"  RNA cancer:       {tumor_df.shape[0]:>6} genes, {tumor_df.shape[1]:>4} samples")
    print(f"  RNA normal:       {normal_df.shape[0]:>6} genes, {normal_df.shape[1]:>4} samples")
    return tumor_df, normal_df


def save_processed_rna_matrices(
    tumor_df,
    normal_df,
    tumor_out,
    normal_out,
    tumor_csd_out=None,
    normal_csd_out=None,
):
    """Save final RNA matrices with headers and optional CSD exports."""
    tumor_out = Path(tumor_out)
    normal_out = Path(normal_out)
    tumor_out.parent.mkdir(parents=True, exist_ok=True)
    normal_out.parent.mkdir(parents=True, exist_ok=True)

    tumor_df.to_csv(tumor_out, sep='\t', index=True, header=True)
    normal_df.to_csv(normal_out, sep='\t', index=True, header=True)
    print(f"Saved: {tumor_out}")
    print(f"Saved: {normal_out}")

    if tumor_csd_out is not None:
        export_csd_input(tumor_df, tumor_csd_out)
    if normal_csd_out is not None:
        export_csd_input(normal_df, normal_csd_out)


def filter_samples_by_kras_vaf(
    expression_df,
    clinical_df,
    maf_path,
    vaf_threshold=0.075,
    extra_removed_samples=None,
):
    """Filter tumor samples using KRAS VAF-derived purity criteria.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Gene-by-sample expression matrix with tumor sample IDs in columns.
    clinical_df : pd.DataFrame
        CPTAC clinical table containing a ``case_id`` column.
    maf_path : str or Path
        Path to the MAF file used to derive KRAS VAF.
    vaf_threshold : float, default=0.075
        Minimum per-sample KRAS VAF required to retain a tumor sample.
    extra_removed_samples : list[str], optional
        Additional manually curated low-purity samples to exclude.
        Samples without a KRAS row in the MAF are otherwise retained.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Filtered expression matrix and filtered clinical table.
    """
    maf = pd.read_csv(maf_path, sep="\t", low_memory=False)
    kras_maf = maf.loc[maf["Hugo_Symbol"].astype(str).str.upper() == "KRAS"].copy()

    if kras_maf.empty:
        raise ValueError("No KRAS mutations were found in the supplied MAF file.")

    kras_maf["Tumor_Sample_Barcode"] = kras_maf["Tumor_Sample_Barcode"].astype(str).str.strip()
    kras_maf["t_depth"] = pd.to_numeric(kras_maf["t_depth"], errors="coerce")
    kras_maf["t_alt_count"] = pd.to_numeric(kras_maf["t_alt_count"], errors="coerce")
    kras_maf = kras_maf.dropna(subset=["t_depth", "t_alt_count"])
    kras_maf = kras_maf.loc[kras_maf["t_depth"] > 0].copy()
    kras_maf["kras_vaf"] = kras_maf["t_alt_count"] / kras_maf["t_depth"]

    sample_vaf = kras_maf.groupby("Tumor_Sample_Barcode", sort=False)["kras_vaf"].min()
    low_purity_samples = set(sample_vaf.loc[sample_vaf < vaf_threshold].index.astype(str))

    extra_removed_samples = {str(sample).strip() for sample in (extra_removed_samples or [])}
    excluded_samples = low_purity_samples | extra_removed_samples

    kept_columns = [column for column in expression_df.columns if str(column).strip() not in excluded_samples]
    filtered_expression = expression_df.loc[:, kept_columns].copy()

    filtered_clinical = clinical_df.copy()
    if "case_id" in filtered_clinical.columns:
        filtered_clinical["case_id"] = filtered_clinical["case_id"].astype(str).str.strip()
        filtered_sample_ids = {str(column).strip() for column in filtered_expression.columns}
        filtered_clinical = filtered_clinical.loc[
            filtered_clinical["case_id"].isin(filtered_sample_ids)
        ].copy()

    removed_count = expression_df.shape[1] - filtered_expression.shape[1]
    print(
        f"Removed {removed_count} / {expression_df.shape[1]} samples using lowest KRAS VAF "
        f"< {vaf_threshold:.3f}"
    )
    if low_purity_samples:
        print(f"Removed {len(low_purity_samples)} samples with lowest KRAS VAF < {vaf_threshold:.3f}")
    if extra_removed_samples:
        print(f"Included manual KRAS VAF exclusions: {sorted(extra_removed_samples)}")

    return filtered_expression, filtered_clinical


def filter_genes_by_zero_fraction(tumor_df, normal_df, max_zero_fraction=0.3):
    """Remove genes with too many zero values in either condition."""
    tumor_df = _prepare_symbol_index(tumor_df)
    normal_df = _prepare_symbol_index(normal_df)

    common_genes = tumor_df.index.intersection(normal_df.index)
    tumor_common = tumor_df.loc[common_genes].copy()
    normal_common = normal_df.loc[common_genes].copy()

    tumor_zero_fraction = tumor_common.eq(0).mean(axis=1)
    normal_zero_fraction = normal_common.eq(0).mean(axis=1)
    keep_mask = (tumor_zero_fraction <= max_zero_fraction) & (normal_zero_fraction <= max_zero_fraction)

    filtered_tumor = tumor_common.loc[keep_mask].sort_index()
    filtered_normal = normal_common.loc[keep_mask].sort_index()

    print(f"Genes before zero filtering: {len(common_genes)}")
    print(f"Genes removed by zero filtering: {int((~keep_mask).sum())}")
    print(f"Genes retained after zero filtering: {filtered_tumor.shape[0]}")

    return filtered_tumor, filtered_normal, keep_mask




def export_csd_input(df, out, float_format="{:.6f}"):
    """Export a gene-by-sample matrix to the whitespace-delimited CSD format."""
    matrix = _prepare_symbol_index(df)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        for gene, row in matrix.iterrows():
            formatted_values = " ".join(float_format.format(value) for value in row.to_numpy(dtype=float))
            handle.write(f"{gene} {formatted_values}\n")

    print(f"Saved: {out}")
