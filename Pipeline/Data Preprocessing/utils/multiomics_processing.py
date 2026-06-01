"""Utility functions for multi-omics harmonization of transcriptome and proteome outputs."""

from pathlib import Path

import pandas as pd

from utils.proteome_processing import save_proteome_outputs
from utils.rna_processing import save_processed_rna_matrices


def _prepare_symbol_index(df):
    """Clean up the gene symbol index of a transcriptome or proteome matrix."""
    prepared = df.copy()
    prepared.index = prepared.index.astype(str).str.strip()
    prepared = prepared.loc[prepared.index.notna()]
    prepared = prepared.loc[prepared.index != ""]
    prepared = prepared[~prepared.index.duplicated(keep="first")]
    return prepared


def load_expression_matrix(matrix_path, sep="\t"):
    """Load a saved expression matrix and use the first column as gene index."""
    matrix = pd.read_csv(matrix_path, sep=sep, index_col=0)
    matrix.index.name = "gene_id"
    return _prepare_symbol_index(matrix)


def harmonize_multiomics_matrices(
    rna_tumor_df,
    rna_normal_df,
    proteome_tumor_df,
    proteome_normal_df,
    out_gene_list=None,
    extra_gene_lists=None,
):
    """Restrict transcriptome and proteome matrices to one shared gene-symbol set."""
    rna_tumor_df = _prepare_symbol_index(rna_tumor_df)
    rna_normal_df = _prepare_symbol_index(rna_normal_df)
    proteome_tumor_df = _prepare_symbol_index(proteome_tumor_df)
    proteome_normal_df = _prepare_symbol_index(proteome_normal_df)

    final_genes = sorted(
        set(rna_tumor_df.index)
        & set(rna_normal_df.index)
        & set(proteome_tumor_df.index)
        & set(proteome_normal_df.index)
    )

    rna_tumor_final = rna_tumor_df.loc[final_genes].copy()
    rna_normal_final = rna_normal_df.loc[final_genes].copy()
    proteome_tumor_final = proteome_tumor_df.loc[final_genes].copy()
    proteome_normal_final = proteome_normal_df.loc[final_genes].copy()

    gene_list_paths = []
    if out_gene_list is not None:
        gene_list_paths.append(Path(out_gene_list))
    for extra_path in extra_gene_lists or []:
        gene_list_paths.append(Path(extra_path))

    for gene_list_path in gene_list_paths:
        gene_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gene_list_path, "w", encoding="utf-8") as handle:
            for gene in final_genes:
                handle.write(f"{gene}\n")
        print(f"Saved: {gene_list_path}")

    print(f"Final harmonized multi-omics genes: {len(final_genes)}")
    print("RNA tumor shape:", rna_tumor_final.shape)
    print("RNA normal shape:", rna_normal_final.shape)
    print("Proteome tumor shape:", proteome_tumor_final.shape)
    print("Proteome normal shape:", proteome_normal_final.shape)
    return (
        rna_tumor_final,
        rna_normal_final,
        proteome_tumor_final,
        proteome_normal_final,
        final_genes,
    )


def save_final_multiomics_outputs(
    rna_tumor_df,
    rna_normal_df,
    proteome_tumor_df,
    proteome_normal_df,
    out_rna_tumor,
    out_rna_normal,
    out_rna_tumor_no_header,
    out_rna_normal_no_header,
    out_proteome_normal_header,
    out_proteome_tumor_header,
    out_proteome_normal_no_header,
    out_proteome_tumor_no_header,
):
    """Save the final harmonized transcriptome and proteome outputs."""
    save_processed_rna_matrices(
        rna_tumor_df,
        rna_normal_df,
        tumor_out=out_rna_tumor,
        normal_out=out_rna_normal,
        tumor_csd_out=out_rna_tumor_no_header,
        normal_csd_out=out_rna_normal_no_header,
    )
    save_proteome_outputs(
        proteome_normal_df,
        proteome_tumor_df,
        out_proteome_norm_header=out_proteome_normal_header,
        out_proteome_tumor_header=out_proteome_tumor_header,
        out_proteome_norm_no_header=out_proteome_normal_no_header,
        out_proteome_tumor_no_header=out_proteome_tumor_no_header,
    )
