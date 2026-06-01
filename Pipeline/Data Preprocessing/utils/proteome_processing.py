"""
Utility functions for proteomics data processing.
Used by notebooks/Processing/01B_data_process_proteome.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import norm as scipy_norm


def _read_gene_list(gene_list_path):
    with open(gene_list_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _prepare_symbol_index(df):
    prepared = df.copy()
    prepared.index = prepared.index.astype(str).str.strip()
    prepared = prepared.loc[prepared.index.notna()]
    prepared = prepared.loc[prepared.index != ""]
    prepared = prepared[~prepared.index.duplicated(keep="first")]
    return prepared


# ---------------------------------------------------------------------------
# Load and QC
# ---------------------------------------------------------------------------

def load_proteome_data(proteome_file):
    """Load a proteome expression matrix from a tab-separated file.

    Parameters
    ----------
    proteome_file : str or Path
        Path to a TSV file where the first column contains gene identifiers.

    Returns
    -------
    pd.DataFrame
        Proteome matrix indexed by gene ID.
    """
    proteome_df = pd.read_csv(proteome_file, sep="\t", index_col=0)
    proteome_df.index.name = "gene_id"
    proteome_df = _prepare_symbol_index(proteome_df)
    print(f"Proteome data shape: {proteome_df.shape}")
    return proteome_df


def filter_samples_by_kras_vaf(
    expression_df,
    clinical_df,
    maf_path,
    vaf_threshold=0.075,
    extra_removed_samples=None,
):
    """Filter tumor proteome samples using KRAS VAF-derived purity criteria."""
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


def check_na_per_gene(proteome_df):
    """Compute gene-level missing-value fractions and summary statistics.

    Returns
    -------
    na_fraction : pd.Series
        Per-gene missing-value fraction across samples.
    stats : dict
        Dataset-level missingness summary suitable for reports or tables.
    """
    na_fraction = proteome_df.isna().mean(axis=1)
    n_total = proteome_df.shape[0] * proteome_df.shape[1]
    n_na = int(proteome_df.isna().sum().sum())
    stats = {
        "Genes with NAs": f"{int((na_fraction > 0).sum())} / {len(na_fraction)}",
        "Avg NA fraction (per gene)": round(float(na_fraction.mean()), 4),
        "Total NAs": f"{n_na:,} / {n_total:,}",
        "Overall NA fraction": round(n_na / n_total, 4),
    }
    return na_fraction, stats


def summarize_missingness_overview(
    normal_df,
    tumor_df,
    normal_label='Normal (healthy)',
    tumor_label='Tumor (cancer)',
    plot=True,
):
    """Compute and optionally plot the baseline missingness summary for two datasets."""
    na_norm_fraction, na_norm_stats = check_na_per_gene(normal_df)
    na_tumor_fraction, na_tumor_stats = check_na_per_gene(tumor_df)

    summary = pd.DataFrame([
        {'Dataset': normal_label, **na_norm_stats},
        {'Dataset': tumor_label, **na_tumor_stats},
    ]).set_index('Dataset')

    if plot:
        histogram_na_distribution_compare(
            na_norm_fraction,
            na_tumor_fraction,
            label1='Normal',
            label2='Tumor',
        )

    return na_norm_fraction, na_tumor_fraction, summary


def summarize_metadata_coverage(sample_metadata):
    """Return per-column non-missing coverage for aligned sample metadata."""
    return sample_metadata.notna().sum().sort_values(ascending=False)


def histogram_na_distribution_compare(na_fraction_1, na_fraction_2, label1='Normal', label2='Tumor', title='Histogram of NA Fractions per Gene'):
    """Compare two missing-value distributions side by side.

    Parameters
    ----------
    na_fraction_1 : pd.Series or array-like
        Missing-value fractions for the first dataset.
    na_fraction_2 : pd.Series or array-like
        Missing-value fractions for the second dataset.
    label1 : str, default='Normal'
        Title label for the first panel.
    label2 : str, default='Tumor'
        Title label for the second panel.
    title : str
        Overall figure title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, na_fraction, label, color in zip(
        axes,
        [na_fraction_1, na_fraction_2],
        [label1, label2],
        ['steelblue', 'tomato'],
    ):
        ax.hist(na_fraction, bins=50, color=color, edgecolor='black', alpha=0.85)
        ax.set_title(f'{label}')
        ax.set_xlabel('Fraction of NAs')
        ax.set_ylabel('Number of Genes')
        ax.grid(axis='y', alpha=0.75)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def remove_genes_with_high_na(proteome_df, na_fraction, threshold=0.5):
    """Filter out genes whose missing-value fraction exceeds a threshold.

    Parameters
    ----------
    proteome_df : pd.DataFrame
        Proteome matrix indexed by gene.
    na_fraction : pd.Series
        Per-gene missing-value fraction aligned to ``proteome_df.index``.
    threshold : float, default=0.5
        Maximum allowed missing-value fraction.

    Returns
    -------
    pd.DataFrame
        Filtered proteome matrix.
    """
    filtered = proteome_df[na_fraction <= threshold]
    print(f"Genes removed (NA > {threshold*100:.0f}%): {len(proteome_df) - len(filtered)}")
    print(f"Remaining genes: {len(filtered)}")
    return filtered


def summarize_filtered_missingness(
    normal_df,
    tumor_df,
    label1='Normal',
    label2='Tumor',
    plot=True,
):
    """Summarize missingness after high-NA filtering for normal and tumor matrices."""
    if plot:
        histogram_na_distribution_compare(
            normal_df.isna().mean(axis=1),
            tumor_df.isna().mean(axis=1),
            label1=label1,
            label2=label2,
            title='Histogram of NA Fractions per Gene (after cleaning)',
        )

    summary = pd.DataFrame([
        {
            'Dataset': label1,
            'Total NAs': int(normal_df.isna().sum().sum()),
            'Total values': int(normal_df.shape[0] * normal_df.shape[1]),
            'Overall NA fraction': float(normal_df.isna().sum().sum() / (normal_df.shape[0] * normal_df.shape[1])),
        },
        {
            'Dataset': label2,
            'Total NAs': int(tumor_df.isna().sum().sum()),
            'Total values': int(tumor_df.shape[0] * tumor_df.shape[1]),
            'Overall NA fraction': float(tumor_df.isna().sum().sum() / (tumor_df.shape[0] * tumor_df.shape[1])),
        },
    ]).set_index('Dataset')
    return summary


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


def impute_missing_values_qrilc(proteome_df, tune_sigma=1.0, random_state=42):
    """
    QRILC: Quantile Regression Imputation of Left-Censored data (Python port).

    For each protein (row), fits a normal distribution to the observed values
    using quantile regression on the truncated distribution, then draws
    imputed values from the lower tail (below the detection limit).

    Key difference from left-censored (MinProb):
    - Works per gene (row), not per sample (column)
    - Estimates mu and sigma from each gene's observed distribution shape
    - Better accounts for gene-specific detection limits

    Reference:
        Lazar et al. (2016) Accounting for the Multiple Natures of Missing
        Values in Label-Free Quantitative Proteomics Data Sets to Improve
        Identification and Estimation of Protein Differences.
        Journal of Proteome Research, 15(4), 1116-1125.

    Parameters
    ----------
    proteome_df : pd.DataFrame
        Genes x samples matrix with missing values.
    tune_sigma : float, default=1.0
        Multiplier applied to the estimated standard deviation.
    random_state : int, default=42
        Seed controlling the random sampling.

    Returns
    -------
    pd.DataFrame
        Imputed proteome matrix.
    """
    imputed_df = proteome_df.apply(pd.to_numeric, errors='coerce').copy()
    rng = np.random.default_rng(random_state)
    global_min = float(np.nanmin(imputed_df.values))

    for gene in imputed_df.index:
        row = imputed_df.loc[gene]
        missing_mask = row.isna()
        n_missing = int(missing_mask.sum())

        if n_missing == 0:
            continue

        observed = row.loc[~missing_mask].dropna().values.astype(float)
        n_obs = len(observed)

        if n_obs < 2:
            imputed_df.loc[gene, missing_mask] = global_min
            continue

        p_miss = n_missing / (n_obs + n_missing)
        observed_sorted = np.sort(observed)

        j = np.arange(1, n_obs + 1)
        q_positions = p_miss + (1.0 - p_miss) * j / (n_obs + 1)
        q_positions = np.clip(q_positions, 1e-10, 1 - 1e-10)

        z_scores = scipy_norm.ppf(q_positions)

        A = np.column_stack([np.ones_like(z_scores), z_scores])
        coeffs, _, _, _ = np.linalg.lstsq(A, observed_sorted, rcond=None)
        mu = float(coeffs[0])
        sigma = max(abs(float(coeffs[1])) * tune_sigma, 1e-6)

        upper_bound = float(observed_sorted[0])
        b = (upper_bound - mu) / sigma

        seed = int(rng.integers(0, 2**31))
        draws = stats.truncnorm.rvs(
            a=-1000.0, b=b,
            loc=mu, scale=sigma,
            size=n_missing,
            random_state=seed,
        )
        imputed_df.loc[gene, missing_mask] = draws

    return imputed_df


def parse_semicolon_numeric_mean(value):
    """Parse semicolon-separated numeric tokens and return their mean.

    Parameters
    ----------
    value : object
        Value that may contain one or more numeric entries separated by
        semicolons.

    Returns
    -------
    float
        Mean of valid numeric entries, or ``np.nan`` if none are available.
    """
    if pd.isna(value):
        return np.nan

    parts = [part.strip() for part in str(value).split(';') if part.strip()]
    numeric_parts = pd.to_numeric(parts, errors='coerce')
    numeric_parts = numeric_parts[~pd.isna(numeric_parts)]

    if len(numeric_parts) == 0:
        return np.nan

    return float(np.mean(numeric_parts))


def prepare_cptac_cell_composition_metadata(clinical_file, sample_ids, metadata_columns):
    """Load and align CPTAC cell-composition metadata to sample IDs.

    Parameters
    ----------
    clinical_file : str or Path
        Path to the CPTAC clinical metadata table.
    sample_ids : sequence of str
        Sample identifiers to align metadata against.
    metadata_columns : list[str]
        Clinical columns to extract and map onto the sample index.

    Returns
    -------
    pd.DataFrame
        Sample-indexed metadata table containing the requested columns.
    """
    clinical_cptac = pd.read_csv(
        clinical_file,
        sep='\t',
        low_memory=False,
        na_values=["'--", '--', 'NA', 'na', 'Not identified', 'Unknown value', ''],
    )

    clinical_cptac = clinical_cptac.loc[:, ['case_id'] + metadata_columns].copy()
    clinical_cptac = clinical_cptac.drop_duplicates(subset=['case_id']).set_index('case_id')

    for column in metadata_columns:
        clinical_cptac[column] = clinical_cptac[column].map(parse_semicolon_numeric_mean)

    sample_metadata = pd.DataFrame(index=pd.Index(sample_ids, dtype='string').astype(str))
    for column in metadata_columns:
        sample_metadata[column] = sample_metadata.index.map(clinical_cptac[column])

    return sample_metadata


# ---------------------------------------------------------------------------
# Missingness diagnostics
# ---------------------------------------------------------------------------

def summarize_missingness_mechanism(
    proteome_df,
    label,
    sample_metadata=None,
    metadata_columns=None,
    tail_quantile=0.25,
    effect_threshold=0.30,
):
    """Build protein-level diagnostics for missingness patterns.

    Missingness mechanisms cannot be proven from observed data alone. This
    function computes protein-intensity proxies that can indicate patterns
    compatible with MNAR behavior.

    Parameters
    ----------
    proteome_df : pd.DataFrame
        Proteome matrix with genes as rows and samples as columns.
    label : str
        Dataset label used in the summary output.
    tail_quantile : float, default=0.25
        Quantile used to define low- and high-abundance gene subsets.
    effect_threshold : float, default=0.30
        Minimum absolute correlation considered notable.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Summary table and protein-level diagnostic objects for plotting.
    """
    numeric_df = proteome_df.apply(pd.to_numeric, errors='coerce')

    gene_missing_fraction = numeric_df.isna().mean(axis=1)
    gene_observed_median = numeric_df.median(axis=1, skipna=True)
    gene_level = pd.DataFrame({
        'missing_fraction': gene_missing_fraction,
        'observed_median_intensity': gene_observed_median,
    })

    gene_intensity_missing_corr = gene_missing_fraction.corr(gene_observed_median, method='spearman')

    low_cutoff = gene_observed_median.quantile(tail_quantile)
    high_cutoff = gene_observed_median.quantile(1 - tail_quantile)
    low_abundance_missing = gene_missing_fraction.loc[gene_observed_median <= low_cutoff].mean()
    high_abundance_missing = gene_missing_fraction.loc[gene_observed_median >= high_cutoff].mean()

    if pd.notna(high_abundance_missing) and high_abundance_missing > 0:
        low_high_missing_ratio = low_abundance_missing / high_abundance_missing
    else:
        low_high_missing_ratio = np.nan

    evidence_flags = []
    if pd.notna(gene_intensity_missing_corr) and gene_intensity_missing_corr <= -effect_threshold:
        evidence_flags.append('MNAR-compatible: missingness rises as observed protein intensity drops.')
    if pd.notna(low_high_missing_ratio) and low_high_missing_ratio >= 1.5:
        evidence_flags.append('MNAR-compatible: low-abundance proteins have substantially more missingness than high-abundance proteins.')
    if not evidence_flags:
        evidence_flags.append('No strong proxy pattern detected; missingness mechanism remains unclear.')

    strongest_metadata_feature = pd.NA
    strongest_metadata_spearman = np.nan
    sample_level = None
    if sample_metadata is not None and metadata_columns:
        sample_level = pd.DataFrame({
            'sample_missing_fraction': numeric_df.isna().mean(axis=0)
        })
        metadata = sample_metadata.copy()
        metadata.index = metadata.index.astype(str)
        sample_level.index = sample_level.index.astype(str)
        sample_level = sample_level.join(metadata, how='left')

        metadata_scores = []
        for column in metadata_columns:
            if column not in sample_level.columns:
                continue
            feature = pd.to_numeric(sample_level[column], errors='coerce')
            valid = feature.notna() & sample_level['sample_missing_fraction'].notna()
            if valid.sum() < 3:
                continue
            if feature.loc[valid].nunique(dropna=True) < 2:
                continue
            corr = sample_level.loc[valid, 'sample_missing_fraction'].corr(feature.loc[valid], method='spearman')
            if pd.notna(corr):
                metadata_scores.append((column, float(corr)))

        if metadata_scores:
            strongest_metadata_feature, strongest_metadata_spearman = max(
                metadata_scores,
                key=lambda item: abs(item[1]),
            )

    summary = pd.DataFrame([{
        'dataset': label,
        'gene_intensity_vs_missing_spearman': gene_intensity_missing_corr,
        'low_abundance_missing_fraction': low_abundance_missing,
        'high_abundance_missing_fraction': high_abundance_missing,
        'low_high_missing_ratio': low_high_missing_ratio,
        'strongest_metadata_feature': strongest_metadata_feature,
        'strongest_metadata_spearman': strongest_metadata_spearman,
        'n_genes': int(numeric_df.shape[0]),
        'n_samples': int(numeric_df.shape[1]),
    }])

    details = {
        'gene_level': gene_level,
        'evidence_flags': evidence_flags,
        'sample_level': sample_level,
    }
    return summary, details


def plot_missingness_mechanism(details, label, ax=None, show=True):
    """Plot exploratory protein-level missingness diagnostics.

    Parameters
    ----------
    details : dict
        Detailed output returned by ``summarize_missingness_mechanism``.
    label : str
        Dataset label used in plot titles.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on. If omitted, a new figure is created.
    show : bool, default=True
        Whether to show the figure when this function creates it.
    """
    gene_level = details['gene_level']

    created_fig = False
    if ax is None:
        fig, scatter_ax = plt.subplots(figsize=(6.5, 4.5))
        created_fig = True
    else:
        scatter_ax = ax
        fig = scatter_ax.figure

    scatter_ax.scatter(
        gene_level['observed_median_intensity'],
        gene_level['missing_fraction'],
        s=18,
        alpha=0.5,
        color='#F7DC6F',
        edgecolors='#B7950B',
        linewidths=0.25,
        rasterized=True,
    )
    scatter_ax.set_title(f'{label}: abundance vs missingness')
    scatter_ax.set_xlabel('Observed median intensity')
    scatter_ax.set_ylabel('Missing fraction')
    scatter_ax.grid(alpha=0.25)

    if created_fig:
        plt.tight_layout()
    if show and created_fig:
        plt.show()


def run_missingness_mechanism_diagnostics(
    proteome_df,
    label,
    sample_metadata=None,
    metadata_columns=None,
    ax=None,
    show=True,
    plot_label=None,
):
    """Run, print, and plot missingness diagnostics for one dataset.

    Parameters
    ----------
    proteome_df : pd.DataFrame
        Proteome matrix to diagnose.
    label : str
        Dataset label used in printed output and plots.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw the missingness plot on.
    show : bool, default=True
        Whether to show the figure when the plotting helper creates it.
    plot_label : str, optional
        Alternate shorter label used only in the plot title.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Summary table and detailed diagnostics.
    """
    summary, details = summarize_missingness_mechanism(
        proteome_df,
        label,
        sample_metadata=sample_metadata,
        metadata_columns=metadata_columns,
    )

    print(f'\nMissingness diagnostics for {label}')
    print(summary.round(3).to_string(index=False))
    print('\nInterpretation:')
    for line in details['evidence_flags']:
        print(f'- {line}')

    plot_missingness_mechanism(details, plot_label or label, ax=ax, show=show)
    return summary, details


def intersect_proteome_genes(proteome_df1, proteome_df2, out_dir=None):
    """Restrict two proteome matrices to their shared gene set.

    Parameters
    ----------
    proteome_df1 : pd.DataFrame
        First proteome matrix.
    proteome_df2 : pd.DataFrame
        Second proteome matrix.
    out_dir : str or Path, optional
        Output directory for writing the shared gene list.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Copies of both matrices filtered to the same sorted gene index.
    """
    common_genes = proteome_df1.index.intersection(proteome_df2.index).sort_values()
    print(f"Number of common genes between datasets: {len(common_genes)}")

    if out_dir is not None:
        out_path = Path(out_dir) / "proteome_common_genes.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for gene in common_genes:
                f.write(str(gene) + "\n")

    return proteome_df1.loc[common_genes], proteome_df2.loc[common_genes]


def harmonize_proteome_with_gene_list(proteome_df1, proteome_df2, gene_list_path, out_gene_list=None):
    """Restrict proteome normal and tumor matrices to a shared gene-symbol list."""
    proteome_df1 = _prepare_symbol_index(proteome_df1)
    proteome_df2 = _prepare_symbol_index(proteome_df2)
    reference_genes = set(_read_gene_list(gene_list_path))

    final_genes = sorted(set(proteome_df1.index) & set(proteome_df2.index) & reference_genes)
    filtered_1 = proteome_df1.loc[final_genes].copy()
    filtered_2 = proteome_df2.loc[final_genes].copy()

    if out_gene_list is not None:
        out_gene_list = Path(out_gene_list)
        out_gene_list.parent.mkdir(parents=True, exist_ok=True)
        with open(out_gene_list, "w") as handle:
            for gene in final_genes:
                handle.write(f"{gene}\n")

    print(f"Final harmonized proteome genes: {len(final_genes)}")
    return filtered_1, filtered_2, final_genes


def finalize_proteome_matrices(
    normal_df,
    tumor_df,
    mapping_dir,
    final_common_genes=None,
):
    """Intersect proteome matrices and optionally harmonize them to a final shared gene list."""
    normal_df, tumor_df = intersect_proteome_genes(normal_df, tumor_df, out_dir=mapping_dir)

    if final_common_genes is not None and Path(final_common_genes).exists():
        normal_df, tumor_df, final_genes = harmonize_proteome_with_gene_list(
            normal_df,
            tumor_df,
            final_common_genes,
        )
        print('Applied final harmonization against RNA-proteome common genes.')
    else:
        final_genes = normal_df.index.tolist()
        if final_common_genes is not None:
            print(f'Final RNA gene list not found at {final_common_genes}; keeping proteome-common genes only.')

    print('Proteome Normal Data Shape:', normal_df.shape)
    print('Proteome Tumor Data Shape:', tumor_df.shape)
    return normal_df, tumor_df, final_genes


def save_proteome_outputs(
    normal_df,
    tumor_df,
    out_proteome_norm_header,
    out_proteome_tumor_header,
    out_proteome_norm_no_header=None,
    out_proteome_tumor_no_header=None,
):
    """Save proteome outputs with headers and optional CSD-ready no-header files."""
    normal_df = _prepare_symbol_index(normal_df).sort_index()
    tumor_df = _prepare_symbol_index(tumor_df).sort_index()

    Path(out_proteome_norm_header).parent.mkdir(parents=True, exist_ok=True)
    Path(out_proteome_tumor_header).parent.mkdir(parents=True, exist_ok=True)

    normal_df.reset_index().to_csv(out_proteome_norm_header, sep='\t', header=True, index=False)
    print('Saved:', out_proteome_norm_header)
    tumor_df.reset_index().to_csv(out_proteome_tumor_header, sep='\t', header=True, index=False)
    print('Saved:', out_proteome_tumor_header)

    if out_proteome_norm_no_header is not None:
        export_csd_input(normal_df, out_proteome_norm_no_header)
    if out_proteome_tumor_no_header is not None:
        export_csd_input(tumor_df, out_proteome_tumor_no_header)
    return normal_df, tumor_df



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
