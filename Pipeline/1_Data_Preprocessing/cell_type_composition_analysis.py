"""Plot CPTAC tissue composition before and after filtering for matched omics samples."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COMPOSITION_COLUMNS_CPTAC = [
    "Neoplastic_cellularity",
    "Acinar_fraction",
    "Islet_fraction",
    "Stromal_fraction",
    "Non_neoplastic_duct",
    "Fat_fraction",
    "Inflammation_fraction",
    "Muscle_fraction",
]


CPTAC_LABEL_MAP = {
    "Neoplastic_cellularity": "Neoplastic",
    "Acinar_fraction": "Acinar",
    "Islet_fraction": "Islet",
    "Stromal_fraction": "Stroma",
    "Non_neoplastic_duct": "Duct",
    "Fat_fraction": "Fat",
    "Inflammation_fraction": "Inflammation",
    "Muscle_fraction": "Muscle",
}

def parse_fraction_entry(value: object) -> float:
    """Convert a clinical fraction field to a single numeric value.

    Semicolon-separated numeric entries are averaged, while common missing-value
    markers are mapped to ``np.nan``.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text in {"", "NA", "na", "N/A", "Not Available", "Not available", "Unknown value", "Not identified", "'--"}:
        return np.nan

    parts = [part.strip() for part in text.split(";") if part.strip() not in {"", "NA", "na", "N/A"}]
    numeric_parts = pd.to_numeric(pd.Series(parts), errors="coerce").dropna()
    if numeric_parts.empty:
        return np.nan

    return float(numeric_parts.mean())


def load_expression_columns(csv_path: Path) -> pd.Index:
    """Read only the sample columns from an expression matrix."""
    return pd.read_csv(csv_path, nrows=0, sep=None, engine="python").columns[1:]


def prepare_cptac_composition_table(
    clinical_df: pd.DataFrame,
    expr_columns: pd.Index,
    sample_col: str = "case_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Align CPTAC composition metadata to expression samples and summarize it."""
    available_columns = [sample_col] + [col for col in COMPOSITION_COLUMNS_CPTAC if col in clinical_df.columns]
    composition_df = clinical_df[available_columns].copy()
    composition_df = composition_df.drop_duplicates(subset=sample_col)
    composition_df[sample_col] = composition_df[sample_col].astype(str).str.strip()

    for column in COMPOSITION_COLUMNS_CPTAC:
        if column in composition_df.columns:
            composition_df[column] = composition_df[column].map(parse_fraction_entry)

    matched_samples = sorted(set(expr_columns.astype(str)).intersection(set(composition_df[sample_col])))
    matched_df = composition_df[composition_df[sample_col].isin(matched_samples)].copy()

    long_df = matched_df.melt(
        id_vars=sample_col,
        value_vars=[col for col in COMPOSITION_COLUMNS_CPTAC if col in matched_df.columns],
        var_name="component",
        value_name="fraction",
    ).dropna(subset=["fraction"])

    summary_df = (
        long_df.groupby("component")
        .agg(
            n_samples=("fraction", "size"),
            mean_fraction=("fraction", "mean"),
            median_fraction=("fraction", "median"),
            std_fraction=("fraction", "std"),
        )
        .sort_values("mean_fraction", ascending=False)
        .reset_index()
    )

    return matched_df, long_df, summary_df, matched_samples


def save_composition_plot(long_df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save a single-cohort tissue-composition boxplot with sample points."""
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=long_df,
        x="component",
        y="fraction",
        color="#D4A373",
        fliersize=2,
    )
    sns.stripplot(
        data=long_df,
        x="component",
        y="fraction",
        color="#3A3A3A",
        alpha=0.45,
        size=3,
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Estimated fraction / cellularity")
    plt.xlabel("")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_before_after_plot(
    before_long_df: pd.DataFrame,
    after_long_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a side-by-side before-versus-after tissue-composition plot."""
    comparison_order = list(CPTAC_LABEL_MAP.values())
    before_after_long_df = pd.concat(
        [
            before_long_df.assign(cohort=f"Before filtering (n={before_long_df['case_id'].nunique()})"),
            after_long_df.assign(cohort=f"After filtering (n={after_long_df['case_id'].nunique()})"),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.25))
    sns.boxplot(
        data=before_after_long_df,
        x="component",
        y="fraction",
        hue="cohort",
        order=comparison_order,
        palette=["#D9DDE2", "#2B8C8C"],
        fliersize=2,
        linewidth=1,
        ax=ax,
    )
    plt.setp(ax.get_xticklabels(), rotation=27, ha="right", fontsize=9, fontweight="semibold")
    ax.set_ylabel("Estimated tissue fraction (%)", fontsize=10.5, fontweight="semibold", labelpad=5)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=9, width=0.9)
    ax.tick_params(axis="x", width=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.legend(title="", frameon=False, loc="upper right", fontsize=9.5)
    fig.tight_layout(pad=0.55)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset and path selection."""
    parser = argparse.ArgumentParser(
        description="Visualize CPTAC tissue composition before and after omics filtering."
    )
    parser.add_argument(
        "--dataset",
        choices=["proteome", "transcriptome", "all"],
        default="proteome",
        help="Choose which dataset plot(s) to generate.",
    )
    parser.add_argument(
        "--before-source",
        choices=["processed", "raw"],
        default="processed",
        help="Choose the default before-filtering proteome file.",
    )
    parser.add_argument(
        "--before-path",
        type=Path,
        help="Optional explicit path for the before-filtering expression matrix.",
    )
    parser.add_argument(
        "--after-path",
        type=Path,
        help="Optional explicit path for the after-filtering expression matrix.",
    )
    parser.add_argument(
        "--clinical-path",
        type=Path,
        help="Optional explicit path for the CPTAC clinical metadata table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit output directory for the plots.",
    )
    return parser.parse_args()


def get_default_before_path(root: Path, before_source: str) -> Path:
    """Return the default proteome before-filtering matrix for the selected source."""
    if before_source == "raw":
        return root / "data" / "raw" / "proteome" / "proteomics_cancer.txt"

    return root / "data" / "processed" / "proteome" / "proteome_cancer_counts.txt"


def main() -> None:
    """Generate tissue-composition plots for the requested omics dataset(s)."""
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or (root / "results" / "cell_composition")
    output_dir.mkdir(parents=True, exist_ok=True)

    cptac_clinical_path = args.clinical_path or (root / "data" / "clinical" / "clinical_CPTAC.tsv")
    clinical_cptac = pd.read_csv(cptac_clinical_path, sep="\t")

    before_expr_path = args.before_path or get_default_before_path(root, args.before_source)
    after_expr_path = args.after_path or (
        root / "data" / "processed" / "proteome" / "proteome_cancer_FINAL_FINAL_with_header.txt"
    )

    all_dataset_configs = {
        "proteome": {
            "label": "proteome",
            "before_expr_path": before_expr_path,
            "after_expr_path": after_expr_path,
            "output_name": "cptac_tissue_composition_proteome.png",
            "before_after_output_name": "cptac_tissue_composition_proteome_before_after_filtering.png",
            "title": "CPTAC tissue composition for matched proteome samples",
        },
        "transcriptome": {
            "label": "transcriptome",
            "before_expr_path": root / "data" / "processed" / "rna_cancer_pre_harmonized.txt",
            "after_expr_path": root / "data" / "processed" / "transcriptome" / "rna_cancer_FINAL_FINAL.txt",
            "output_name": "cptac_tissue_composition_transcriptome.png",
            "before_after_output_name": "cptac_tissue_composition_transcriptome_before_after_filtering.png",
            "title": "CPTAC tissue composition for matched transcriptome samples",
        },
    }

    if args.dataset == "all":
        dataset_configs = list(all_dataset_configs.values())
    else:
        dataset_configs = [all_dataset_configs[args.dataset]]

    for dataset in dataset_configs:
        before_expr_columns = load_expression_columns(dataset["before_expr_path"])
        after_expr_columns = load_expression_columns(dataset["after_expr_path"])
        _, before_cptac_long_df, _, before_matched_samples = prepare_cptac_composition_table(
            clinical_cptac,
            before_expr_columns,
            sample_col="case_id",
        )
        _, after_cptac_long_df, _, after_matched_samples = prepare_cptac_composition_table(
            clinical_cptac,
            after_expr_columns,
            sample_col="case_id",
        )

        before_cptac_long_df["component"] = before_cptac_long_df["component"].map(CPTAC_LABEL_MAP).fillna(
            before_cptac_long_df["component"]
        )
        after_cptac_long_df["component"] = after_cptac_long_df["component"].map(CPTAC_LABEL_MAP).fillna(
            after_cptac_long_df["component"]
        )

        save_composition_plot(
            after_cptac_long_df,
            output_dir / dataset["output_name"],
            dataset["title"],
        )
        save_before_after_plot(
            before_cptac_long_df,
            after_cptac_long_df,
            output_dir / dataset["before_after_output_name"],
        )
        print(
            f"Matched CPTAC {dataset['label']} samples before filtering: {len(before_matched_samples)}"
        )
        print(
            f"Matched CPTAC {dataset['label']} samples after filtering: {len(after_matched_samples)}"
        )
        print(f"Before path: {dataset['before_expr_path']}")
        print(f"After path: {dataset['after_expr_path']}")

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
