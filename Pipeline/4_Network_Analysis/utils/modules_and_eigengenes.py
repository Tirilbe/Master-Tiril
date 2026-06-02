from __future__ import annotations

from pathlib import Path
from typing import Any

import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import re
import seaborn as sns
from matplotlib.colors import to_hex
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

from utils.network_and_gcc import (
    apply_csd_style,
    apply_cytoscape_layout,
    convert_ensembl_to_symbol,
    cytoscape_client,
    load_edges_clean,
)
from utils.topology_and_hubs import load_graph


def _ensure_graph_has_weight(
    graph: nx.Graph,
    edge_input: str | Path | pd.DataFrame | None = None,
) -> nx.Graph:
    weighted_graph = graph.copy()
    weight_lookup: dict[tuple[str, str], float] = {}

    if edge_input is not None:
        weighted_edges = load_edges_clean(edge_input, convert_symbols=True)
        weighted_edges["weight"] = weighted_edges["value"].where(weighted_edges["value"].notna(), 1.0)
        weight_lookup = {
            tuple(sorted((str(row.source).strip(), str(row.target).strip()))): float(row.weight)
            for row in weighted_edges.itertuples(index=False)
        }

    for source, target, edge_data in weighted_graph.edges(data=True):
        edge_key = tuple(sorted((str(source).strip(), str(target).strip())))
        current_weight = pd.to_numeric(pd.Series([edge_data.get("weight")]), errors="coerce").iloc[0]
        current_value = pd.to_numeric(pd.Series([edge_data.get("value")]), errors="coerce").iloc[0]

        if pd.notna(current_value):
            edge_data["value"] = float(current_value)

        if pd.notna(current_weight):
            edge_data["weight"] = float(current_weight)
        elif edge_key in weight_lookup:
            edge_data["weight"] = weight_lookup[edge_key]
            if pd.isna(current_value):
                edge_data["value"] = weight_lookup[edge_key]
        elif pd.notna(current_value):
            edge_data["weight"] = float(current_value)
        else:
            edge_data["weight"] = 1.0

    return weighted_graph
def leiden_modules_with_edge_types(
    graph: nx.Graph | str | Path | pd.DataFrame,
    edges_df: str | Path | pd.DataFrame,
    resolution: float = 1.0,
    n_iterations: int = -1,
    output_dir: str | Path = "Modules",
) -> list[list[str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edges_df = load_edges_clean(edges_df, convert_symbols=False)

    if not isinstance(graph, nx.Graph):
        graph = load_graph(graph)
    graph = _ensure_graph_has_weight(graph, edges_df)

    ig_graph = ig.Graph.from_networkx(graph)
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=n_iterations,
    )

    original_node_names = [str(name).strip() for name in ig_graph.vs["_nx_name"]]
    ensembl_ids = [gene for gene in original_node_names if gene.startswith("ENSG")]
    id_to_symbol = convert_ensembl_to_symbol(ensembl_ids) if ensembl_ids else {}

    def map_gene_name(gene: str) -> str:
        gene = str(gene).strip()
        if gene.startswith("ENSG"):
            return id_to_symbol.get(gene) or gene
        return gene

    edges_df["source"] = edges_df["source"].map(map_gene_name)
    edges_df["target"] = edges_df["target"].map(map_gene_name)

    modules: list[list[str]] = []
    rows: list[tuple[str, str, int]] = []
    edge_rows: list[tuple[str, str, str, str, Any]] = []

    for index, community in enumerate(partition, start=1):
        module_genes = [map_gene_name(original_node_names[node_index]) for node_index in community]
        modules.append(module_genes)
        (output_dir / f"module_{index}.txt").write_text("\n".join(module_genes) + "\n", encoding="utf-8")

        module_size = len(module_genes)
        for gene in module_genes:
            rows.append((gene, str(index), module_size))

        module_set = set(module_genes)
        module_edges = edges_df[edges_df["source"].isin(module_set) & edges_df["target"].isin(module_set)].copy()
        if "type" not in module_edges.columns:
            module_edges["type"] = ""
        if "value" not in module_edges.columns:
            module_edges["value"] = np.nan
        module_edges = module_edges[["source", "target", "type", "value"]]
        module_edges.to_csv(output_dir / f"module_{index}_edges.tsv", sep="\t", index=False)

        for _, row in module_edges.iterrows():
            edge_rows.append((str(index), row["source"], row["target"], row["type"], row["value"]))

    module_table = pd.DataFrame(rows, columns=["Gene", "Module", "Module_size"])
    module_edge_table = pd.DataFrame(edge_rows, columns=["Module", "source", "target", "type", "value"])
    module_table.to_csv(output_dir / "module_table.csv", index=False)
    module_edge_table.to_csv(output_dir / "module_edge_table.tsv", sep="\t", index=False)

    print(f"{len(modules)} modules detected")
    print(f"Saved module edge files with edge type to {output_dir}")
    return modules


def compute_rewiring_scores(module_edge_table: pd.DataFrame, output_file: str | Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for module_id, module_edges in module_edge_table.groupby("Module"):
        type_counts = module_edges["type"].value_counts()
        c_count = type_counts.get("C", 0)
        s_count = type_counts.get("S", 0)
        d_count = type_counts.get("D", 0)
        total = c_count + s_count + d_count
        if total == 0:
            c_frac = s_frac = d_frac = homogeneity_score = 0.0
        else:
            c_frac = c_count / total
            s_frac = s_count / total
            d_frac = d_count / total
            homogeneity_score = max(c_frac, s_frac, d_frac)

        dominant_type = "N/A" if total == 0 else ("C" if c_frac == max(c_frac, s_frac, d_frac) else "S" if s_frac == max(c_frac, s_frac, d_frac) else "D")
        rows.append(
            {
                "Module": str(module_id),
                "n_edges": total,
                "C_fraction": round(c_frac, 4),
                "S_fraction": round(s_frac, 4),
                "D_fraction": round(d_frac, 4),
                "homogeneity_score": round(homogeneity_score, 4),
                "dominant_type": dominant_type,
            }
        )

    result_df = pd.DataFrame(rows).sort_values("Module").reset_index(drop=True)
    if output_file is not None:
        result_df.to_csv(output_file, index=False)
        print(f"Saved rewiring scores for {len(result_df)} modules to {output_file}")
    return result_df


def normalize_module_table(module_data: pd.DataFrame | dict | list) -> pd.DataFrame:
    if isinstance(module_data, pd.DataFrame):
        module_table = module_data.copy()
        if "module" in module_table.columns and "Module" not in module_table.columns:
            module_table = module_table.rename(columns={"module": "Module"})
        if "gene" in module_table.columns and "Gene" not in module_table.columns:
            module_table = module_table.rename(columns={"gene": "Gene"})
        if not {"Gene", "Module"}.issubset(module_table.columns):
            raise ValueError("Module DataFrame must contain 'Gene' and 'Module' columns")
        module_table = module_table[["Gene", "Module"]].copy()
    elif isinstance(module_data, dict):
        rows = []
        for module, genes in module_data.items():
            for gene in genes:
                rows.append({"Gene": gene, "Module": module})
        module_table = pd.DataFrame(rows)
    elif isinstance(module_data, list):
        rows = []
        for index, item in enumerate(module_data, start=1):
            genes = item if isinstance(item, (list, tuple, set, np.ndarray, pd.Series)) else [item]
            for gene in genes:
                rows.append({"Gene": gene, "Module": str(index)})
        module_table = pd.DataFrame(rows)
    else:
        raise TypeError("module_data must be a DataFrame, dict, or list-like collection")

    if module_table.empty:
        return pd.DataFrame(columns=["Gene", "Module"])

    module_table["Gene"] = module_table["Gene"].astype(str).str.strip()
    module_table["Module"] = module_table["Module"].astype(str).str.strip()
    module_table = module_table[(module_table["Gene"] != "") & (module_table["Module"] != "")].drop_duplicates()
    return module_table.reset_index(drop=True)


def _sort_module_ids(module_ids: list[str]) -> list[str]:
    try:
        return sorted(module_ids, key=lambda value: int(value))
    except (TypeError, ValueError):
        return sorted(module_ids, key=str)


def _module_gene_sets(module_data: pd.DataFrame | dict | list) -> dict[str, set[str]]:
    module_table = normalize_module_table(module_data)
    if module_table.empty:
        return {}

    return {
        str(module_id): set(group["Gene"].astype(str).str.strip())
        for module_id, group in module_table.groupby("Module")
    }


def compute_module_overlap_matrices(
    module_data_a: pd.DataFrame | dict | list,
    module_data_b: pd.DataFrame | dict | list,
    label_a: str = "RNA",
    label_b: str = "Protein",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    module_sets_a = _module_gene_sets(module_data_a)
    module_sets_b = _module_gene_sets(module_data_b)

    module_ids_a = _sort_module_ids(list(module_sets_a))
    module_ids_b = _sort_module_ids(list(module_sets_b))

    overlap_counts = pd.DataFrame(0, index=module_ids_a, columns=module_ids_b, dtype=int)
    overlap_jaccard = pd.DataFrame(0.0, index=module_ids_a, columns=module_ids_b, dtype=float)
    summary_rows: list[dict[str, Any]] = []

    for module_a in module_ids_a:
        genes_a = module_sets_a[module_a]
        for module_b in module_ids_b:
            genes_b = module_sets_b[module_b]
            overlap = sorted(genes_a & genes_b)
            union_size = len(genes_a | genes_b)
            overlap_count = len(overlap)
            jaccard = (overlap_count / union_size) if union_size else 0.0

            overlap_counts.loc[module_a, module_b] = overlap_count
            overlap_jaccard.loc[module_a, module_b] = jaccard
            summary_rows.append(
                {
                    f"{label_a.lower()}_module": module_a,
                    f"{label_b.lower()}_module": module_b,
                    "overlap_genes": ", ".join(overlap[:20]),
                    "overlap_count": overlap_count,
                    "jaccard": float(jaccard),
                }
            )

    overlap_counts.index.name = f"{label_a} module"
    overlap_counts.columns.name = f"{label_b} module"
    overlap_jaccard.index.name = f"{label_a} module"
    overlap_jaccard.columns.name = f"{label_b} module"

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["jaccard", "overlap_count"], ascending=[False, False]).reset_index(drop=True)

    return overlap_counts, overlap_jaccard, summary_df


def plot_module_overlap_heatmaps(
    overlap_counts: pd.DataFrame,
    overlap_jaccard: pd.DataFrame,
    label_a: str = "RNA",
    label_b: str = "Protein",
    figsize: tuple[int, int] = (12, 9),
) -> tuple[plt.Figure, plt.Figure]:
    counts_fig, counts_ax = plt.subplots(figsize=figsize)
    sns.heatmap(overlap_counts, cmap="YlOrRd", annot=True, fmt="d", linewidths=0.5, ax=counts_ax)
    counts_ax.set_title(f"{label_a} vs {label_b} module overlap counts")
    counts_ax.set_xlabel(f"{label_b} modules")
    counts_ax.set_ylabel(f"{label_a} modules")
    counts_fig.tight_layout()

    jaccard_fig, jaccard_ax = plt.subplots(figsize=figsize)
    sns.heatmap(overlap_jaccard, cmap="mako", annot=True, fmt=".2f", linewidths=0.5, vmin=0.0, vmax=1.0, ax=jaccard_ax)
    jaccard_ax.set_title(f"{label_a} vs {label_b} module overlap Jaccard")
    jaccard_ax.set_xlabel(f"{label_b} modules")
    jaccard_ax.set_ylabel(f"{label_a} modules")
    jaccard_fig.tight_layout()

    return counts_fig, jaccard_fig


def save_module_overlap_outputs(
    module_data_a: pd.DataFrame | dict | list,
    module_data_b: pd.DataFrame | dict | list,
    output_dir: str | Path,
    label_a: str = "RNA",
    label_b: str = "Protein",
    save_heatmaps: bool = True,
) -> dict[str, pd.DataFrame | Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlap_counts, overlap_jaccard, overlap_summary = compute_module_overlap_matrices(
        module_data_a,
        module_data_b,
        label_a=label_a,
        label_b=label_b,
    )

    counts_csv_path = output_dir / "rna_protein_module_overlap_counts.csv"
    jaccard_csv_path = output_dir / "rna_protein_module_overlap_jaccard.csv"
    summary_csv_path = output_dir / "rna_protein_module_overlap_summary.csv"

    overlap_counts.to_csv(counts_csv_path)
    overlap_jaccard.to_csv(jaccard_csv_path)
    overlap_summary.to_csv(summary_csv_path, index=False)

    result: dict[str, pd.DataFrame | Path] = {
        "overlap_counts": overlap_counts,
        "overlap_jaccard": overlap_jaccard,
        "overlap_summary": overlap_summary,
        "counts_csv_path": counts_csv_path,
        "jaccard_csv_path": jaccard_csv_path,
        "summary_csv_path": summary_csv_path,
    }

    if save_heatmaps:
        counts_fig, jaccard_fig = plot_module_overlap_heatmaps(
            overlap_counts,
            overlap_jaccard,
            label_a=label_a,
            label_b=label_b,
        )
        counts_png_path = output_dir / "rna_protein_module_overlap_counts.png"
        jaccard_png_path = output_dir / "rna_protein_module_overlap_jaccard.png"
        counts_fig.savefig(counts_png_path, dpi=300, bbox_inches="tight")
        jaccard_fig.savefig(jaccard_png_path, dpi=300, bbox_inches="tight")
        plt.close(counts_fig)
        plt.close(jaccard_fig)
        result["counts_png_path"] = counts_png_path
        result["jaccard_png_path"] = jaccard_png_path

    return result


def load_expression_matrix(path: str | Path, sample_prefix: str | None = None) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        first_line = handle.readline().rstrip("\r\n")
        second_line = handle.readline().rstrip("\r\n")

    stripped_first_line = first_line.strip()
    first_tokens = [token for token in re.split(r"[\s,]+", stripped_first_line) if token]
    second_tokens = [token for token in re.split(r"[\s,]+", second_line.strip()) if token]
    has_header = stripped_first_line.startswith("gene_id") or (
        bool(first_tokens) and len(second_tokens) == len(first_tokens) + 1
    )
    if has_header:
        df = pd.read_csv(path, sep=r"\s+|,", engine="python", index_col=0)
        df.columns = [str(column).strip() for column in df.columns]
    else:
        df = pd.read_csv(path, sep=r"\s+", header=None)
        df = df.rename(columns={0: "gene_id"}).set_index("gene_id")
        sample_prefix = sample_prefix or "sample"
        df.columns = [f"{sample_prefix}_{index + 1}" for index in range(df.shape[1])]

    df.index = df.index.astype(str).str.strip()
    return df.apply(pd.to_numeric, errors="coerce")


def convert_expression_index_to_symbols(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    original_ids = converted.index.astype(str).str.strip().tolist()
    ensembl_ids = [gene_id for gene_id in original_ids if gene_id.startswith("ENSG")]

    if ensembl_ids:
        id_to_symbol = convert_ensembl_to_symbol(sorted(set(ensembl_ids)))
        converted.index = [id_to_symbol.get(gene_id) or gene_id if gene_id.startswith("ENSG") else gene_id for gene_id in original_ids]
    else:
        converted.index = original_ids

    converted.index = pd.Index(pd.Series(converted.index).astype(str).str.strip(), name="gene_symbol")
    converted = converted[converted.index.notna()]
    converted = converted[converted.index != ""]
    if converted.index.duplicated().any():
        converted = converted.groupby(level=0).mean()
    return converted


def compute_module_eigengenes(
    expr_healthy: pd.DataFrame,
    expr_cancer: pd.DataFrame,
    module_df: pd.DataFrame | dict | list,
    min_genes: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eigengenes_healthy: dict[str, np.ndarray] = {}
    eigengenes_cancer: dict[str, np.ndarray] = {}
    module_table = normalize_module_table(module_df)

    for module in module_table["Module"].unique():
        genes = module_table.loc[module_table["Module"] == module, "Gene"].tolist()
        genes = list(set(genes) & set(expr_healthy.index) & set(expr_cancer.index))
        if len(genes) < min_genes:
            print(f"Skipping module {module}: only {len(genes)} genes")
            continue

        healthy_matrix = expr_healthy.loc[genes]
        cancer_matrix = expr_cancer.loc[genes]
        combined = pd.concat([healthy_matrix, cancer_matrix], axis=1)

        std = combined.std(axis=1)
        combined = combined.loc[std > 0]
        if combined.shape[0] < min_genes:
            print(f"Skipping module {module}: too few variable genes")
            continue

        mean = combined.mean(axis=1)
        std = combined.std(axis=1)
        combined_z = combined.subtract(mean, axis=0).divide(std, axis=0)
        pca_input = combined_z.T
        pca = PCA(n_components=1)
        pc1_all = pca.fit_transform(pca_input).flatten()

        avg_expr = pca_input.mean(axis=1)
        if np.corrcoef(pc1_all, avg_expr)[0, 1] < 0:
            pc1_all = -pc1_all

        healthy_count = healthy_matrix.shape[1]
        eigengenes_healthy[module] = pc1_all[:healthy_count]
        eigengenes_cancer[module] = pc1_all[healthy_count:]

    healthy_df = pd.DataFrame(eigengenes_healthy, index=expr_healthy.columns)
    cancer_df = pd.DataFrame(eigengenes_cancer, index=expr_cancer.columns)
    return healthy_df, cancer_df


def test_module_eigengenes(eigengenes_df: pd.DataFrame) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    modules = [column for column in eigengenes_df.columns if column != "condition"]
    for module in modules:
        healthy_vals = eigengenes_df[eigengenes_df["condition"] == "Healthy"][module]
        cancer_vals = eigengenes_df[eigengenes_df["condition"] == "Cancer"][module]
        stat, pval = mannwhitneyu(healthy_vals, cancer_vals, alternative="two-sided")
        results.append(
            {
                "Module": module,
                "Healthy_median": healthy_vals.median(),
                "Cancer_median": cancer_vals.median(),
                "Difference_median": cancer_vals.median() - healthy_vals.median(),
                "U_statistic": stat,
                "p_value": pval,
            }
        )
    return pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)


def adjust_pvalues(results_df: pd.DataFrame, method: str = "fdr_bh") -> pd.DataFrame:
    adjusted = results_df.copy()
    adjusted["p_adj"] = multipletests(adjusted["p_value"], method=method)[1]
    return adjusted


def run_eigengene_tests(eigengene_df: pd.DataFrame) -> pd.DataFrame:
    return adjust_pvalues(test_module_eigengenes(eigengene_df), method="fdr_bh").sort_values("p_adj").reset_index(drop=True)


def plot_modules(eigengenes_df: pd.DataFrame, module: str | int) -> None:
    module_name = str(module)
    if module_name not in eigengenes_df.columns and module in eigengenes_df.columns:
        module_name = module
    if module_name not in eigengenes_df.columns:
        raise KeyError(f"Module {module!r} not found")

    plot_df = eigengenes_df[["condition", module_name]].copy()
    plot_df[module_name] = pd.to_numeric(plot_df[module_name], errors="coerce")
    healthy_vals = plot_df.loc[plot_df["condition"] == "Healthy", module_name].dropna()
    cancer_vals = plot_df.loc[plot_df["condition"] == "Cancer", module_name].dropna()
    p_value = np.nan
    if len(healthy_vals) > 0 and len(cancer_vals) > 0:
        _, p_value = mannwhitneyu(healthy_vals, cancer_vals, alternative="two-sided")

    palette = {"Healthy": "#6BAED6", "Cancer": "#FB6A4A"}
    ax = sns.boxplot(data=plot_df, x="condition", y=module_name, hue="condition", palette=palette, dodge=False, legend=False)
    sns.stripplot(data=plot_df, x="condition", y=module_name, color="black", alpha=0.3, ax=ax)
    plt.title(f"Module {module_name} eigengene by condition")
    if pd.notna(p_value):
        y_max = plot_df[module_name].max()
        y_min = plot_df[module_name].min()
        y_range = y_max - y_min
        offset = y_range * 0.08 if pd.notna(y_range) and y_range > 0 else 0.1
        ax.set_ylim(y_min - offset * 0.2, y_max + offset * 1.8)
        ax.plot([0, 0, 1, 1], [y_max + offset * 0.2, y_max + offset, y_max + offset, y_max + offset * 0.2], color="black", linewidth=1)
        ax.text(0.5, y_max + offset * 1.1, f"p = {p_value:.3g}", ha="center", va="bottom")
    plt.show()


def compute_module_hubs(graph: nx.Graph, module_df: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    module_df = normalize_module_table(module_df)
    hubs: list[dict[str, Any]] = []
    for module in module_df["Module"].unique():
        genes = module_df[module_df["Module"] == module]["Gene"].tolist()
        subgraph = graph.subgraph(genes)
        degree_dict = dict(subgraph.degree())
        top_hubs = sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)[:top_n]
        hubs.append({"Module": module, "Gene": [gene for gene, _ in top_hubs], "Degree": [degree for _, degree in top_hubs]})
    return hubs


def compute_kme(expr_data: pd.DataFrame, eigengenes_df: pd.DataFrame, module_df: pd.DataFrame) -> pd.DataFrame:
    module_df = normalize_module_table(module_df)
    rows: list[dict[str, Any]] = []
    for module in module_df["Module"].unique():
        if module not in eigengenes_df.columns:
            continue
        genes = module_df[module_df["Module"] == module]["Gene"].tolist()
        eigengene = eigengenes_df[module]
        for gene in genes:
            if gene in expr_data.index:
                corr, pval = spearmanr(expr_data.loc[gene], eigengene)
                rows.append({"Gene": gene, "Module": module, "kME": corr, "kME_abs": abs(corr), "kME_p": pval})
    return pd.DataFrame(rows)


def compute_gene_rewiring(module_edge_table: pd.DataFrame, module_df: pd.DataFrame) -> pd.DataFrame:
    module_df = normalize_module_table(module_df)
    gene_scores: dict[str, dict[str, int]] = {}
    for module_id, edges in module_edge_table.groupby("Module"):
        module_genes = set(module_df[module_df["Module"] == str(module_id)]["Gene"])
        for _, row in edges.iterrows():
            for gene in [row["source"], row["target"]]:
                if gene in module_genes:
                    gene_scores.setdefault(gene, {"C": 0, "S": 0, "D": 0})
                    edge_type = row["type"]
                    if edge_type in ["C", "S", "D"]:
                        gene_scores[gene][edge_type] += 1

    rows: list[dict[str, Any]] = []
    for gene, counts in gene_scores.items():
        total = counts["C"] + counts["S"] + counts["D"]
        if total > 0:
            c_frac = counts["C"] / total
            s_frac = counts["S"] / total
            d_frac = counts["D"] / total
            homogeneity_score = max(c_frac, s_frac, d_frac)
            dominant_type = "C" if c_frac == max(c_frac, s_frac, d_frac) else "S" if s_frac == max(c_frac, s_frac, d_frac) else "D"
        else:
            c_frac = s_frac = d_frac = homogeneity_score = 0.0
            dominant_type = "N/A"
        rows.append(
            {
                "Gene": gene,
                "n_edges": total,
                "C_fraction": round(c_frac, 4),
                "S_fraction": round(s_frac, 4),
                "D_fraction": round(d_frac, 4),
                "homogeneity_score": round(homogeneity_score, 4),
                "dominant_type": dominant_type,
            }
        )
    return pd.DataFrame(rows)


def combine_hubs_and_kme_and_re(
    hubs_list: list[dict[str, Any]] | list[tuple[str, Any]],
    kme_df: pd.DataFrame,
    rewiring_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    kme_lookup = kme_df.set_index(["Gene", "Module"])[["kME", "kME_abs"]].to_dict(orient="index") if not kme_df.empty else {}

    for item in hubs_list:
        if isinstance(item, dict):
            module = item.get("Module")
            genes = item.get("Gene", [])
            degrees = item.get("Degree", [])
            hub_pairs = list(zip(genes, degrees))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            module, hub_pairs = item
        else:
            continue

        for gene, degree in hub_pairs:
            kme_info = kme_lookup.get((gene, module))
            row = {
                "Gene": gene,
                "Module": module,
                "degree": degree,
                "kME": kme_info["kME"] if kme_info else None,
                "kME_abs": kme_info["kME_abs"] if kme_info else None,
            }
            rows.append(row)

    result = pd.DataFrame(rows)
    if rewiring_df is not None and not rewiring_df.empty and not result.empty:
        result = result.merge(rewiring_df, on="Gene", how="left")
    return result


def _build_module_color_map(module_ids: list) -> dict[str, str]:
    try:
        sorted_ids = sorted(module_ids, key=lambda value: int(value))
    except (ValueError, TypeError):
        sorted_ids = sorted(module_ids, key=str)
    palette = sns.color_palette("husl", n_colors=len(sorted_ids))
    return {module: to_hex(color) for module, color in zip(sorted_ids, palette)}


def mark_modules_in_cytoscape(
    network_name: str,
    module_data: pd.DataFrame | dict | list,
    min_module_size: int = 3,
    base_size: int = 30,
    highlight_size: int = 42,
) -> pd.DataFrame:
    module_table = normalize_module_table(module_data)
    if module_table.empty:
        raise ValueError("Module table is empty")

    p4c = cytoscape_client()
    p4c.cytoscape_ping()
    p4c.set_current_network(network_name)

    node_table = p4c.get_table_columns(table="node", columns=["name"], network=network_name)
    node_names = set(node_table["name"].astype(str).str.strip())

    module_table = module_table[module_table["Gene"].isin(node_names)].copy()
    if module_table.empty:
        raise ValueError(f"No module genes matched nodes in Cytoscape network '{network_name}'")

    module_sizes = module_table.groupby("Module")["Gene"].nunique().sort_values(ascending=False)
    kept_modules = module_sizes[module_sizes >= min_module_size]
    if kept_modules.empty:
        raise ValueError(f"No modules with at least {min_module_size} genes matched nodes in '{network_name}'")

    module_table = module_table[module_table["Module"].isin(kept_modules.index)].copy()
    module_colors = _build_module_color_map(kept_modules.index.tolist())

    for module, genes in module_table.groupby("Module")["Gene"]:
        gene_list = genes.dropna().astype(str).str.strip().drop_duplicates().tolist()
        color = module_colors[module]
        p4c.set_node_color_bypass(node_names=gene_list, new_colors=[color] * len(gene_list), network=network_name)
        size_value = highlight_size if kept_modules[module] >= 10 else base_size
        p4c.set_node_size_bypass(node_names=gene_list, new_sizes=[size_value] * len(gene_list), network=network_name)

    summary = kept_modules.rename("module_size").reset_index().rename(columns={"index": "Module"})
    summary["color"] = summary["Module"].map(module_colors)
    print(f"Marked {len(summary)} modules in Cytoscape network '{network_name}'.")
    return summary


def mark_selected_modules_in_cytoscape(
    network_name: str,
    module_data: pd.DataFrame | dict | list,
    selected_modules: list[str | int] | tuple[str | int, ...] | pd.Series,
    min_module_size: int = 3,
    base_size: int = 30,
    highlight_size: int = 42,
    background_color: str = "#C9C9C9",
    background_border_color: str = "#9A9A9A",
    background_edge_color: str = "#D8D8D8",
) -> pd.DataFrame:
    module_table = normalize_module_table(module_data)
    if module_table.empty:
        raise ValueError("Module table is empty")

    requested_modules = pd.Series(list(selected_modules)).dropna().astype(str).str.strip().tolist()
    requested_modules = list(dict.fromkeys(module for module in requested_modules if module != ""))
    if not requested_modules:
        raise ValueError("selected_modules must contain at least one module ID")

    p4c = cytoscape_client()
    p4c.cytoscape_ping()
    p4c.set_current_network(network_name)

    node_table = p4c.get_table_columns(table="node", columns=["name"], network=network_name)
    node_names = node_table["name"].dropna().astype(str).str.strip().tolist()
    node_name_set = set(node_names)

    module_table = module_table[module_table["Gene"].isin(node_name_set)].copy()
    if module_table.empty:
        raise ValueError(f"No module genes matched nodes in Cytoscape network '{network_name}'")

    module_sizes = module_table.groupby("Module")["Gene"].nunique().sort_values(ascending=False)
    kept_modules = module_sizes[module_sizes >= min_module_size]
    if kept_modules.empty:
        raise ValueError(f"No modules with at least {min_module_size} genes matched nodes in '{network_name}'")

    selected_kept_modules = [module for module in requested_modules if module in kept_modules.index]
    if not selected_kept_modules:
        raise ValueError(
            f"None of the selected modules passed the size filter or matched nodes in '{network_name}'. Requested: {requested_modules}"
        )

    p4c.set_node_color_bypass(node_names=node_names, new_colors=[background_color] * len(node_names), network=network_name)
    p4c.set_node_border_color_bypass(node_names=node_names, new_colors=[background_border_color] * len(node_names), network=network_name)
    p4c.set_node_size_bypass(node_names=node_names, new_sizes=[base_size] * len(node_names), network=network_name)

    selected_table = module_table[module_table["Module"].isin(selected_kept_modules)].copy()
    module_colors = _build_module_color_map(kept_modules.index.tolist())
    matched_gene_counts = selected_table.groupby("Module")["Gene"].nunique().to_dict()

    for module, genes in selected_table.groupby("Module")["Gene"]:
        gene_list = genes.dropna().astype(str).str.strip().drop_duplicates().tolist()
        color = module_colors[module]
        p4c.set_node_color_bypass(node_names=gene_list, new_colors=[color] * len(gene_list), network=network_name)
        p4c.set_node_border_color_bypass(node_names=gene_list, new_colors=[color] * len(gene_list), network=network_name)
        p4c.set_node_size_bypass(node_names=gene_list, new_sizes=[highlight_size] * len(gene_list), network=network_name)

    selected_genes_set = set(selected_table["Gene"].dropna().astype(str).str.strip().tolist())
    edge_table = p4c.get_table_columns(table="edge", columns=["name", "source", "target"], network=network_name)
    if not edge_table.empty:
        background_edge_ids = edge_table[
            ~(edge_table["source"].isin(selected_genes_set) & edge_table["target"].isin(selected_genes_set))
        ].index.tolist()
        if background_edge_ids:
            p4c.set_edge_color_bypass(edge_names=background_edge_ids, new_colors=[background_edge_color] * len(background_edge_ids), network=network_name)

    skipped_modules = [module for module in requested_modules if module not in selected_kept_modules]
    if skipped_modules:
        print(
            "Skipped selected modules that were missing from the network or below the size threshold: "
            f"{', '.join(skipped_modules)}"
        )

    summary = pd.DataFrame({"Module": selected_kept_modules})
    summary["module_size"] = summary["Module"].map(kept_modules)
    summary["matched_genes"] = summary["Module"].map(matched_gene_counts)
    summary["color"] = summary["Module"].map(module_colors)
    print(
        f"Highlighted {len(summary)} selected modules in Cytoscape network '{network_name}' "
        f"and set the remaining nodes and edges to gray."
    )
    return summary


_CSD_EDGE_COLORS: dict[str, str] = {
    "C": "#1F6FEB",
    "S": "#1A7F37",
    "D": "#C93C37",
    "MIXED": "#6E7781",
}


def _apply_edge_color_bypasses(network: str | int, p4c: Any) -> None:
    try:
        edge_table = p4c.get_table_columns(table="edge", columns=["name", "interaction_type"], network=network)
    except Exception:
        return
    if edge_table.empty or "interaction_type" not in edge_table.columns:
        return
    for interaction_type, color in _CSD_EDGE_COLORS.items():
        mask = edge_table["interaction_type"].astype(str).str.upper() == interaction_type
        edge_ids = edge_table.index[mask].tolist()
        if edge_ids:
            p4c.set_edge_color_bypass(edge_names=edge_ids, new_colors=[color] * len(edge_ids), network=network)


def create_module_subnetworks_in_cytoscape(
    network_name: str,
    module_data: pd.DataFrame | dict | list,
    min_module_size: int = 3,
    subnetwork_prefix: str | None = None,
) -> pd.DataFrame:
    module_table = normalize_module_table(module_data)
    if module_table.empty:
        raise ValueError("Module table is empty")

    p4c = cytoscape_client()
    p4c.cytoscape_ping()
    p4c.set_current_network(network_name)

    node_table = p4c.get_table_columns(table="node", columns=["name"], network=network_name)
    node_names = set(node_table["name"].astype(str).str.strip())

    module_table = module_table[module_table["Gene"].isin(node_names)].copy()
    if module_table.empty:
        raise ValueError(f"No module genes matched nodes in Cytoscape network '{network_name}'")

    module_sizes = module_table.groupby("Module")["Gene"].nunique().sort_values(ascending=False)
    kept_modules = module_sizes[module_sizes >= min_module_size]
    if kept_modules.empty:
        raise ValueError(f"No modules with at least {min_module_size} genes matched nodes in '{network_name}'")

    prefix = subnetwork_prefix or f"{network_name}_Module"
    module_colors = _build_module_color_map(kept_modules.index.tolist())
    summaries: list[dict[str, Any]] = []

    for module_id in kept_modules.index.tolist():
        module_nodes = (
            module_table.loc[module_table["Module"] == module_id, "Gene"]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .tolist()
        )
        subnetwork_name = f"{prefix}_{module_id}"
        if subnetwork_name in p4c.get_network_list():
            p4c.delete_network(subnetwork_name)
        subnetwork = p4c.create_subnetwork(
            nodes=module_nodes,
            nodes_by_col="name",
            subnetwork_name=subnetwork_name,
            network=network_name,
        )
        p4c.set_current_network(subnetwork)
        apply_csd_style(subnetwork, f"{subnetwork_name}_style")
        color = module_colors[module_id]
        p4c.set_node_color_bypass(node_names=module_nodes, new_colors=[color] * len(module_nodes), network=subnetwork)
        _apply_edge_color_bypasses(subnetwork, p4c)
        apply_cytoscape_layout(subnetwork)
        summaries.append(
            {
                "Module": str(module_id),
                "module_size": int(kept_modules[module_id]),
                "subnetwork_name": subnetwork_name,
                "color": color,
            }
        )

    summary = pd.DataFrame(summaries).sort_values(["module_size", "Module"], ascending=[False, True]).reset_index(drop=True)
    print(f"Created {len(summary)} module subnetworks from Cytoscape network '{network_name}'.")
    return summary
