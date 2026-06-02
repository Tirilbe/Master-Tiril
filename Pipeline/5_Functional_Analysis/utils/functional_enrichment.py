from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

try:
    from utils.network_and_gcc import _import_optional, convert_genes_to_symbols
except ModuleNotFoundError:
    network_analysis_dir = Path(__file__).resolve().parents[2] / "4_Network_Analysis"
    if str(network_analysis_dir) not in sys.path:
        sys.path.insert(0, str(network_analysis_dir))
    from utils.network_and_gcc import _import_optional, convert_genes_to_symbols


def run_go_enrichment(
    genes: list[str] | tuple[str, ...] | pd.Series,
    background: list[str] | tuple[str, ...] | pd.Series | None = None,
    gene_sets: str = "GO_Biological_Process_2023",
    organism: str = "Human",
    outdir: str | Path | None = None,
    cutoff: float = 0.05,
) -> pd.DataFrame:
    gseapy = _import_optional("gseapy")
    gene_list = convert_genes_to_symbols(genes)
    background_list = convert_genes_to_symbols(background) if background is not None else None
    if not gene_list:
        return pd.DataFrame()

    result = gseapy.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism=organism,
        background=background_list,
        outdir=str(outdir) if outdir is not None else None,
        cutoff=cutoff,
    )
    return result.results if hasattr(result, "results") else pd.DataFrame(result)


def run_module_go_enrichment(
    module_path: str | Path,
    background_genes: list[str] | tuple[str, ...] | pd.Series,
    outdir: str | Path,
    gene_sets: str = "GO_Biological_Process_2023",
    organism: str = "Human",
    cutoff: float = 0.05,
) -> pd.DataFrame:
    module_path = Path(module_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(module_path, header=None)[0].dropna().astype(str).tolist()
    return run_go_enrichment(genes, background=background_genes, gene_sets=gene_sets, organism=organism, outdir=outdir, cutoff=cutoff)


def list_module_files(folder_path: str | Path) -> list[Path]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(
        path for path in folder.glob("module_*.txt")
        if path.is_file() and "edges" not in path.stem.lower()
    )


def run_go_for_all_modules(
    modules_folder: str | Path,
    background_genes: list[str] | tuple[str, ...] | pd.Series,
    outdir_base: str | Path,
    gene_sets: str = "GO_Biological_Process_2023",
    organism: str = "Human",
    cutoff: float = 0.05,
) -> dict[str, pd.DataFrame]:
    modules_folder = Path(modules_folder)
    outdir_base = Path(outdir_base)
    outdir_base.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for module_file in list_module_files(modules_folder):
        module_name = module_file.stem
        module_outdir = outdir_base / module_name
        print(f"Running GO enrichment for {module_name}...")
        enrichment_df = run_module_go_enrichment(
            module_file,
            background_genes=background_genes,
            outdir=module_outdir,
            gene_sets=gene_sets,
            organism=organism,
            cutoff=cutoff,
        )
        results[module_name] = enrichment_df
    return results


def go_enrichment_for_module_folder(
    module_dir: str | Path,
    gene_sets: str = "GO_Biological_Process_2023",
    output_dir: str | Path | None = None,
    pval_cutoff: float = 0.05,
    padj_cutoff: float = 0.1,
    min_genes: int = 1,
    enabled: bool = True,
    enrichment_max_retries: int = 3,
    enrichment_retry_delay: float = 3.0,
    enrichment_retry_backoff: float = 2.0,
) -> dict[str, pd.DataFrame]:
    del enrichment_max_retries, enrichment_retry_delay, enrichment_retry_backoff
    module_dir = Path(module_dir)
    output_dir = Path(output_dir) if output_dir is not None else module_dir / "GO"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not enabled:
        print("Skipping module enrichment: enabled=False")
        return {}

    results: dict[str, pd.DataFrame] = {}
    for module_file in list_module_files(module_dir):
        genes = [line.strip() for line in module_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        out_csv = output_dir / f"{module_file.stem}_GO.csv"
        if len(genes) < min_genes:
            print(f"Skipping {module_file.stem}: {len(genes)} genes < min_genes={min_genes}")
            if out_csv.exists():
                out_csv.unlink()
            continue
        try:
            enrichment_df = run_go_enrichment(
                genes,
                background=None,
                gene_sets=gene_sets,
                outdir=None,
                cutoff=pval_cutoff,
            )
            enrichment_df = enrichment_df[enrichment_df["Adjusted P-value"] <= padj_cutoff]
            results[module_file.stem] = enrichment_df
            enrichment_df.to_csv(out_csv, index=False)
        except Exception as exc:
            print(f"Enrichment failed for {module_file.name}: {exc}")
            results[module_file.stem] = pd.DataFrame()
            results[module_file.stem].to_csv(out_csv, index=False)

    print(f"Wrote GO results for {len(results)} modules to {output_dir}")
    return results


def collect_go_results(output_folder: str | Path, filename: str = "go_results.tsv") -> dict[str, pd.DataFrame]:
    output_folder = Path(output_folder)
    collected: dict[str, pd.DataFrame] = {}
    for module_dir in sorted(path for path in output_folder.iterdir() if path.is_dir()):
        candidate = module_dir / filename
        if candidate.exists():
            collected[module_dir.name] = pd.read_csv(candidate, sep="\t")
    return collected


def extract_top_go_terms(
    all_results: dict[str, pd.DataFrame],
    sort_by: str = "Adjusted P-value",
    top_n: int = 5,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for module_name, result_df in all_results.items():
        if result_df is None or result_df.empty:
            continue
        ranked = result_df.sort_values(sort_by, ascending=True).head(top_n).copy()
        ranked.insert(0, "Module", module_name)
        rows.append(ranked)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_cross_network_go_overlap(
    rna_results: dict[str, pd.DataFrame],
    protein_results: dict[str, pd.DataFrame],
    term_col: str = "Term",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _term_set(result_df: pd.DataFrame | None) -> set[str]:
        if result_df is None or result_df.empty or term_col not in result_df.columns:
            return set()
        return {
            str(term).strip()
            for term in result_df[term_col].dropna().tolist()
            if str(term).strip()
        }

    pairwise_rows: list[dict[str, Any]] = []
    for rna_module, rna_df in rna_results.items():
        rna_terms = _term_set(rna_df)
        for protein_module, protein_df in protein_results.items():
            protein_terms = _term_set(protein_df)
            shared_terms = sorted(rna_terms & protein_terms)
            union_size = len(rna_terms | protein_terms)
            pairwise_rows.append(
                {
                    "RNA_Module": rna_module,
                    "Protein_Module": protein_module,
                    "RNA_GO_Term_Count": len(rna_terms),
                    "Protein_GO_Term_Count": len(protein_terms),
                    "Shared_GO_Term_Count": len(shared_terms),
                    "Jaccard_Overlap": len(shared_terms) / union_size if union_size else 0.0,
                    "Shared_GO_Terms": "; ".join(shared_terms),
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    if pairwise_df.empty:
        empty_columns = [
            "RNA_Module",
            "Best_Protein_Module",
            "RNA_GO_Term_Count",
            "Protein_GO_Term_Count",
            "Shared_GO_Term_Count",
            "Jaccard_Overlap",
            "Shared_GO_Terms",
        ]
        return pd.DataFrame(columns=empty_columns), pairwise_df

    best_matches_df = (
        pairwise_df.sort_values(
            by=["RNA_Module", "Shared_GO_Term_Count", "Jaccard_Overlap", "Protein_Module"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates(subset="RNA_Module", keep="first")
        .rename(columns={"Protein_Module": "Best_Protein_Module"})
        .reset_index(drop=True)
    )
    return best_matches_df, pairwise_df


def plot_top_go_terms(
    top_go_df: pd.DataFrame,
    top_n_per_module: int = 3,
    figsize: tuple[int, int] = (12, 8),
    color: str = "steelblue",
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if top_go_df.empty:
        print("No GO enrichment results to plot.")
        return

    selected = top_go_df.groupby("Module", group_keys=False).head(top_n_per_module).copy()
    selected["-log10(FDR)"] = -selected["Adjusted P-value"].clip(lower=1e-300).map(lambda value: np.log10(value))
    plt.figure(figsize=figsize)
    sns.barplot(data=selected, x="-log10(FDR)", y="Term", hue="Module", orient="h", color=color)
    plt.tight_layout()
    plt.show()


def build_enrichment_graph(
    top_go_df: pd.DataFrame,
    module_col: str = "Module",
    term_col: str = "Term",
    score_col: str = "Combined Score",
):
    nx = _import_optional("networkx")
    graph = nx.Graph()
    for row in top_go_df.itertuples(index=False):
        module = getattr(row, module_col.replace(" ", "_"), None) or getattr(row, module_col)
        term = getattr(row, term_col.replace(" ", "_"), None) or getattr(row, term_col)
        score = getattr(row, score_col.replace(" ", "_"), None) or getattr(row, score_col)
        graph.add_node(module, bipartite="module")
        graph.add_node(term, bipartite="term")
        graph.add_edge(module, term, weight=score)
    return graph


def get_color_map_from_metadata(metadata_df: pd.DataFrame) -> dict[str, str]:
    if metadata_df.empty or "Module" not in metadata_df.columns:
        return {}
    if "color" in metadata_df.columns:
        return dict(zip(metadata_df["Module"].astype(str), metadata_df["color"].astype(str)))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]
    modules = metadata_df["Module"].astype(str).drop_duplicates().tolist()
    return {module: palette[index % len(palette)] for index, module in enumerate(modules)}


def visualize_enrichment_network(
    graph,
    module_colors: dict[str, str] | None = None,
    title: str = "Module-Term Enrichment Network",
    figsize: tuple[int, int] = (14, 10),
) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    module_colors = module_colors or {}
    pos = nx.spring_layout(graph, seed=42, k=0.7)
    module_nodes = [node for node, data in graph.nodes(data=True) if data.get("bipartite") == "module"]
    term_nodes = [node for node, data in graph.nodes(data=True) if data.get("bipartite") == "term"]

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(graph, pos, alpha=0.25, width=1.2)
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=module_nodes,
        node_color=[module_colors.get(node, "#4C78A8") for node in module_nodes],
        node_size=850,
        node_shape="o",
        edgecolors="black",
        linewidths=0.8,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=term_nodes,
        node_color="#F2E8CF",
        node_size=550,
        node_shape="s",
        edgecolors="#6B705C",
        linewidths=0.6,
    )
    nx.draw_networkx_labels(graph, pos, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_module_similarity_heatmap(enrichment_matrix: pd.DataFrame, title: str = "Module Similarity") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if enrichment_matrix.empty:
        print("No enrichment matrix to plot.")
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(enrichment_matrix, cmap="mako", square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def extract_top_shared_terms(top_go_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if top_go_df.empty:
        return pd.DataFrame()
    counts = top_go_df.groupby("Term")["Module"].nunique().sort_values(ascending=False)
    shared_terms = counts.head(top_n).reset_index().rename(columns={"Module": "module_count"})
    return shared_terms


def extract_top_terms_from_all_modules(
    all_results: dict[str, pd.DataFrame],
    n_terms_per_module: int = 3,
) -> pd.DataFrame:
    return extract_top_go_terms(all_results, top_n=n_terms_per_module)


def run_all_modules_enrichment(
    module_folder: str | Path,
    all_genes: list[str] | tuple[str, ...] | pd.Series,
    output_base: str | Path,
    library: str = "GO_Biological_Process_2023",
) -> dict[str, pd.DataFrame]:
    return run_go_for_all_modules(
        modules_folder=module_folder,
        background_genes=all_genes,
        outdir_base=output_base,
        gene_sets=library,
    )
