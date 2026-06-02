from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
import importlib
import inspect
import json
import os

import networkx as nx
import numpy as np
import pandas as pd


def _import_optional(module_name: str, package_name: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        package_hint = package_name or module_name
        raise ImportError(
            f"Optional dependency '{module_name}' is required for this step. "
            f"Install '{package_hint}' in the active environment before rerunning."
        ) from exc


@dataclass(slots=True)
class AnalysisContext:
    project_root: Path
    notebook_dir: Path
    notebook_name: str
    results_root: Path
    output_dir: Path
    network_collection: str = "counts"
    run_cytoscape: bool = False
    run_enrichment: bool = False
    save_figures: bool = True
    network_files: dict[str, Path] = field(default_factory=dict)
    expression_files: dict[str, Path] = field(default_factory=dict)
    expression_inputs: dict[str, Path] = field(default_factory=dict)
    clinical_files: dict[str, Path] = field(default_factory=dict)

    def output_path(self, *parts: str) -> Path:
        target_path = self.output_dir.joinpath(*parts)
        if target_path.exists():
            return target_path

        notebook_alias_path = _resolve_notebook_folder_alias(self.results_root, self.notebook_name, parts)
        if notebook_alias_path is not None and notebook_alias_path.exists():
            return notebook_alias_path

        compatibility_path = _resolve_module_folder_alias(self.output_dir, parts)
        if compatibility_path is not None and compatibility_path.exists():
            return compatibility_path

        return target_path

    def legacy_notebook_output_dir(self) -> Path:
        return self.results_root / self.notebook_dir.name

    def notebook_output_path(self, notebook_name: str, *parts: str) -> Path:
        target_path = (self.results_root / Path(notebook_name).stem).joinpath(*parts)
        if target_path.exists():
            return target_path

        alias_path = _resolve_notebook_folder_alias(self.results_root, notebook_name, parts)
        if alias_path is not None and alias_path.exists():
            return alias_path

        legacy_path = self.legacy_notebook_output_dir().joinpath(*parts)
        if legacy_path.exists():
            return legacy_path

        return target_path


def _normalize_notebook_name(notebook_name: str | Path | None, notebook_dir: Path) -> str:
    if notebook_name is not None:
        return Path(notebook_name).stem
    inferred_name = _infer_active_notebook_name()
    if inferred_name is not None:
        return inferred_name
    if notebook_dir.suffix:
        return notebook_dir.stem
    return notebook_dir.name


def _infer_active_notebook_name() -> str | None:
    env_candidates = [
        os.environ.get("VSCODE_NOTEBOOK_FILE"),
        os.environ.get("VSCODE_IPYNB_FILE"),
        os.environ.get("JPY_SESSION_NAME"),
    ]
    for candidate in env_candidates:
        if candidate:
            return Path(candidate).stem

    user_ns_candidates = [
        "__vsc_ipynb_file__",
        "__notebook_file__",
        "notebook_path",
        "notebook_name",
    ]

    try:
        from IPython import get_ipython

        shell = get_ipython()
    except Exception:
        shell = None

    if shell is not None:
        user_ns = getattr(shell, "user_ns", {}) or {}
        for key in user_ns_candidates:
            candidate = user_ns.get(key)
            if isinstance(candidate, str) and candidate:
                return Path(candidate).stem

    for frame_info in inspect.stack():
        frame_globals = frame_info.frame.f_globals
        for key in user_ns_candidates:
            candidate = frame_globals.get(key)
            if isinstance(candidate, str) and candidate:
                return Path(candidate).stem

    return None


def _resolve_module_folder_alias(base_dir: Path, parts: tuple[str, ...]) -> Path | None:
    if not parts:
        return None

    first, *remaining = parts
    aliases = {
        "RNA_Leiden_Modules": "RNA_Leiden_Modules_0.7",
        "Protein_Leiden_Modules": "Protein_Leiden_Modules_0.5",
        "Protein_Leiden_Modules_0.7": "Protein_Leiden_Modules_0.5",
    }
    alias = aliases.get(first)
    if alias is None:
        return None
    return base_dir.joinpath(alias, *remaining)


def _resolve_notebook_folder_alias(results_root: Path, notebook_name: str | Path, parts: tuple[str, ...]) -> Path | None:
    notebook_stem = Path(notebook_name).stem
    aliases = {
        "network_and_gcc": ["01_network_and_gcc"],
        "01_network_and_gcc": ["network_and_gcc"],
        "topology_and_hubs": ["02_topology_and_hubs"],
        "02_topology_and_hubs": ["topology_and_hubs"],
        "modules": ["03_modules", "03_modules_and_eigengenes", "03b_module_eigengenes"],
        "03_modules": ["modules", "03_modules_and_eigengenes", "03b_module_eigengenes"],
        "functional_enrichment": ["04_functional_enrichment"],
        "04_functional_enrichment": ["functional_enrichment"],
        "cross_modal_overlap": ["05_cross_modal_overlap"],
        "05_cross_modal_overlap": ["cross_modal_overlap"],
        "survival_analysis": ["01_survival_analysis"],
        "01_survival_analysis": ["survival_analysis"],
        "03_modules_and_eigengenes": ["03_modules"],
        "03b_module_eigengenes": ["03_modules", "modules"],
    }
    for alias in aliases.get(notebook_stem, []):
        alias_path = (results_root / alias).joinpath(*parts)
        if alias_path.exists():
            return alias_path
    return None


def _resolve_preferred_processed_file(
    project_root: Path,
    preferred_name: str,
    legacy_name: str,
    subdir: str | None = None,
) -> Path:
    processed_root = project_root / "data" / "processed"
    if subdir:
        preferred_path = processed_root / subdir / preferred_name
    else:
        preferred_path = processed_root / preferred_name
    if preferred_path.exists():
        return preferred_path
    return processed_root / legacy_name


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "environment.yml").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root by searching for environment.yml")


def _build_network_files(project_root: Path, network_collection: str) -> dict[str, Path]:
    network_roots = {
        "counts": {
            "rna_csd": "CSDSelection_rna_WTO_1_20.txt",
            "prot_csd": "CSDSelection_proteome_WTO_1_20.txt",
            "rna_c": "CNetwork_rna_WTO_1_20.txt",
            "prot_c": "CNetwork_proteome_WTO_1_20.txt",
            "rna_s": "SNetwork_rna_WTO_1_20.txt",
            "prot_s": "SNetwork_proteome_WTO_1_20.txt",
            "rna_d": "DNetwork_rna_WTO_1_20.txt",
            "prot_d": "DNetwork_proteome_WTO_1_20.txt",
        },
    }

    if network_collection not in network_roots:
        valid_collections = ", ".join(sorted(network_roots))
        raise ValueError(
            f"Unknown network_collection '{network_collection}'. Expected one of: {valid_collections}."
        )

    return {
        key: project_root / "results" / "CSD" / filename
        for key, filename in network_roots[network_collection].items()
    }


def build_analysis_context(
    notebook_dir: str | Path | None = None,
    notebook_name: str | Path | None = None,
    network_collection: str = "counts",
    run_cytoscape: bool = False,
    run_enrichment: bool = False,
    save_figures: bool = True,
) -> AnalysisContext:
    resolved_notebook_dir = Path(notebook_dir or Path.cwd()).resolve()
    project_root = find_project_root(resolved_notebook_dir)
    normalized_notebook_name = _normalize_notebook_name(notebook_name, resolved_notebook_dir)
    results_root = project_root / "results" / "notebooks"
    output_dir = results_root / normalized_notebook_name
    output_dir.mkdir(parents=True, exist_ok=True)

    expression_files = {
        "rna_healthy": _resolve_preferred_processed_file(
            project_root,
            preferred_name="rna_normal_FINAL_FINAL.txt",
            legacy_name="rna_normal_FINAL_FINAL.txt",
            subdir="transcriptome",
        ),
        "rna_cancer": _resolve_preferred_processed_file(
            project_root,
            preferred_name="rna_cancer_FINAL_FINAL.txt",
            legacy_name="rna_cancer_FINAL_FINAL.txt",
            subdir="transcriptome",
        ),
        "prot_healthy": project_root / "data" / "processed" / "proteome" / "proteome_healthy_FINAL_FINAL_with_header.txt",
        "prot_cancer": project_root / "data" / "processed" / "proteome" / "proteome_cancer_FINAL_FINAL_with_header.txt",
    }

    return AnalysisContext(
        project_root=project_root,
        notebook_dir=resolved_notebook_dir,
        notebook_name=normalized_notebook_name,
        results_root=results_root,
        output_dir=output_dir,
        network_collection=network_collection,
        run_cytoscape=run_cytoscape,
        run_enrichment=run_enrichment,
        save_figures=save_figures,
        network_files=_build_network_files(project_root, network_collection),
        expression_files=expression_files,
        expression_inputs=expression_files,
        clinical_files={
            "tcga": project_root / "data" / "clinical" / "clinical_TCGA.tsv",
            "cptac": project_root / "data" / "clinical" / "clinical_CPTAC.tsv",
            "tcga_slide": project_root / "data" / "clinical" / "slide.tsv",
        },
    )


def configured_inputs_table(ctx: AnalysisContext) -> pd.DataFrame:
    rows = [
        ("RNA CSD network", ctx.network_files["rna_csd"]),
        ("Protein CSD network", ctx.network_files["prot_csd"]),
        ("RNA C network", ctx.network_files["rna_c"]),
        ("Protein C network", ctx.network_files["prot_c"]),
        ("RNA S network", ctx.network_files["rna_s"]),
        ("Protein S network", ctx.network_files["prot_s"]),
        ("RNA D network", ctx.network_files["rna_d"]),
        ("Protein D network", ctx.network_files["prot_d"]),
        ("RNA healthy expression", ctx.expression_files["rna_healthy"]),
        ("RNA cancer expression", ctx.expression_files["rna_cancer"]),
        ("Protein healthy expression", ctx.expression_files["prot_healthy"]),
        ("Protein cancer expression", ctx.expression_files["prot_cancer"]),
        ("CPTAC clinical", ctx.clinical_files["cptac"]),
    ]
    table = pd.DataFrame(rows, columns=["input", "path"])
    table["path"] = table["path"].map(lambda value: str(Path(value)))
    table["exists"] = table["path"].map(lambda value: Path(value).exists())
    return table


def validate_inputs(ctx: AnalysisContext) -> list[str]:
    table = configured_inputs_table(ctx)
    missing = table.loc[~table["exists"], "path"].tolist()
    if missing:
        print("Missing input files:")
        for path in missing:
            print(f"  - {path}")
    else:
        print("All configured input files were found.")
    return missing


@lru_cache(maxsize=32)
def _convert_ensembl_to_symbol_cached(ensembl_ids: tuple[str, ...]) -> dict[str, str | None]:
    mygene = _import_optional("mygene")
    mg = mygene.MyGeneInfo()
    query_result = mg.querymany(
        list(ensembl_ids),
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
    )

    id_to_symbol: dict[str, str | None] = {}
    for item in query_result:
        if item.get("notfound"):
            id_to_symbol[item["query"]] = None
        else:
            id_to_symbol[item["query"]] = item.get("symbol")
    return id_to_symbol


def convert_ensembl_to_symbol(ensembl_ids: list[str] | tuple[str, ...] | pd.Series) -> dict[str, str | None]:
    cleaned = pd.Series(list(ensembl_ids)).dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""].drop_duplicates().tolist()
    if not cleaned:
        return {}
    return _convert_ensembl_to_symbol_cached(tuple(sorted(cleaned)))


def convert_genes_to_symbols(genes: list[str] | tuple[str, ...] | pd.Series) -> list[str]:
    cleaned = pd.Series(list(genes)).dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""].drop_duplicates().tolist()
    if not cleaned:
        return []

    ensembl_ids = [gene for gene in cleaned if gene.startswith("ENSG")]
    if not ensembl_ids:
        return cleaned

    id_to_symbol = convert_ensembl_to_symbol(ensembl_ids)
    converted = [id_to_symbol.get(gene) or gene if gene.startswith("ENSG") else gene for gene in cleaned]
    return pd.Series(converted).dropna().astype(str).str.strip().drop_duplicates().tolist()


def load_edges_clean(edge_input: str | Path | pd.DataFrame, convert_symbols: bool = False) -> pd.DataFrame:
    if isinstance(edge_input, (str, Path)):
        edges = pd.read_csv(edge_input, sep="\t", header=None, names=["source", "target", "value", "type"])
    elif isinstance(edge_input, pd.DataFrame):
        edges = edge_input.copy()
    else:
        raise TypeError(f"edge_input must be a file path or DataFrame, got {type(edge_input)!r}")

    required = {"source", "target"}
    if not required.issubset(edges.columns):
        if edges.shape[1] < 2:
            raise ValueError("Edge DataFrame must contain at least source and target columns")
        colmap = {edges.columns[0]: "source", edges.columns[1]: "target"}
        if edges.shape[1] >= 3:
            colmap[edges.columns[2]] = "value"
        if edges.shape[1] >= 4:
            colmap[edges.columns[3]] = "type"
        edges = edges.rename(columns=colmap)

    if "value" not in edges.columns:
        edges["value"] = np.nan
    if "type" not in edges.columns:
        edges["type"] = ""

    edges["value"] = pd.to_numeric(edges["value"], errors="coerce")
    edges[["source", "target"]] = edges[["source", "target"]].astype(str)
    edges["source"] = edges["source"].str.strip()
    edges["target"] = edges["target"].str.strip()
    edges = edges[(edges["source"] != "") & (edges["target"] != "")].copy()

    header_mask = edges["source"].str.lower().eq("source") & edges["target"].str.lower().eq("target")
    if header_mask.any():
        edges = edges.loc[~header_mask].copy()

    if convert_symbols:
        all_ids = pd.concat([edges["source"], edges["target"]]).drop_duplicates().tolist()
        ensembl_ids = [gene for gene in all_ids if gene.startswith("ENSG")]
        if ensembl_ids:
            id_to_symbol = convert_ensembl_to_symbol(ensembl_ids)
            edges["source"] = edges["source"].map(lambda value: id_to_symbol.get(value) or value)
            edges["target"] = edges["target"].map(lambda value: id_to_symbol.get(value) or value)

    return edges.reset_index(drop=True)


def network_stats(edge_input: str | Path | pd.DataFrame, network_name: str | None = None) -> dict[str, float | int | str]:
    edges_df = load_edges_clean(edge_input)
    edges_df["weight"] = edges_df["value"].where(edges_df["value"].notna(), 1.0)
    graph = nx.from_pandas_edgelist(edges_df, source="source", target="target", edge_attr=[column for column in ["value", "weight", "type"] if column in edges_df.columns])
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    component_count = nx.number_connected_components(graph)
    largest_component = max(nx.connected_components(graph), key=len)
    gcc_size = len(largest_component)
    density = nx.density(graph)
    degree_values = list(dict(graph.degree()).values())

    if network_name is None:
        network_name = Path(edge_input).stem if isinstance(edge_input, (str, Path)) else "Graph"

    type_col = edges_df.get("type")
    if type_col is not None and type_col.astype(str).str.strip().ne("").any():
        type_counts = type_col.astype(str).str.upper().value_counts()
        total = len(edges_df)
        frac_c = type_counts.get("C", 0) / total
        frac_s = type_counts.get("S", 0) / total
        frac_d = type_counts.get("D", 0) / total
    else:
        frac_c = frac_s = frac_d = np.nan

    result = {
        "network": network_name,
        "n_nodes": node_count,
        "n_edges": edge_count,
        "n_components": component_count,
        "gcc_size": gcc_size,
        "gcc_fraction": gcc_size / node_count if node_count else np.nan,
        "density": density,
        "avg_degree": float(np.mean(degree_values)) if degree_values else np.nan,
        "median_degree": float(np.median(degree_values)) if degree_values else np.nan,
        "frac_c": frac_c,
        "frac_s": frac_s,
        "frac_d": frac_d,
    }

    print("-" * 40)
    for key, value in result.items():
        print(f"{key}: {value}")
    return result


def extract_gcc_file(edge_file: str | Path | pd.DataFrame) -> tuple[nx.Graph, pd.DataFrame]:
    edges = load_edges_clean(edge_file)
    if edges.empty:
        raise ValueError(f"No edges loaded from {edge_file}")

    graph = nx.from_pandas_edgelist(
        edges,
        source="source",
        target="target",
        edge_attr=[column for column in edges.columns if column not in ["source", "target"]],
    )
    if graph.number_of_nodes() == 0:
        raise ValueError("The graph is empty. Cannot extract GCC.")

    gcc_nodes = max(nx.connected_components(graph), key=len)
    gcc_set = set(gcc_nodes)
    graph_gcc = graph.subgraph(gcc_set).copy()
    gcc_edges = edges[edges["source"].isin(gcc_set) & edges["target"].isin(gcc_set)].copy()
    return graph_gcc, gcc_edges


def ensure_gcc_output(
    edge_file: str | Path,
    output_file: str | Path,
    *,
    convert_symbols: bool = True,
) -> Path:
    edge_path = Path(edge_file)
    output_path = Path(output_file)
    metadata_path = output_path.with_name(f"{output_path.name}.source.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    refresh_output = not output_path.exists()
    expected_source = str(edge_path.resolve())

    if not refresh_output:
        if not metadata_path.exists():
            refresh_output = True
        else:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                refresh_output = True
            else:
                refresh_output = metadata.get("source_path") != expected_source

    if not refresh_output:
        refresh_output = output_path.stat().st_mtime < edge_path.stat().st_mtime

    if refresh_output:
        _, gcc_edges = extract_gcc_file(edge_path)
        cleaned_edges = load_edges_clean(gcc_edges, convert_symbols=convert_symbols)
        cleaned_edges.to_csv(output_path, sep="\t", index=False)
        metadata_path.write_text(
            json.dumps({"source_path": expected_source}, indent=2),
            encoding="utf-8",
        )
        print(f"Refreshed GCC export: {output_path}")
    else:
        print(f"Using cached GCC export: {output_path}")

    return output_path


def _canonical_edge_pairs(edges: pd.DataFrame) -> pd.Series:
    return edges.apply(
        lambda row: tuple(sorted((str(row["source"]), str(row["target"])))),
        axis=1,
    )


def _edge_type_sets(edges: pd.DataFrame) -> dict[str, set[tuple[str, str]]]:
    typed_edges = edges.copy()
    typed_edges["edge_pair"] = _canonical_edge_pairs(typed_edges)
    return {
        edge_type: set(typed_edges.loc[typed_edges["type"] == edge_type, "edge_pair"])
        for edge_type in ["C", "S", "D"]
    }


def _gene_type_sets(edges: pd.DataFrame) -> dict[str, set[str]]:
    type_sets: dict[str, set[str]] = {}
    for edge_type in ["C", "S", "D"]:
        typed_edges = edges.loc[edges["type"] == edge_type, ["source", "target"]]
        genes = set(typed_edges["source"].astype(str)).union(typed_edges["target"].astype(str))
        type_sets[edge_type] = genes
    return type_sets


def _venn_region_counts(type_sets: dict[str, set[Any]], prefix: str) -> dict[str, int]:
    c_set, s_set, d_set = type_sets["C"], type_sets["S"], type_sets["D"]
    return {
        f"C_only_{prefix}": len(c_set - s_set - d_set),
        f"S_only_{prefix}": len(s_set - c_set - d_set),
        f"D_only_{prefix}": len(d_set - c_set - s_set),
        f"C_and_S_{prefix}": len((c_set & s_set) - d_set),
        f"C_and_D_{prefix}": len((c_set & d_set) - s_set),
        f"S_and_D_{prefix}": len((s_set & d_set) - c_set),
        f"C_and_S_and_D_{prefix}": len(c_set & s_set & d_set),
    }


def _node_type_summary(edges: pd.DataFrame, network_name: str) -> dict[str, float | int | str]:
    node_to_types: dict[str, set[str]] = {}
    for row in edges[["source", "target", "type"]].itertuples(index=False):
        for node in [str(row.source), str(row.target)]:
            node_to_types.setdefault(node, set()).add(str(row.type))

    combo_counts: dict[str, int] = {}
    for type_set in node_to_types.values():
        combo = "+".join(sorted(type_set))
        combo_counts[combo] = combo_counts.get(combo, 0) + 1

    combo_table = pd.DataFrame(
        {
            "edge_type_combo": list(combo_counts.keys()),
            "n_nodes": list(combo_counts.values()),
        }
    )
    if combo_table.empty:
        pure_nodes = 0
        mixed_nodes = 0
    else:
        combo_table["n_edge_types"] = combo_table["edge_type_combo"].str.count("\\+") + 1
        combo_table["node_class"] = combo_table["n_edge_types"].map(lambda n: "pure" if n == 1 else "mixed")
        pure_nodes = int(combo_table.loc[combo_table["node_class"] == "pure", "n_nodes"].sum())
        mixed_nodes = int(combo_table.loc[combo_table["node_class"] == "mixed", "n_nodes"].sum())

    total_nodes = len(node_to_types)
    return {
        "network": network_name,
        "n_nodes": total_nodes,
        "pure_edge_type_nodes": pure_nodes,
        "mixed_edge_type_nodes": mixed_nodes,
        "pure_edge_type_fraction": pure_nodes / total_nodes if total_nodes else 0,
        "mixed_edge_type_fraction": mixed_nodes / total_nodes if total_nodes else 0,
    }


def _draw_type_venn_on_axis(ax: Any, type_sets: dict[str, set[Any]], item_label: str = "genes") -> dict[str, int]:
    plt = _import_optional("matplotlib.pyplot", "matplotlib")
    circle_cls = _import_optional("matplotlib.patches", "matplotlib").Circle

    counts = _venn_region_counts(type_sets, item_label)
    colors = {"C": "#4C78A8", "S": "#54A24B", "D": "#E45756"}

    circles = {
        "C": circle_cls((0.34, 0.59), 0.23, color=colors["C"], alpha=0.35),
        "S": circle_cls((0.66, 0.59), 0.23, color=colors["S"], alpha=0.35),
        "D": circle_cls((0.50, 0.32), 0.23, color=colors["D"], alpha=0.35),
    }
    for circle in circles.values():
        ax.add_patch(circle)

    label_positions = {
        f"C_only_{item_label}": (0.28, 0.63),
        f"S_only_{item_label}": (0.72, 0.63),
        f"D_only_{item_label}": (0.50, 0.21),
        f"C_and_S_{item_label}": (0.50, 0.66),
        f"C_and_D_{item_label}": (0.41, 0.43),
        f"S_and_D_{item_label}": (0.59, 0.43),
        f"C_and_S_and_D_{item_label}": (0.50, 0.49),
    }
    for key, (x_pos, y_pos) in label_positions.items():
        ax.text(
            x_pos,
            y_pos,
            f"{counts[key]:,}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#1A1A1A",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    return counts


def global_network_homogeneity(ctx: AnalysisContext) -> dict[str, Any]:
    plt = _import_optional("matplotlib.pyplot", "matplotlib")

    homogeneity_jobs = [
        ("RNA", ctx.output_path("RNA_CSD_GCC.txt")),
        ("Protein", ctx.output_path("Protein_CSD_GCC.txt")),
    ]

    gene_count_rows: list[dict[str, Any]] = []
    edge_count_rows: list[dict[str, Any]] = []
    node_summary_rows: list[dict[str, Any]] = []
    gene_sets_by_network: dict[str, dict[str, set[str]]] = {}

    for label, edge_path in homogeneity_jobs:
        edges = pd.read_csv(edge_path, sep="\t")

        gene_sets = _gene_type_sets(edges)
        gene_sets_by_network[label] = gene_sets
        gene_count_rows.append({"network": label, **_venn_region_counts(gene_sets, "genes")})

        edge_sets = _edge_type_sets(edges)
        edge_count_rows.append({"network": label, **_venn_region_counts(edge_sets, "edges")})

        node_summary_rows.append(_node_type_summary(edges, label))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))
    for ax, label, panel_label in zip(axes, ["RNA", "Protein"], ["a)", "b)"]):
        _draw_type_venn_on_axis(ax, gene_sets_by_network[label], item_label="genes")
        ax.text(0.02, 0.90, panel_label, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="left", va="top")
    fig.tight_layout(w_pad=1.0)
    if ctx.save_figures:
        fig.savefig(ctx.output_path("RNA_Protein_CSD_GCC_gene_type_venn_side_by_side.png"), dpi=300, bbox_inches="tight")

    gene_homogeneity = pd.DataFrame(gene_count_rows)
    edge_homogeneity = pd.DataFrame(edge_count_rows)
    node_homogeneity = pd.DataFrame(node_summary_rows)

    gene_homogeneity.to_csv(ctx.output_path("gene_type_venn_counts.csv"), index=False)
    edge_homogeneity.to_csv(ctx.output_path("edge_type_venn_counts.csv"), index=False)
    node_homogeneity.to_csv(ctx.output_path("node_edge_type_homogeneity_summary.csv"), index=False)

    return {
        "figure": fig,
        "gene_homogeneity": gene_homogeneity,
        "edge_homogeneity": edge_homogeneity,
        "node_homogeneity": node_homogeneity,
    }


def cytoscape_client():
    return _import_optional("py4cytoscape", "py4cytoscape")


def best_cytoscape_layout() -> str:
    p4c = cytoscape_client()
    available_layouts = p4c.get_layout_names()
    lookup = {name.lower(): name for name in available_layouts}
    preferred = ["force-directed", "prefuse force directed"]
    for candidate in preferred:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError("No supported force-directed Cytoscape layout is available. Expected one of: force-directed, prefuse force directed.")


def apply_cytoscape_layout(network: str | int) -> str:
    p4c = cytoscape_client()
    layout_name = best_cytoscape_layout()
    p4c.layout_network(layout_name=layout_name, network=network)
    p4c.fit_content(network=network)
    print(f"Applied Cytoscape layout '{layout_name}' to {network}.")
    return layout_name


def apply_csd_style(network_name: str | int, style_name: str = "CSD_style") -> None:
    p4c = cytoscape_client()
    if style_name in p4c.get_visual_style_names():
        p4c.delete_visual_style(style_name)
    p4c.create_visual_style(style_name)
    p4c.set_visual_style(style_name, network=network_name)
    p4c.set_visual_property_default({"visualProperty": "NETWORK_BACKGROUND_PAINT", "value": "#F7F7F5"}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "NODE_SHAPE", "value": "ELLIPSE"}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "NODE_FILL_COLOR", "value": "#E8E4DB"}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "NODE_BORDER_WIDTH", "value": 1.8}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "NODE_BORDER_PAINT", "value": "#4A403A"}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "NODE_SIZE", "value": 30}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "NODE_LABEL_COLOR", "value": "#2F2A26"}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "NODE_LABEL_FONT_SIZE", "value": 18}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "NODE_TRANSPARENCY", "value": 235}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "EDGE_WIDTH", "value": 2.8}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "EDGE_TRANSPARENCY", "value": 190}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "EDGE_CURVED", "value": True}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "EDGE_STROKE_UNSELECTED_PAINT", "value": "#7F8C8D"}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "EDGE_TARGET_ARROW_SHAPE", "value": "NONE"}, style_name=style_name)
    p4c.set_visual_property_default({"visualProperty": "EDGE_SOURCE_ARROW_SHAPE", "value": "NONE"}, style_name=style_name)
    p4c.set_edge_color_mapping(
        table_column="interaction_type",
        table_column_values=["C", "S", "D", "MIXED"],
        colors=["#1F6FEB", "#1A7F37", "#C93C37", "#6E7781"],
        mapping_type="d",
        style_name=style_name,
        network=network_name,
    )


def open_in_cytoscape(edge_file: str | Path | pd.DataFrame, network_name: str, collection: str = "CSD") -> Any:
    p4c = cytoscape_client()
    p4c.cytoscape_ping()
    edges = load_edges_clean(edge_file, convert_symbols=True)
    required_columns = {"source", "target", "value", "type"}
    if not required_columns.issubset(edges.columns):
        raise ValueError(f"Edge file must contain columns: {required_columns}")
    if "interaction_type" not in edges.columns:
        edges = edges.rename(columns={"type": "interaction_type"})
    edges["interaction_type"] = edges["interaction_type"].astype(str).str.strip().str.upper()
    edges = edges[edges["interaction_type"].isin(["C", "S", "D"])].copy()
    if edges.empty:
        raise ValueError("No edges left after filtering for interaction_type in {'C','S','D'}")

    net_suid = p4c.create_network_from_data_frames(edges=edges, title=network_name, collection=collection)
    p4c.set_current_network(net_suid)
    apply_csd_style(net_suid, f"{network_name}_style")
    apply_cytoscape_layout(net_suid)
    return net_suid


def extract_gcc_cytoscape(network: str, subnetwork_name: str | None = None) -> Any:
    p4c = cytoscape_client()
    p4c.cytoscape_ping()
    graph = p4c.create_networkx_from_network(network=network)
    if graph.number_of_nodes() == 0:
        raise ValueError("The network is empty. Cannot extract GCC.")
    gcc_nodes = max(nx.weakly_connected_components(graph), key=len) if graph.is_directed() else max(nx.connected_components(graph), key=len)
    subnetwork_name = subnetwork_name or f"{network}_GCC"
    gcc_suid = p4c.create_subnetwork(nodes=list(gcc_nodes), nodes_by_col="name", subnetwork_name=subnetwork_name, network=network)
    p4c.set_current_network(gcc_suid)
    apply_csd_style(gcc_suid, f"{subnetwork_name}_style")
    apply_cytoscape_layout(gcc_suid)
    return gcc_suid
