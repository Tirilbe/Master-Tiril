from __future__ import annotations

from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

try:
    from utils.network_and_gcc import cytoscape_client, extract_gcc_cytoscape, open_in_cytoscape
except ModuleNotFoundError:
    network_analysis_dir = Path(__file__).resolve().parents[2] / "4_Network_Analysis"
    if str(network_analysis_dir) not in sys.path:
        sys.path.insert(0, str(network_analysis_dir))
    from utils.network_and_gcc import cytoscape_client, extract_gcc_cytoscape, open_in_cytoscape


def _existing_cytoscape_network_names() -> set[str]:
    p4c = cytoscape_client()
    p4c.cytoscape_ping()
    network_list = p4c.get_network_list()
    if isinstance(network_list, pd.DataFrame):
        if "name" in network_list.columns:
            names = network_list["name"].tolist()
        else:
            names = network_list.iloc[:, 0].tolist()
    else:
        names = list(network_list)
    return {str(name).strip() for name in names}


def ensure_overlap_gcc_networks_in_cytoscape(
    rna_edge_input: str | Path | pd.DataFrame,
    protein_edge_input: str | Path | pd.DataFrame,
    *,
    rna_network_name: str = "RNA_CSD",
    protein_network_name: str = "Protein_CSD",
    rna_gcc_network_name: str = "RNA_CSD_GCC",
    protein_gcc_network_name: str = "Protein_CSD_GCC",
    collection: str = "CSD",
) -> pd.DataFrame:
    existing_names = _existing_cytoscape_network_names()
    actions = []

    for edge_input, network_name, gcc_network_name in [
        (rna_edge_input, rna_network_name, rna_gcc_network_name),
        (protein_edge_input, protein_network_name, protein_gcc_network_name),
    ]:
        if network_name in existing_names:
            parent_suid = pd.NA
            parent_status = "reused"
        else:
            parent_suid = open_in_cytoscape(edge_input, network_name=network_name, collection=collection)
            existing_names.add(network_name)
            parent_status = "created"
        actions.append(
            {
                "network": network_name,
                "kind": "full_network",
                "status": parent_status,
                "suid": parent_suid,
                "parent_network": pd.NA,
            }
        )

        if gcc_network_name in existing_names:
            gcc_suid = pd.NA
            gcc_status = "reused"
        else:
            gcc_suid = extract_gcc_cytoscape(network_name, gcc_network_name)
            existing_names.add(gcc_network_name)
            gcc_status = "created"
        actions.append(
            {
                "network": gcc_network_name,
                "kind": "gcc_subnetwork",
                "status": gcc_status,
                "suid": gcc_suid,
                "parent_network": network_name,
            }
        )

    return pd.DataFrame(actions)


def mark_overlap_nodes_in_cytoscape(
    network_names: list[str] | str,
    overlap_nodes: set[str] | list[str],
    color: str = "#E36209",
    size: int = 40,
) -> None:
    p4c = cytoscape_client()
    p4c.cytoscape_ping()
    if isinstance(network_names, str):
        network_names = [network_names]
    overlap_nodes = [str(node).strip() for node in overlap_nodes]
    for network_name in network_names:
        node_table = p4c.get_table_columns(table="node", columns=["name"], network=network_name)
        node_names_in_net = set(node_table["name"].astype(str).str.strip())
        present = [node for node in overlap_nodes if node in node_names_in_net]
        if not present:
            print(f"No overlap nodes found in '{network_name}'.")
            continue
        p4c.set_node_color_bypass(node_names=present, new_colors=[color] * len(present), network=network_name)
        p4c.set_node_size_bypass(node_names=present, new_sizes=[size] * len(present), network=network_name)
        p4c.set_node_label_bypass(node_names=present, new_labels=present, network=network_name)
        print(f"Marked {len(present)} overlap nodes in '{network_name}'.")

def permutation_overlap_test(set_a, set_b, universe, n_perm=10000, seed=42):
    rng = np.random.default_rng(seed)

    set_a = set(set_a)
    set_b = set(set_b)
    universe = np.array(sorted(set(universe)))

    observed = len(set_a & set_b)
    n_a = len(set_a)
    n_b = len(set_b)

    null_overlaps = np.empty(n_perm, dtype=int)

    for i in range(n_perm):
        random_a = set(rng.choice(universe, size=n_a, replace=False))
        random_b = set(rng.choice(universe, size=n_b, replace=False))
        null_overlaps[i] = len(random_a & random_b)

    # +1 correction avoids p=0
    p_value = (np.sum(null_overlaps >= observed) + 1) / (n_perm + 1)

    return {
        "observed_overlap": observed,
        "expected_overlap_mean": null_overlaps.mean(),
        "expected_overlap_sd": null_overlaps.std(ddof=1),
        "empirical_p_value": p_value,
        "null_overlaps": null_overlaps,
    }


def load_allvalues_node_universe(allvalues_path: str | Path, chunksize: int = 1_000_000) -> set[str]:
    """Load the measured gene/node universe from a large AllValues CSD file."""
    nodes: set[str] = set()
    for chunk in pd.read_csv(allvalues_path, sep="\t", usecols=["Gene1", "Gene2"], chunksize=chunksize):
        nodes.update(chunk["Gene1"].dropna().astype(str).str.strip())
        nodes.update(chunk["Gene2"].dropna().astype(str).str.strip())
    nodes.discard("")
    return nodes


def build_undirected_edge_set(edges: pd.DataFrame) -> set[tuple[str, str]]:
    """Return undirected edge tuples from a DataFrame with source/target columns."""
    required = {"source", "target"}
    if not required.issubset(edges.columns):
        raise ValueError("edges must contain 'source' and 'target' columns")

    clean_edges = edges.loc[:, ["source", "target"]].dropna().copy()
    clean_edges["source"] = clean_edges["source"].astype(str).str.strip()
    clean_edges["target"] = clean_edges["target"].astype(str).str.strip()
    clean_edges = clean_edges[(clean_edges["source"] != "") & (clean_edges["target"] != "")]
    clean_edges = clean_edges[clean_edges["source"] != clean_edges["target"]]

    return {
        tuple(sorted((row.source, row.target)))
        for row in clean_edges.itertuples(index=False)
    }


def _graph_from_edges(edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(build_undirected_edge_set(edges))
    return graph




def allvalues_threshold_node_overlap_test(
    rna_selected_edges: pd.DataFrame,
    protein_selected_edges: pd.DataFrame,
    rna_allvalues_nodes: set[str] | list[str],
    protein_allvalues_nodes: set[str] | list[str],
    *,
    n_perm: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    """Test whether selected CSD nodes overlap more than random node sets.

    The AllValues files define the measurable cross-omics gene universe. The
    null model preserves the observed number of RNA and protein CSD-selected
    nodes and samples random node sets from the shared AllValues universe.
    """
    rng = np.random.default_rng(seed)
    rna_edge_set = build_undirected_edge_set(rna_selected_edges)
    protein_edge_set = build_undirected_edge_set(protein_selected_edges)
    rna_selected_nodes = {node for edge in rna_edge_set for node in edge}
    protein_selected_nodes = {node for edge in protein_edge_set for node in edge}

    rna_universe = {str(node).strip() for node in rna_allvalues_nodes}
    protein_universe = {str(node).strip() for node in protein_allvalues_nodes}
    rna_universe.discard("")
    protein_universe.discard("")

    missing_rna = sorted(rna_selected_nodes - rna_universe)
    missing_protein = sorted(protein_selected_nodes - protein_universe)
    if missing_rna or missing_protein:
        messages = []
        if missing_rna:
            messages.append(f"RNA missing {len(missing_rna)} selected nodes, e.g. {', '.join(missing_rna[:10])}")
        if missing_protein:
            messages.append(f"Protein missing {len(missing_protein)} selected nodes, e.g. {', '.join(missing_protein[:10])}")
        raise ValueError("; ".join(messages))

    shared_universe = rna_universe & protein_universe
    rna_test_nodes = rna_selected_nodes & shared_universe
    protein_test_nodes = protein_selected_nodes & shared_universe
    shared_nodes_array = np.array(sorted(shared_universe), dtype=object)
    observed_overlap = len(rna_test_nodes & protein_test_nodes)
    null_overlaps = np.empty(n_perm, dtype=int)

    for i in range(n_perm):
        random_rna_nodes = set(rng.choice(shared_nodes_array, size=len(rna_test_nodes), replace=False))
        random_protein_nodes = set(rng.choice(shared_nodes_array, size=len(protein_test_nodes), replace=False))
        null_overlaps[i] = len(random_rna_nodes & random_protein_nodes)

    p_enrichment = (np.sum(null_overlaps >= observed_overlap) + 1) / (n_perm + 1)
    p_depletion = (np.sum(null_overlaps <= observed_overlap) + 1) / (n_perm + 1)
    p_two_sided = min(1.0, 2 * min(p_enrichment, p_depletion))
    exact_p_enrichment = float(
        hypergeom.sf(observed_overlap - 1, len(shared_universe), len(rna_test_nodes), len(protein_test_nodes))
    )
    exact_p_depletion = float(
        hypergeom.cdf(observed_overlap, len(shared_universe), len(rna_test_nodes), len(protein_test_nodes))
    )
    exact_p_two_sided = min(1.0, 2 * min(exact_p_enrichment, exact_p_depletion))

    return {
        "observed_overlap": int(observed_overlap),
        "expected_overlap_mean": float(null_overlaps.mean()),
        "expected_overlap_sd": float(null_overlaps.std(ddof=1)) if n_perm > 1 else np.nan,
        "p_enrichment": float(p_enrichment),
        "p_depletion": float(p_depletion),
        "p_two_sided": float(p_two_sided),
        "exact_p_enrichment": exact_p_enrichment,
        "exact_p_depletion": exact_p_depletion,
        "exact_p_two_sided": float(exact_p_two_sided),
        "exact_expected_overlap": float(len(rna_test_nodes) * len(protein_test_nodes) / len(shared_universe)),
        "n_permutations": int(n_perm),
        "rna_selected_edges": int(len(rna_edge_set)),
        "protein_selected_edges": int(len(protein_edge_set)),
        "rna_selected_nodes": int(len(rna_test_nodes)),
        "protein_selected_nodes": int(len(protein_test_nodes)),
        "rna_selected_nodes_outside_shared_universe": int(len(rna_selected_nodes - shared_universe)),
        "protein_selected_nodes_outside_shared_universe": int(len(protein_selected_nodes - shared_universe)),
        "rna_allvalues_nodes": int(len(rna_universe)),
        "protein_allvalues_nodes": int(len(protein_universe)),
        "shared_allvalues_nodes": int(len(shared_universe)),
        "null_overlaps": null_overlaps,
    }

import networkx as nx
import numpy as np

def edge_set(G):
    return {tuple(sorted(e)) for e in G.edges()}

import numpy as np
import pandas as pd
import networkx as nx


def degree_preserving_edge_overlap_test(
    reference_edges: pd.DataFrame,
    randomized_edges: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    allowed_nodes: set[str] | list[str] | None = None,
    n_perm: int = 10_000,
    nswap_multiplier: int = 10,
    max_tries_multiplier: int = 100,
    seed: int = 42,
) -> dict[str, object]:
    """
    Test whether edge overlap between two networks is higher/lower than expected
    under a degree-preserving null model.

    One network is held fixed. The other network is repeatedly randomized using
    double-edge swaps, preserving its node set, edge count and degree sequence.

    If allowed_nodes is provided, both networks are first restricted to edges
    where both nodes are in the shared measurable universe.
    """

    rng = np.random.default_rng(seed)

    def clean_edges(df: pd.DataFrame) -> pd.DataFrame:
        edges = df[[source_col, target_col]].copy()
        edges[source_col] = edges[source_col].astype(str).str.strip()
        edges[target_col] = edges[target_col].astype(str).str.strip()
        edges = edges[(edges[source_col] != "") & (edges[target_col] != "")]
        edges = edges[edges[source_col] != edges[target_col]]

        if allowed_nodes is not None:
            edges = edges[
                edges[source_col].isin(allowed_nodes)
                & edges[target_col].isin(allowed_nodes)
            ]

        return edges

    def build_edge_set(df: pd.DataFrame) -> set[tuple[str, str]]:
        return {
            tuple(sorted((row[source_col], row[target_col])))
            for _, row in df.iterrows()
        }

    def graph_from_edge_set(edge_set: set[tuple[str, str]]) -> nx.Graph:
        G = nx.Graph()
        G.add_edges_from(edge_set)
        return G

    if allowed_nodes is not None:
        allowed_nodes = {str(node).strip() for node in allowed_nodes}
        allowed_nodes.discard("")

    reference_edges_clean = clean_edges(reference_edges)
    randomized_edges_clean = clean_edges(randomized_edges)

    reference_edge_set = build_edge_set(reference_edges_clean)
    randomized_edge_set = build_edge_set(randomized_edges_clean)

    reference_graph = graph_from_edge_set(reference_edge_set)
    randomized_graph = graph_from_edge_set(randomized_edge_set)

    observed_overlap = len(reference_edge_set & randomized_edge_set)

    if randomized_graph.number_of_edges() < 2:
        raise ValueError("Need at least two edges in the randomized network.")

    nswap = max(1, randomized_graph.number_of_edges() * nswap_multiplier)
    max_tries = max(nswap, randomized_graph.number_of_edges() * max_tries_multiplier)

    null_overlaps = []
    failed_permutations = 0
    attempts = 0

    while len(null_overlaps) < n_perm:
        attempts += 1
        permuted_graph = randomized_graph.copy()

        try:
            nx.double_edge_swap(
                permuted_graph,
                nswap=nswap,
                max_tries=max_tries,
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            )

            permuted_edge_set = {
                tuple(sorted(edge))
                for edge in permuted_graph.edges()
            }

            null_overlaps.append(len(reference_edge_set & permuted_edge_set))

        except nx.NetworkXAlgorithmError:
            failed_permutations += 1

            if failed_permutations > n_perm:
                raise RuntimeError(
                    "Too many failed double-edge-swap attempts. "
                    "Try increasing max_tries_multiplier or lowering nswap_multiplier."
                )

    null_overlaps = np.array(null_overlaps, dtype=int)

    p_enrichment = (np.sum(null_overlaps >= observed_overlap) + 1) / (n_perm + 1)
    p_depletion = (np.sum(null_overlaps <= observed_overlap) + 1) / (n_perm + 1)
    p_two_sided = min(1.0, 2 * min(p_enrichment, p_depletion))

    return {
        "observed_overlap": int(observed_overlap),
        "expected_overlap_mean": float(null_overlaps.mean()),
        "expected_overlap_sd": float(null_overlaps.std(ddof=1)),
        "p_enrichment": float(p_enrichment),
        "p_depletion": float(p_depletion),
        "p_two_sided": float(p_two_sided),
        "n_permutations": int(n_perm),
        "failed_permutations": int(failed_permutations),
        "attempts": int(attempts),
        "nswap": int(nswap),
        "max_tries": int(max_tries),
        "reference_edges": int(reference_graph.number_of_edges()),
        "randomized_edges": int(randomized_graph.number_of_edges()),
        "reference_nodes": int(reference_graph.number_of_nodes()),
        "randomized_nodes": int(randomized_graph.number_of_nodes()),
        "background_node_count": (
            int(len(allowed_nodes)) if allowed_nodes is not None else np.nan
        ),
        "null_overlaps": null_overlaps,
    }