from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from utils.network_and_gcc import convert_ensembl_to_symbol, cytoscape_client, load_edges_clean


def _compute_degree_distribution_stats(graph: nx.Graph) -> dict[str, Any]:
    degrees = [degree for _, degree in graph.degree()]
    degree_counts = np.bincount(degrees)
    ks = np.nonzero(degree_counts)[0]
    counts = degree_counts[ks]
    log_k = np.log10(ks)
    log_c = np.log10(counts)
    coeffs = np.polyfit(log_k, log_c, 1)
    gamma = -coeffs[0]
    fit_line = np.poly1d(coeffs)
    r2 = 1 - np.sum((log_c - fit_line(log_k)) ** 2) / np.sum((log_c - np.mean(log_c)) ** 2)
    return {
        "degrees": degrees,
        "degree_counts": degree_counts,
        "ks": ks,
        "counts": counts,
        "fit_line": fit_line,
        "gamma": gamma,
        "r2": r2,
    }


def load_graph(edge_input: str | Path | pd.DataFrame) -> nx.Graph:
    edges = load_edges_clean(edge_input, convert_symbols=True)
    edges["weight"] = edges["value"].where(edges["value"].notna(), 1.0)
    edge_attrs = [column for column in ["value", "weight", "type"] if column in edges.columns]
    return nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr=edge_attrs)


def plot_degree_distribution(
    graph: nx.Graph,
    network_name: str,
    color: str = "#4C72B0",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> dict[str, Any]:
    stats = _compute_degree_distribution_stats(graph)
    ks = stats["ks"]
    counts = stats["counts"]
    fit_line = stats["fit_line"]
    gamma = stats["gamma"]
    r2 = stats["r2"]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        created_fig = True
    else:
        fig = ax.figure

    ax.scatter(
        ks,
        counts,
        color=color,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        s=40,
    )
    k_range = np.logspace(np.log10(ks.min()), np.log10(ks.max()), 200)
    ax.plot(
        k_range,
        10 ** fit_line(np.log10(k_range)),
        "r--",
        linewidth=1.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("Count $P(k)$")
    ax.set_title(f"Degree Distribution - {network_name}")
    ax.set_ylim(bottom=1)
    ax.text(
        0.98,
        0.96,
        rf"$P(k) \propto k^{{-\gamma}}$" + "\n" + rf"$\gamma={gamma:.2f}$" + "\n" + rf"$R^2={r2:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    sns.despine(ax=ax)
    fig.tight_layout()

    if save_path is not None and created_fig:
        fig.savefig(save_path, dpi=150)
    if show and created_fig:
        plt.show()

    print(f"Power-law exponent gamma = {gamma:.3f}, R^2 = {r2:.3f}")
    return {
        "degrees": stats["degrees"],
        "degree_counts": stats["degree_counts"],
        "gamma": gamma,
        "r2": r2,
    }


def plot_combined_degree_distribution(
    graphs: list[tuple[nx.Graph, str, str]],
    title: str = "Degree Distribution Comparison",
    save_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    fig, axes = plt.subplots(1, len(graphs), figsize=(7 * len(graphs), 5), squeeze=False)
    results: dict[str, dict[str, Any]] = {}
    axes_flat = axes.ravel()

    for ax, (graph, network_name, color) in zip(axes_flat, graphs):
        results[network_name] = plot_degree_distribution(
            graph,
            network_name,
            color=color,
            ax=ax,
            show=False,
        )

    fig.suptitle(title)
    for ax in axes_flat[len(graphs) :]:
        ax.remove()
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return results


def render_ranked_html_table(
    df: pd.DataFrame,
    title: str,
    label: str,
    formatters: dict[str, Any] | None = None,
    accent_color: str = "#A64B00",
) -> pd.DataFrame:
    from IPython.display import HTML, display

    if df.empty:
        print(f"No {label.lower()} found.")
        return df

    pretty_df = df.copy()
    if formatters is not None:
        for column, formatter in formatters.items():
            if column in pretty_df.columns:
                pretty_df[column] = pretty_df[column].map(formatter)

        table_html = pretty_df.to_html(index=False, border=0, justify="center", classes="ranked-table")
        html = f"""
    <div style=\"margin: 8px 0 16px 0;\">
            <h4 style=\"margin: 0 0 8px 0; font-weight: 700; color: {accent_color};\">{label.title()} - {title}</h4>
      <style>
        .ranked-table {{ border-collapse: separate; border-spacing: 0; min-width: 360px; font-size: 13px; color: #1f2933; overflow: hidden; border-radius: 10px; }}
        .ranked-table thead th {{ background: {accent_color}; color: #ffffff; padding: 10px 14px; font-weight: 700; border: none; }}
        .ranked-table tbody td {{ padding: 9px 14px; border-bottom: 1px solid #d6dee6; }}
        .ranked-table tbody tr:nth-child(odd) td {{ background: #f4f7fb; }}
        .ranked-table tbody tr:nth-child(even) td {{ background: #e9eff6; }}
        .ranked-table tbody tr:last-child td {{ border-bottom: none; }}
      </style>
      <div style=\"display: inline-block; border: 2px solid {accent_color}; border-radius: 12px; padding: 0; background: #ffffff; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08); overflow: hidden;\">{table_html}</div>
    </div>
    """
    display(HTML(html))
    return df


def find_top_hubs(
    gcc_df: pd.DataFrame,
    top_n: int | None = None,
    min_degree: int | None = None,
) -> list[tuple[str, int]]:
    degree_counts = pd.concat([gcc_df["source"], gcc_df["target"]]).value_counts()
    if min_degree is not None:
        degree_counts = degree_counts[degree_counts > min_degree]
    if top_n is not None:
        degree_counts = degree_counts.head(top_n)
    top_hubs = degree_counts.reset_index()
    top_hubs.columns = ["Hub", "Degree"]
    return list(top_hubs.itertuples(index=False, name=None))


def find_top_hubs_by_type(
    gcc_df: pd.DataFrame,
    top_n: int = 5,
    min_degree: int | None = None,
) -> dict[str, pd.DataFrame]:
    if "type" not in gcc_df.columns:
        raise ValueError("gcc_df must contain a 'type' column with C/S/D edge labels")

    edge_df = gcc_df.copy()
    edge_df["source"] = edge_df["source"].astype(str).str.strip()
    edge_df["target"] = edge_df["target"].astype(str).str.strip()
    edge_df["type"] = edge_df["type"].astype(str).str.strip().str.upper()

    results: dict[str, pd.DataFrame] = {}
    for interaction_type in sorted(edge_df["type"].dropna().unique()):
        type_edges = edge_df.loc[edge_df["type"] == interaction_type, ["source", "target"]]
        degree_counts = pd.concat([type_edges["source"], type_edges["target"]]).value_counts()
        if min_degree is not None:
            degree_counts = degree_counts[degree_counts > min_degree]
        top_hubs = degree_counts.head(top_n).reset_index()
        top_hubs.columns = ["Hub", "Degree"]
        top_hubs.insert(0, "InteractionType", interaction_type)
        results[interaction_type] = top_hubs.reset_index(drop=True)

    return results


def build_node_degree_table(graph: nx.Graph) -> pd.DataFrame:
    degree_table = pd.DataFrame(graph.degree(), columns=["Node", "Degree"])
    if degree_table.empty:
        return degree_table
    degree_table["Node"] = degree_table["Node"].astype(str).str.strip()
    degree_table["Degree"] = pd.to_numeric(degree_table["Degree"], errors="coerce").fillna(0).astype(int)
    return degree_table.sort_values(["Degree", "Node"], ascending=[False, True]).reset_index(drop=True)


def display_top_hubs_table(hubs_df: pd.DataFrame, title: str, top_n: int = 10) -> pd.DataFrame:
    top_hubs = hubs_df.head(top_n).copy()
    if top_hubs.empty:
        print("No hubs found.")
        return top_hubs

    top_hubs.insert(0, "Rank", np.arange(1, len(top_hubs) + 1))
    render_ranked_html_table(
        top_hubs,
        title=title,
        label=f"hubs (top {top_n})",
        formatters={"Degree": lambda value: f"{value:.0f}"},
        accent_color="#1D4ED8",
    )
    return top_hubs


def analyze_hub_homogeneity(gcc_df: pd.DataFrame, hubs: list[tuple[str, int]] | list[str]) -> list[dict[str, Any]]:
    if "type" not in gcc_df.columns:
        raise ValueError("gcc_df must contain a 'type' column with C/S/D edge labels")

    edge_df = gcc_df.copy()
    edge_df["source"] = edge_df["source"].astype(str).str.strip()
    edge_df["target"] = edge_df["target"].astype(str).str.strip()
    edge_df["type"] = edge_df["type"].astype(str).str.strip().str.upper()

    hub_names: list[str] = []
    for hub in hubs:
        if isinstance(hub, (tuple, list, np.ndarray)) and len(hub) > 0:
            hub_names.append(str(hub[0]).strip())
        else:
            hub_names.append(str(hub).strip())

    hub_names = pd.Series(hub_names)
    hub_names = hub_names[hub_names != ""].drop_duplicates().tolist()

    rows: list[dict[str, Any]] = []
    for hub in hub_names:
        hub_edges = edge_df[(edge_df["source"] == hub) | (edge_df["target"] == hub)].copy()
        type_counts = hub_edges["type"].value_counts()
        c_count = int(type_counts.get("C", 0))
        s_count = int(type_counts.get("S", 0))
        d_count = int(type_counts.get("D", 0))
        total_edges = c_count + s_count + d_count

        if total_edges == 0:
            c_fraction = s_fraction = d_fraction = homogeneity_score = 0.0
            dominant_type = "N/A"
        else:
            c_fraction = c_count / total_edges
            s_fraction = s_count / total_edges
            d_fraction = d_count / total_edges
            homogeneity_score = max(c_fraction, s_fraction, d_fraction)
            dominant_type = max(
                {"C": c_fraction, "S": s_fraction, "D": d_fraction},
                key=lambda edge_type: {"C": c_fraction, "S": s_fraction, "D": d_fraction}[edge_type],
            )

        rows.append(
            {
                "Hub": hub,
                "n_edges": total_edges,
                "C_count": c_count,
                "S_count": s_count,
                "D_count": d_count,
                "C_fraction": round(c_fraction, 4),
                "S_fraction": round(s_fraction, 4),
                "D_fraction": round(d_fraction, 4),
                "homogeneity_score": round(homogeneity_score, 4),
                "dominant_type": dominant_type,
            }
        )
    return rows


def mark_hubs_in_cytoscape(
    network_name: str,
    hubs: list[str] | list[tuple[str, int]],
    color: str = "#FFD700",
    size: int = 40,
    label_hubs: bool = True,
    use_gene_symbols: bool = True,
) -> None:
    p4c = cytoscape_client()
    p4c.cytoscape_ping()
    p4c.set_current_network(network_name)
    node_table = p4c.get_table_columns(table="node", columns=["name"], network=network_name)
    node_names = set(node_table["name"].astype(str).str.strip())

    hub_ids = []
    for hub in hubs:
        if isinstance(hub, (tuple, list, np.ndarray)) and len(hub) > 0:
            hub_ids.append(hub[0])
        else:
            hub_ids.append(hub)
    hub_ids = pd.Series(hub_ids).dropna().astype(str).str.strip().tolist()
    hubs_in_network = [hub for hub in hub_ids if hub in node_names]
    if not hubs_in_network:
        print(f"No hubs found in {network_name} to mark.")
        return

    p4c.set_node_color_bypass(node_names=hubs_in_network, new_colors=[color] * len(hubs_in_network), network=network_name)
    p4c.set_node_size_bypass(node_names=hubs_in_network, new_sizes=[size] * len(hubs_in_network), network=network_name)
    if label_hubs:
        if use_gene_symbols:
            ensembl_hubs = [hub for hub in hubs_in_network if hub.startswith("ENSG")]
            id_to_symbol = convert_ensembl_to_symbol(ensembl_hubs) if ensembl_hubs else {}
            labels = [id_to_symbol.get(hub, hub) if hub.startswith("ENSG") else hub for hub in hubs_in_network]
        else:
            labels = hubs_in_network
        p4c.set_node_label_bypass(node_names=hubs_in_network, new_labels=labels, network=network_name)
