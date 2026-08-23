"""Plotting helpers for transition systems and their coarse graining."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
import networkx as nx
import numpy as np


def plot_transition_graph(
    transition_list,
    lumps,
    output_path="data/transition_graph.png",
    *,
    seed=7,
    layout="spring",
    positions=None,
    arc_curvature=0.12,
):
    """Plot a directed transition graph with its coarse-grained state groups.

    Args:
        transition_list: Iterable of ``(source, target, rate)`` transitions.
        lumps: Iterable of iterables containing the microstates in each coarse
            state.
        output_path: Location at which to save the resulting figure.
        seed: Random seed used by the spring layout.
        layout: Node-layout algorithm. ``"spring"`` preserves the original
            force-directed layout. ``"kamada_kawai"`` generally gives dense
            graphs more space and reduces edge crossings. Transition rates are
            deliberately not treated as distances in the latter layout.
        positions: Optional mapping from every node to an ``(x, y)`` position.
            When provided, these coordinates override ``layout`` and ``seed``.
        arc_curvature: Curvature of transition arrows and their rate labels.
            The default ``0.12`` preserves the existing appearance.

    Returns:
        The Matplotlib figure and axes containing the graph.
    """
    transitions = list(transition_list)
    coarse_states = [list(lump) for lump in lumps]
    arc_curvature = float(arc_curvature)
    if not np.isfinite(arc_curvature):
        raise ValueError("arc_curvature must be finite.")
    connection_style = f"arc3,rad={arc_curvature:g}"

    graph = nx.DiGraph()
    for source, target, rate in transitions:
        graph.add_edge(source, target, rate=float(rate), weight=float(rate))
    for lump in coarse_states:
        graph.add_nodes_from(lump)

    if not graph:
        raise ValueError("Cannot plot an empty transition graph.")

    node_to_lump = {}
    for lump_index, lump in enumerate(coarse_states):
        for node in lump:
            if node in node_to_lump:
                raise ValueError(f"Microstate {node!r} occurs in more than one lump.")
            node_to_lump[node] = lump_index

    if positions is not None:
        missing_nodes = [node for node in graph if node not in positions]
        if missing_nodes:
            raise ValueError(f"positions is missing nodes: {missing_nodes!r}.")
        positions = {
            node: np.asarray(positions[node], dtype=float) for node in graph
        }
        invalid_nodes = [
            node
            for node, position in positions.items()
            if position.shape != (2,) or not np.isfinite(position).all()
        ]
        if invalid_nodes:
            raise ValueError(
                "Each position must contain two finite coordinates; "
                f"invalid nodes: {invalid_nodes!r}."
            )
    elif layout == "spring":
        positions = nx.spring_layout(graph, seed=seed, weight="weight", k=1.15)
    elif layout == "kamada_kawai":
        positions = nx.kamada_kawai_layout(graph, weight=None)
    else:
        raise ValueError(
            f"Unknown layout {layout!r}; expected 'spring' or 'kamada_kawai'."
        )
    figure, axes = plt.subplots(figsize=(9, 7), constrained_layout=True)

    color_map = plt.get_cmap("tab10")
    lump_colors = [color_map(index % color_map.N) for index in range(len(coarse_states))]
    node_colors = [
        lump_colors[node_to_lump[node]] if node in node_to_lump else "#b8bec7"
        for node in graph.nodes
    ]

    all_positions = np.asarray(list(positions.values()))
    graph_span = np.maximum(np.ptp(all_positions, axis=0), 1.0)
    for lump_index, lump in enumerate(coarse_states):
        lump_positions = np.asarray([positions[node] for node in lump if node in positions])
        if lump_positions.size == 0:
            continue
        lower = lump_positions.min(axis=0)
        upper = lump_positions.max(axis=0)
        center = (lower + upper) / 2
        width, height = np.maximum(upper - lower + 0.30 * graph_span, 0.36 * graph_span)
        boundary = Ellipse(
            center,
            width=width,
            height=height,
            facecolor=lump_colors[lump_index],
            edgecolor=lump_colors[lump_index],
            alpha=0.12,
            linestyle="--",
            linewidth=2,
            zorder=0,
        )
        axes.add_patch(boundary)

    rates = np.asarray([data["rate"] for _, _, data in graph.edges(data=True)])
    if rates.size and rates.max() > rates.min():
        edge_widths = 1.5 + 2.5 * (rates - rates.min()) / (rates.max() - rates.min())
    else:
        edge_widths = np.full(len(graph.edges), 2.2)

    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axes,
        node_color=node_colors,
        node_size=1500,
        edgecolors="white",
        linewidths=2,
    )
    nx.draw_networkx_labels(graph, positions, ax=axes, font_size=12, font_weight="bold")
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axes,
        width=edge_widths,
        edge_color="#46505a",
        arrows=True,
        arrowsize=20,
        node_size=1500,
        connectionstyle=connection_style,
    )
    edge_labels = {
        (source, target): f"{data['rate']:g}"
        for source, target, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        ax=axes,
        edge_labels=edge_labels,
        label_pos=0.5,
        font_size=9,
        rotate=False,
        connectionstyle=connection_style,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.95},
    )

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor=color,
            alpha=0.55,
            label=f"Coarse state {index}",
        )
        for index, color in enumerate(lump_colors)
    ]
    if legend_handles:
        axes.legend(handles=legend_handles, loc="upper left", frameon=False)

    axes.set_title("Transition graph and coarse graining", fontsize=15, pad=14)
    axes.set_axis_off()
    axes.margins(0.22)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight", facecolor="white")
    return figure, axes
