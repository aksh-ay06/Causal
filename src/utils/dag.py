"""Causal DAG definition and utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from src.utils.config import FIGURES_DIR


class CausalDAG:
    """Lightweight wrapper around a ``networkx.DiGraph`` for causal analysis."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    # ── Build ──────────────────────────────────────────────────────────────
    def add_edges(self, edges: list[tuple[str, str]]) -> None:
        self.graph.add_edges_from(edges)

    @classmethod
    def smoking_health_dag(cls) -> "CausalDAG":
        """Pre‑built DAG for the smoking → health causal question."""
        dag = cls()
        dag.add_edges(
            [
                # Confounders → Treatment
                ("Age", "Smoking"),
                ("Gender", "Smoking"),
                ("Race", "Smoking"),
                ("Education", "Smoking"),
                ("Income", "Smoking"),
                # Confounders → Outcomes
                ("Age", "HealthScore"),
                ("Age", "Cancer"),
                ("Gender", "HealthScore"),
                ("Gender", "Cancer"),
                ("Race", "HealthScore"),
                ("Education", "HealthScore"),
                ("Education", "Cancer"),
                ("Income", "HealthScore"),
                # Treatment → Outcomes
                ("Smoking", "HealthScore"),
                ("Smoking", "Cancer"),
                # Confounder inter‑relations
                ("Education", "Income"),
                ("Age", "Income"),
            ]
        )
        return dag

    # ── Identification helpers ─────────────────────────────────────────────
    def parents(self, node: str) -> set[str]:
        return set(self.graph.predecessors(node))

    def backdoor_variables(self, treatment: str, outcome: str) -> set[str]:
        """Return parent‑based backdoor adjustment set (simple heuristic)."""
        return self.parents(treatment) | (self.parents(outcome) - {treatment})

    # ── Visualisation ──────────────────────────────────────────────────────
    def plot(self, save: bool = True, filename: str = "causal_dag.png") -> None:
        fig, ax = plt.subplots(figsize=(10, 7))
        pos = nx.spring_layout(self.graph, seed=0, k=2)
        nx.draw_networkx(
            self.graph,
            pos,
            ax=ax,
            node_color="#4C72B0",
            font_color="white",
            font_weight="bold",
            node_size=2500,
            arrowsize=20,
            edge_color="#999999",
        )
        ax.set_title("Causal DAG — Smoking → Health Outcomes", fontsize=14)
        plt.tight_layout()
        if save:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
