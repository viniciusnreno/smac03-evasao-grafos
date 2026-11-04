"""Figuras estáticas (PNG, 200 dpi) para relatório e slides."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Paleta categórica validada (slots 1-3 passam CVD em todos os pares)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OUT_COLOR = {"Dropout": ORANGE, "Enrolled": BLUE, "Graduate": AQUA}
OUT_PT = {"Dropout": "Evasão", "Enrolled": "Matriculado", "Graduate": "Graduado"}
OUT_COLOR_PT = {OUT_PT[k]: v for k, v in OUT_COLOR.items()}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def outcome_legend(ax, loc="upper right"):
    h = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=8, label=OUT_PT[k])
         for k, c in OUT_COLOR.items()]
    ax.legend(handles=h, loc=loc)


# ------------------------------------------------------------------ Modelo 1
def similarity_network(out: Path, G: nx.Graph, comms, target: pd.Series, n_sample: int = 1500):
    """Amostra do grafo de similaridade com layout guiado pelas comunidades."""
    rng = np.random.default_rng(42)
    nodes = rng.choice(G.number_of_nodes(), n_sample, replace=False)
    H = G.subgraph(nodes).copy()
    H.remove_nodes_from([v for v in list(H) if H.degree(v) == 0])
    pos = nx.spring_layout(H, weight="weight", seed=42, iterations=80, k=0.08)
    t = target.to_numpy()
    comm_of = {v: i for i, c in enumerate(sorted(comms, key=len, reverse=True)) for v in c}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, mode in zip(axes, ("desfecho", "comunidade")):
        nx.draw_networkx_edges(H, pos, ax=ax, width=0.15, alpha=0.25, edge_color="#9a9893")
        if mode == "desfecho":
            colors = [OUT_COLOR[t[v]] for v in H]
            ax.set_title("Modelo 1 — estudantes coloridos pelo desfecho")
        else:
            cmap = plt.get_cmap("tab10")
            colors = [cmap(comm_of[v] % 10) for v in H]
            ax.set_title("Modelo 1 — comunidades (Louvain)")
        nx.draw_networkx_nodes(H, pos, ax=ax, node_size=7, node_color=colors, linewidths=0)
        ax.set_axis_off()
    outcome_legend(axes[0], "lower left")
    fig.text(0.5, 0.01, f"Amostra aleatória de {H.number_of_nodes()} de {G.number_of_nodes()} estudantes "
             "(k-NN, k=15). O desfecho não é usado na construção do grafo.", ha="center", color=INK2, fontsize=9)
    _save(fig, out / "fig1_rede_similaridade.png")


def community_outcomes(out: Path, profiles: pd.DataFrame):
    p = profiles.copy()
    labels = [f"C{r.comunidade} (n={r.tamanho})" for r in p.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    left = np.zeros(len(p))
    for col, key in (("evasao_%", "Dropout"), ("matriculado_%", "Enrolled"), ("graduado_%", "Graduate")):
        ax.barh(labels, p[col], left=left, color=OUT_COLOR[key], label=OUT_PT[key],
                edgecolor=SURF, linewidth=1.5, height=0.7)
        left += p[col].to_numpy()
    for i, v in enumerate(p["evasao_%"]):
        ax.text(1, i, f"{v:.0f}%", va="center", ha="left", color="white", fontsize=8, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% dos estudantes da comunidade")
    ax.set_title("Desfecho por comunidade (ordenado por taxa de evasão)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis="y", visible=False)
    _save(fig, out / "fig2_desfecho_por_comunidade.png")


def validation(out: Path, cv: pd.DataFrame, cv_pre: pd.DataFrame):
    m = cv.merge(cv_pre, on="metodo", suffixes=("_full", "_pre"))
    m = m.sort_values("AUC_full")
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(y + 0.19, m["AUC_full"], height=0.36, color=BLUE, label="Ingresso + 1º/2º semestres")
    ax.barh(y - 0.19, m["AUC_pre"], height=0.36, color=ORANGE, label="Apenas ingresso (alerta precoce)")
    for i, (a, b) in enumerate(zip(m["AUC_full"], m["AUC_pre"])):
        ax.text(a + 0.004, i + 0.19, f"{a:.3f}", va="center", fontsize=8, color=INK2)
        ax.text(b + 0.004, i - 0.19, f"{b:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, m["metodo"])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC-ROC (Evasão vs. demais), validação cruzada 5-fold")
    ax.set_title("Escore de risco: relacional vs. tabular")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.35, -0.15))
    ax.grid(axis="y", visible=False)
    _save(fig, out / "fig3_validacao_auc.png")


def divergent(out: Path, div: pd.DataFrame, top: int = 10):
    d = div.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(d["atributo"], d["difere_Evasao_x_Graduado_%"], color=ORANGE, height=0.38,
            label="Pares Evasão × Graduado", align="edge")
    ax.barh(d["atributo"], d["difere_Graduado_x_Graduado_%"], color=AQUA, height=-0.38,
            label="Pares Graduado × Graduado", align="edge")
    ax.set_xlabel("% dos pares em que o atributo difere")
    ax.set_title("Perfis de ingresso quase idênticos: o que separa os desfechos?")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, out / "fig4_pares_divergentes.png")


def sensitivity(out: Path, sens: pd.DataFrame):
    if sens.empty:
        return
    s = sens.copy()
    s["config"] = s.apply(lambda r: f"α={r.alpha_ACAD:.2f} β={r.beta_SOCIO:.2f} γ={r.gamma_DEMO:.2f}", axis=1)
    piv = s.pivot_table(index="config", columns="k", values="AUC_voto", sort=False)
    piv2 = s.pivot_table(index="config", columns="k", values="assortatividade", sort=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    colors = [BLUE, ORANGE, AQUA]
    for ax, data, title in ((axes[0], piv, "AUC do voto dos vizinhos"),
                            (axes[1], piv2, "Assortatividade pelo desfecho")):
        y = np.arange(len(data))
        for i, kk in enumerate(data.columns):
            ax.barh(y + (i - 1) * 0.26, data[kk], height=0.24, color=colors[i], label=f"k={kk}")
        ax.set_yticks(y, data.index)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(0.6, 0.95)
    axes[1].legend(loc="lower right")
    _save(fig, out / "fig5_sensibilidade.png")


def mixing_heatmap(out: Path, M: pd.DataFrame):
    M = M.rename(index=OUT_PT, columns=OUT_PT)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(M.to_numpy(), cmap="Blues")
    ax.set_xticks(range(3), M.columns); ax.set_yticks(range(3), M.index)
    for i in range(3):
        for j in range(3):
            v = M.iat[i, j]
            ax.text(j, i, f"{100*v:.1f}%", ha="center", va="center",
                    color="white" if v > M.values.max() * 0.6 else INK, fontsize=9)
    ax.grid(False)
    ax.set_title("Matriz de mistura das arestas")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out / "fig6_matriz_mistura.png")


# ------------------------------------------------------------------ Modelo 2
def factor_network(out: Path, F: nx.Graph, fcomms):
    H = F.copy()
    H.remove_nodes_from([v for v in list(H) if H.degree(v) == 0])
    comm_of = {v: i for i, c in enumerate(sorted(fcomms, key=len, reverse=True)) for v in c}
    # cor da comunidade = cor do desfecho que ela contém
    comm_color = {}
    for v in H:
        if H.nodes[v]["kind"] == "desfecho":
            comm_color[comm_of[v]] = OUT_COLOR_PT[v]
    H.remove_nodes_from([v for v in list(H) if H.degree(v) <= 2 and H.nodes[v]["kind"] == "fator"])
    pos = nx.kamada_kawai_layout(H, weight=None)
    fig, ax = plt.subplots(figsize=(16, 12))
    w = np.array([d["lift"] for *_, d in H.edges(data=True)])
    nx.draw_networkx_edges(H, pos, ax=ax, width=0.3 + 1.2 * (w - w.min()) / (w.max() - w.min()),
                           alpha=0.25, edge_color="#9a9893")
    outc = [v for v in H if H.nodes[v]["kind"] == "desfecho"]
    facs = [v for v in H if v not in outc]
    nx.draw_networkx_nodes(H, pos, nodelist=facs, ax=ax, node_size=[40 + 400 * H.nodes[v]["prob"] for v in facs],
                           node_color=[comm_color.get(comm_of[v], "#b7b5ae") for v in facs],
                           edgecolors=SURF, linewidths=1)
    nx.draw_networkx_nodes(H, pos, nodelist=outc, ax=ax, node_size=900, node_shape="s",
                           node_color=[OUT_COLOR_PT[v] for v in outc], edgecolors=INK, linewidths=1.5)
    for v in outc:
        x, y = pos[v]
        ax.text(x, y - 0.06, v.upper(), ha="center", va="top", fontsize=13, fontweight="bold", color=INK,
                bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.85, pad=1))
    for v in facs:
        x, y = pos[v]
        ax.text(x, y + 0.022, "=\n".join(v.split("=", 1)), ha="center", va="bottom",
                fontsize=7, color=INK, linespacing=0.9)
    ax.set_title("Modelo 2 — rede de fatores (arestas: lift ≥ 1,15; cor: comunidade Louvain)")
    ax.set_axis_off()
    _save(fig, out / "fig7_rede_fatores.png")


def ego_dropout(out: Path, ego: pd.DataFrame, top: int = 15):
    e = ego.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(e["fator"], e["lift"], color=ORANGE, height=0.7)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    for i, (l, n) in enumerate(zip(e["lift"], e["estudantes"])):
        ax.text(l + 0.03, i, f"{l:.2f} (n={n})", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("lift com Evasão (1 = independência)")
    ax.set_title("Vizinhança do nó Evasão na rede de fatores")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, e["lift"].max() * 1.25)
    _save(fig, out / "fig8_vizinhanca_evasao.png")


def all_figures(out: Path, *, G, comms, target, profiles, evals, cv, cv_pre, div, F, egos, fcomms, sens, mixing):
    similarity_network(out, G, comms, target)
    community_outcomes(out, profiles)
    validation(out, cv, cv_pre)
    divergent(out, div)
    sensitivity(out, sens)
    mixing_heatmap(out, mixing)
    factor_network(out, F, fcomms)
    ego_dropout(out, egos["Evasão"])
