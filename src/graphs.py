"""Construção dos grafos.

Modelo 1 — Rede de Similaridade Multirrelacional (estudante x estudante)
    G1 = (V, E, w): V = estudantes; (i, j) ∈ E se j está entre os k vizinhos mais
    similares de i (ou vice-versa); w(i, j) = R_ij, com

        R_ij = α·S_ACAD(i, j) + β·S_SOCIO(i, j) + γ·S_DEMO(i, j),   α+β+γ = 1

    e S_g = média das similaridades por atributo do grupo g:
        ordinal : 1 − |r_i − r_j| / (A − 1)
        nominal : 1 se igual, 0 caso contrário
        ausente : 0,5 (neutro)
    O desfecho (Target) NÃO entra no cálculo: ele é usado apenas para avaliar
    a estrutura obtida.

Modelo 2 — Rede de Fatores (fator x fator, com nós de desfecho)
    Grafo bipartido estudante–fator projetado sobre os fatores. Peso da aresta
    (a, b) = lift(a, b) = P(a ∧ b) / (P(a)·P(b)); só se mantêm associações
    positivas (lift ≥ lift_min) com suporte mínimo. Os três desfechos
    (Evasão, Matriculado, Graduado) entram como nós, de modo que a vizinhança
    de "Evasão" mostra a constelação de fatores associada ao abandono.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd

GROUPS = ("ACAD", "SOCIO", "DEMO")


# ------------------------------------------------------------------ Modelo 1
def group_similarity(f: pd.DataFrame, schema: dict, attrs: list[str], rows: np.ndarray) -> np.ndarray:
    """Similaridade média (rows x N) considerando apenas `attrs`."""
    n = len(f)
    acc = np.zeros((len(rows), n), dtype=np.float32)
    for a in attrs:
        col = f[a].to_numpy(dtype=np.float32)
        x, y = col[rows][:, None], col[None, :]
        if schema[a]["kind"] == "ord":
            span = max(len(schema[a]["labels"]) - 1, 1)
            s = 1.0 - np.abs(x - y) / span
        else:
            s = (x == y).astype(np.float32)
        s = np.where(np.isnan(x) | np.isnan(y), 0.5, s)
        acc += s
    return acc / max(len(attrs), 1)


def similarity_knn(f: pd.DataFrame, schema: dict, weights: dict[str, float], k: int = 15,
                   attrs: list[str] | None = None, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (idx, sim) dos k vizinhos mais similares de cada estudante."""
    attrs = attrs or list(f.columns)
    by_group = {g: [a for a in attrs if schema[a]["group"] == g] for g in GROUPS}
    wsum = sum(weights[g] for g in GROUPS if by_group[g])
    n = len(f)
    idx = np.empty((n, k), dtype=np.int64)
    sim = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        R = np.zeros((len(rows), n), dtype=np.float32)
        for g in GROUPS:
            if by_group[g]:
                R += weights[g] / wsum * group_similarity(f, schema, by_group[g], rows)
        R[np.arange(len(rows)), rows] = -1.0  # sem laços
        part = np.argpartition(-R, k, axis=1)[:, :k]
        vals = np.take_along_axis(R, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx[rows] = np.take_along_axis(part, order, axis=1)
        sim[rows] = np.take_along_axis(vals, order, axis=1)
    return idx, sim


def build_similarity_graph(f: pd.DataFrame, schema: dict, target: pd.Series,
                           weights: dict[str, float] | None = None, k: int = 15,
                           attrs: list[str] | None = None) -> nx.Graph:
    weights = weights or {"ACAD": 0.5, "SOCIO": 0.3, "DEMO": 0.2}
    idx, sim = similarity_knn(f, schema, weights, k=k, attrs=attrs)
    G = nx.Graph(model="similaridade", k=k, **{f"w_{g}": weights[g] for g in GROUPS})
    for i in range(len(f)):
        G.add_node(i, target=target.iloc[i])
    for i in range(len(f)):
        for j, s in zip(idx[i], sim[i]):
            if G.has_edge(i, j):
                continue
            G.add_edge(i, int(j), weight=float(s))
    return G


# ------------------------------------------------------------------ Modelo 2
def build_factor_graph(onehot: pd.DataFrame, target_pt: pd.Series, min_support: int = 40,
                       lift_min: float = 1.15) -> nx.Graph:
    outcomes = pd.get_dummies(target_pt).astype(np.int8)
    M = pd.concat([onehot, outcomes], axis=1)
    keep = M.columns[M.sum() >= min_support]
    M = M[keep]
    X = M.to_numpy(dtype=np.float32)
    n = X.shape[0]
    co = X.T @ X                       # co-ocorrências
    p = np.diag(co) / n
    lift = (co / n) / np.outer(p, p)
    names = list(M.columns)
    attr_of = [c.split("=")[0] for c in names]

    G = nx.Graph(model="fatores", n_students=n)
    for i, c in enumerate(names):
        G.add_node(c, support=int(co[i, i]), prob=float(p[i]),
                   kind="desfecho" if c in outcomes.columns else "fator", attr=attr_of[i])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if attr_of[i] == attr_of[j]:
                continue  # mesmos atributo -> mutuamente exclusivos
            if co[i, j] < min_support or lift[i, j] < lift_min:
                continue
            G.add_edge(names[i], names[j], weight=float(lift[i, j]), lift=float(lift[i, j]),
                       count=int(co[i, j]), pmi=float(np.log2(lift[i, j])))
    return G
