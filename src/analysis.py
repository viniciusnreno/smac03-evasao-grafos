"""Métricas e análises sobre os grafos."""
from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sp
from networkx.algorithms import community as nxc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42


# ------------------------------------------------------------ estrutura geral
def graph_summary(G: nx.Graph, attr: str | None = "target") -> dict:
    degs = np.array([d for _, d in G.degree()])
    out = {
        "vertices": G.number_of_nodes(),
        "arestas": G.number_of_edges(),
        "densidade": nx.density(G),
        "grau_medio": float(degs.mean()),
        "grau_max": int(degs.max()),
        "componentes": nx.number_connected_components(G),
        "clustering_medio": nx.average_clustering(G, weight=None),
    }
    if attr:
        out["assortatividade_desfecho"] = nx.attribute_assortativity_coefficient(G, attr)
        out.update(homophily(G, attr))
    return out


def homophily(G: nx.Graph, attr: str = "target") -> dict:
    lab = nx.get_node_attributes(G, attr)
    same = sum(lab[u] == lab[v] for u, v in G.edges())
    p = pd.Series(lab).value_counts(normalize=True)
    return {"homofilia_observada": same / G.number_of_edges(), "homofilia_esperada": float((p ** 2).sum())}


def mixing_matrix(G: nx.Graph, attr: str = "target") -> pd.DataFrame:
    lab = nx.get_node_attributes(G, attr)
    cats = sorted(set(lab.values()))
    M = pd.DataFrame(0, index=cats, columns=cats, dtype=float)
    for u, v in G.edges():
        M.loc[lab[u], lab[v]] += 1
        M.loc[lab[v], lab[u]] += 1
    return M / M.values.sum()


# ------------------------------------------------------- detecção de comunidades
def detect_communities(G: nx.Graph) -> dict[str, list[set]]:
    return {
        "Louvain (Multilevel)": nxc.louvain_communities(G, weight="weight", seed=SEED, resolution=1.0),
        "Fast Greedy": list(nxc.greedy_modularity_communities(G, weight="weight")),
        "Label Propagation": list(nxc.asyn_lpa_communities(G, weight="weight", seed=SEED)),
    }


def partition_vector(comms: list[set], n: int) -> np.ndarray:
    lab = np.empty(n, dtype=int)
    for c, nodes in enumerate(sorted(comms, key=len, reverse=True)):
        for v in nodes:
            lab[v] = c
    return lab


def evaluate_partition(G: nx.Graph, comms: list[set], target: np.ndarray) -> dict:
    lab = partition_vector(comms, len(target))
    inter = sum(1 for u, v in G.edges() if lab[u] != lab[v])
    sizes = sorted((len(c) for c in comms), reverse=True)
    dropout_rate = pd.Series(target == "Dropout").groupby(lab).mean()
    big = [c for c in range(len(sizes)) if sizes[c] >= 30]
    return {
        "comunidades": len(comms),
        "comunidades_>=30": len(big),
        "maior": sizes[0],
        "modularidade": nxc.modularity(G, comms, weight="weight"),
        "arestas_inter_%": 100 * inter / G.number_of_edges(),
        "NMI_desfecho": normalized_mutual_info_score(target, lab),
        "ARI_desfecho": adjusted_rand_score(target, lab),
        "amplitude_evasao_%": 100 * (dropout_rate[big].max() - dropout_rate[big].min()) if big else np.nan,
    }


def community_profiles(comms: list[set], target: pd.Series, onehot: pd.DataFrame,
                       min_size: int = 30, top: int = 6) -> pd.DataFrame:
    """Perfil de cada comunidade: tamanho, distribuição de desfechos e fatores sobre-representados."""
    lab = partition_vector(comms, len(target))
    glob = onehot.mean()
    rows = []
    for c in np.unique(lab):
        m = lab == c
        if m.sum() < min_size:
            continue
        t = target[m].value_counts(normalize=True)
        local = onehot[m].mean()
        lift = (local / glob.replace(0, np.nan)).fillna(0)
        cand = lift[(local >= 0.5) & (lift >= 1.2)].sort_values(ascending=False)
        rows.append({
            "comunidade": int(c), "tamanho": int(m.sum()),
            "evasao_%": 100 * t.get("Dropout", 0), "matriculado_%": 100 * t.get("Enrolled", 0),
            "graduado_%": 100 * t.get("Graduate", 0),
            "fatores_caracteristicos": "; ".join(f"{k} ({100*local[k]:.0f}%)" for k in cand.index[:top]),
        })
    return pd.DataFrame(rows).sort_values("evasao_%", ascending=False).reset_index(drop=True)


# ---------------------------------------------- perfis semelhantes, desfechos opostos
def divergent_pairs(G: nx.Graph, f: pd.DataFrame, schema: dict, target: pd.Series, q: float = 0.75):
    """Compara pares muito similares Evasão–Graduado com pares muito similares Graduado–Graduado.

    Use com o grafo construído só com atributos de ingresso: pares com o mesmo perfil de
    entrada mas desfechos opostos. Retorna, para TODOS os atributos (inclusive os de
    trajetória, que não entraram no grafo), a frequência com que diferem em cada tipo de
    par; o excesso indica o que "separa" trajetórias de perfis aparentemente iguais.
    """
    w = np.array([d["weight"] for *_, d in G.edges(data=True)])
    thr = np.quantile(w, q)
    t = target.to_numpy()
    X = f.to_numpy()
    div, same = [], []
    for u, v, d in G.edges(data=True):
        if d["weight"] < thr:
            continue
        pair = {t[u], t[v]}
        if pair == {"Dropout", "Graduate"}:
            div.append((u, v))
        elif pair == {"Graduate"}:
            same.append((u, v))

    def diff_rate(pairs):
        a = np.array([u for u, _ in pairs]); b = np.array([v for _, v in pairs])
        xa, xb = X[a], X[b]
        na, nb = np.isnan(xa), np.isnan(xb)  # ausente (ex.: sem aprovação -> sem média) é um valor próprio
        neq = ((xa != xb) & ~(na | nb)) | (na != nb)
        return neq.mean(axis=0)

    res = pd.DataFrame({
        "atributo": f.columns,
        "grupo": [schema[c]["group"] for c in f.columns],
        "difere_Evasao_x_Graduado_%": 100 * diff_rate(div),
        "difere_Graduado_x_Graduado_%": 100 * diff_rate(same),
    })
    res["excesso_pp"] = res.iloc[:, 2] - res.iloc[:, 3]
    return res.sort_values("excesso_pp", ascending=False).reset_index(drop=True), len(div), len(same), thr


def surprising_nodes(G: nx.Graph, target: pd.Series, outcome: str = "Dropout", max_share: float = 0.25):
    """Estudantes com desfecho `outcome` cuja vizinhança quase não tem esse desfecho
    (heterofilia local): casos que um sistema baseado em perfil não anteciparia."""
    t = target.to_numpy()
    rows = []
    for v in G.nodes():
        if t[v] != outcome:
            continue
        nb = list(G[v])
        share = np.mean([t[u] == outcome for u in nb]) if nb else np.nan
        rows.append((v, share))
    df = pd.DataFrame(rows, columns=["estudante", "vizinhos_mesmo_desfecho"])
    return df, float((df["vizinhos_mesmo_desfecho"] <= max_share).mean())


# --------------------------------------------- escore de risco relacional + validação
def _adj(G: nx.Graph, n: int) -> sp.csr_matrix:
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), weight="weight", format="csr")
    return sp.csr_matrix(A)


def neighbor_vote(A: sp.csr_matrix, y: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Fração ponderada de vizinhos rotulados que evadiram."""
    yk = np.where(known, y, 0.0)
    num = A @ yk
    den = A @ known.astype(float)
    prior = y[known].mean()
    return np.where(den > 0, num / np.maximum(den, 1e-12), prior)


def label_spreading(A: sp.csr_matrix, y: np.ndarray, known: np.ndarray, alpha: float = 0.8,
                    iters: int = 30) -> np.ndarray:
    """Propagação de rótulos (Zhou et al., 2004) com normalização simétrica."""
    d = np.asarray(A.sum(axis=1)).ravel()
    dinv = sp.diags(1 / np.sqrt(np.maximum(d, 1e-12)))
    S = dinv @ A @ dinv
    prior = y[known].mean()
    Y0 = np.where(known, y - prior, 0.0)
    F = Y0.copy()
    for _ in range(iters):
        F = alpha * (S @ F) + (1 - alpha) * Y0
    return F


def cross_validate_risk(G: nx.Graph, y: np.ndarray, onehot: pd.DataFrame, f: pd.DataFrame,
                        folds: int = 5) -> pd.DataFrame:
    """AUC (Evasão vs. demais) de escores relacionais vs. baselines tabulares."""
    n = len(y)
    A = _adj(G, n)
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    scores = {k: np.zeros(n) for k in ("Grafo: voto dos vizinhos", "Grafo: propagação de rótulos",
                                       "Tabular: regressão logística", "Tabular: melhor variável isolada",
                                       "Híbrido: logística + escore relacional")}
    best_single = []
    for tr, te in skf.split(np.zeros(n), y):
        known = np.zeros(n, bool); known[tr] = True
        vote = neighbor_vote(A, y, known)  # sem laços: o rótulo do próprio nó nunca é usado
        spread = label_spreading(A, y, known)
        scores["Grafo: voto dos vizinhos"][te] = vote[te]
        scores["Grafo: propagação de rótulos"][te] = spread[te]
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(onehot.iloc[tr], y[tr])
        scores["Tabular: regressão logística"][te] = lr.predict_proba(onehot.iloc[te])[:, 1]
        Xh = np.column_stack([onehot.to_numpy(float), vote])
        hy = LogisticRegression(max_iter=2000, C=1.0).fit(Xh[tr], y[tr])
        scores["Híbrido: logística + escore relacional"][te] = hy.predict_proba(Xh[te])[:, 1]
        # variável isolada: taxa de evasão por categoria do atributo (aprendida no treino)
        best, best_auc = None, -1
        for a in f.columns:
            rate = pd.Series(y[tr]).groupby(f[a].iloc[tr].fillna(-1).to_numpy()).mean()
            s_tr = f[a].iloc[tr].fillna(-1).map(rate).fillna(y[tr].mean())
            auc = roc_auc_score(y[tr], s_tr)
            if auc > best_auc:
                best, best_auc, best_rate = a, auc, rate
        best_single.append(best)
        scores["Tabular: melhor variável isolada"][te] = f[best].iloc[te].fillna(-1).map(best_rate).fillna(y[tr].mean())
    rows = [{"metodo": k, "AUC": roc_auc_score(y, s)} for k, s in scores.items()]
    out = pd.DataFrame(rows)
    out.attrs["melhor_variavel"] = pd.Series(best_single).mode()[0]
    out.attrs["scores"] = scores
    return out


def precision_at_k(score: np.ndarray, y: np.ndarray, frac: float = 0.2) -> float:
    k = int(len(y) * frac)
    top = np.argsort(-score)[:k]
    return float(y[top].mean())


# ------------------------------------------------------------ Modelo 2: fatores
def factor_centrality(F: nx.Graph) -> pd.DataFrame:
    strength = dict(F.degree(weight="weight"))
    btw = nx.betweenness_centrality(F, weight=lambda u, v, d: 1 / d["weight"])
    pr = nx.pagerank(F, weight="weight")
    df = pd.DataFrame({"grau": dict(F.degree()), "forca": strength, "intermediacao": btw, "pagerank": pr})
    df["tipo"] = pd.Series(nx.get_node_attributes(F, "kind"))
    return df.sort_values("pagerank", ascending=False)


def outcome_neighborhood(F: nx.Graph, outcome: str) -> pd.DataFrame:
    rows = [{"fator": v, "lift": d["lift"], "estudantes": d["count"]} for v, d in F[outcome].items()
            if F.nodes[v]["kind"] == "fator"]
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def synergy(onehot: pd.DataFrame, y: np.ndarray, factors: list[str], min_count: int = 40) -> pd.DataFrame:
    """Para pares de fatores ligados à evasão: taxa de evasão do par vs. de cada fator isolado."""
    base = y.mean()
    rows = []
    for i, a in enumerate(factors):
        for b in factors[i + 1:]:
            if a.split("=")[0] == b.split("=")[0]:
                continue
            m = (onehot[a] == 1) & (onehot[b] == 1)
            if m.sum() < min_count:
                continue
            ra, rb, rab = y[onehot[a] == 1].mean(), y[onehot[b] == 1].mean(), y[m.to_numpy()].mean()
            rows.append({"fator_A": a, "fator_B": b, "n": int(m.sum()), "evasao_A_%": 100 * ra,
                         "evasao_B_%": 100 * rb, "evasao_A∧B_%": 100 * rab,
                         "ganho_sobre_max_pp": 100 * (rab - max(ra, rb)),
                         "lift_A∧B": rab / base})
    return pd.DataFrame(rows).sort_values("ganho_sobre_max_pp", ascending=False).reset_index(drop=True)
