"""Explorador interativo dos modelos (demo do protótipo).

Uso:
    python -m src.main          # gera outputs/artefatos.pkl
    streamlit run app/explorador.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plots import OUT_COLOR, OUT_PT, OUT_COLOR_PT  # noqa: E402

st.set_page_config(page_title="Trajetórias em Rede", layout="wide")


@st.cache_resource
def load():
    with open(ROOT / "outputs/artefatos.pkl", "rb") as fh:
        d = pickle.load(fh)
    # escore em percentil (propagação de rótulos, fora da amostra na validação cruzada)
    d["risk_pct"] = pd.Series(d["risk"]).rank(pct=True).to_numpy() * 100
    return d


d = load()
G, F, f, schema, target, part = d["G"], d["F"], d["f"], d["schema"], d["target"], d["partition"]
t = target.to_numpy()


def describe(i: int) -> dict:
    return {a: (schema[a]["labels"][int(f.at[i, a])] if not np.isnan(f.at[i, a]) else "—") for a in f.columns}


st.title("Trajetórias Acadêmicas em Rede")
st.caption("SMAC03 – Grafos · UNIFEI · Dataset UCI 697 (4.424 estudantes)")

tab1, tab2, tab3 = st.tabs(["🎓 Estudante", "👥 Comunidades", "🔗 Rede de fatores"])

# ------------------------------------------------------------------ estudante
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        modo = st.radio("Escolher estudante", ["por número", "maior risco", "evasão 'inesperada'"], horizontal=True)
        if modo == "por número":
            i = int(st.number_input("Índice do estudante", 0, len(t) - 1, 0))
        elif modo == "maior risco":
            top = np.argsort(-d["risk_pct"])[:50]
            i = int(st.selectbox("Top 50 por escore de risco", top))
        else:
            surpr = [v for v in G if t[v] == "Dropout" and np.mean([t[u] == "Dropout" for u in G[v]]) <= 0.25]
            i = int(st.selectbox(f"{len(surpr)} evadidos com vizinhança majoritariamente não-evadida", surpr))

        nb = sorted(G[i].items(), key=lambda kv: -kv[1]["weight"])
        nb_out = pd.Series([t[u] for u, _ in nb]).map(OUT_PT).value_counts(normalize=True) * 100
        st.metric("Desfecho real", OUT_PT[t[i]])
        st.metric("Escore de risco relacional (percentil)", f"{d['risk_pct'][i]:.0f}")
        st.metric("Comunidade", f"C{part[i]}  ·  evasão {100*np.mean(t[part == part[i]] == 'Dropout'):.0f}%")
        st.write("**Vizinhos por desfecho (%)**")
        st.dataframe(nb_out.round(1).rename("%"))
    with c2:
        H = nx.ego_graph(G, i, radius=1)
        pos = nx.spring_layout(H, weight="weight", seed=1)
        fig, ax = plt.subplots(figsize=(7, 5))
        nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25, width=0.6)
        nx.draw_networkx_nodes(H, pos, ax=ax, node_color=[OUT_COLOR[t[v]] for v in H],
                               node_size=[500 if v == i else 120 for v in H], edgecolors="#0b0b0b",
                               linewidths=[2 if v == i else 0 for v in H])
        nx.draw_networkx_labels(H, pos, labels={v: str(v) for v in H}, font_size=6, ax=ax)
        ax.set_axis_off()
        ax.set_title(f"Rede ego do estudante {i} (cor = desfecho)")
        st.pyplot(fig)
    st.write("**Perfil do estudante e dos 5 vizinhos mais similares**")
    rows = {f"{i} (alvo)": describe(i)} | {f"{u} · {OUT_PT[t[u]]} · sim {e['weight']:.2f}": describe(u) for u, e in nb[:5]}
    st.dataframe(pd.DataFrame(rows), width="stretch")

# ------------------------------------------------------------------ comunidades
with tab2:
    prof = pd.read_csv(ROOT / "outputs/tabelas/m1_perfis_comunidades.csv")
    st.write("Comunidades Louvain do Modelo 1, ordenadas pela taxa de evasão (média institucional: 32,1%).")
    st.dataframe(prof.round(1), width="stretch")
    c = st.selectbox("Ver membros da comunidade", prof["comunidade"])
    membros = np.where(part == c)[0]
    st.write(f"{len(membros)} estudantes")
    st.dataframe(pd.DataFrame({v: describe(v) | {"Desfecho": OUT_PT[t[v]]} for v in membros[:200]}).T,
                 width="stretch")

# ------------------------------------------------------------------ fatores
with tab3:
    outcome = st.radio("Desfecho", ["Evasão", "Graduado", "Matriculado"], horizontal=True)
    viz = pd.read_csv(ROOT / f"outputs/tabelas/m2_vizinhanca_{outcome}.csv")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.write(f"**Fatores ligados a {outcome} (lift ≥ 1,15)**")
        st.dataframe(viz.round(2), width="stretch")
    with c2:
        fator = st.selectbox("Explorar a vizinhança de um fator", viz["fator"])
        H = F.subgraph([fator, outcome] + list(F[fator])).copy()
        pos = nx.spring_layout(H, seed=3, k=0.9)
        fig, ax = plt.subplots(figsize=(7, 6))
        nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.2)
        nx.draw_networkx_nodes(H, pos, ax=ax, node_size=250,
                               node_color=[OUT_COLOR_PT.get(v, "#b7b5ae") if v != fator else "#6250d6" for v in H])
        nx.draw_networkx_labels(H, pos, font_size=7, ax=ax)
        ax.set_axis_off()
        st.pyplot(fig)
    if outcome == "Evasão":
        st.write("**Sinergias: pares de fatores com evasão maior que a de cada fator isolado**")
        st.dataframe(pd.read_csv(ROOT / "outputs/tabelas/m2_sinergia_pares_evasao.csv").head(20).round(1),
                     width="stretch")
