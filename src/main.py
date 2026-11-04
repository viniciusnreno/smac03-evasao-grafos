"""Pipeline completo: dados -> grafos -> métricas -> tabelas e figuras em outputs/.

Uso:
    python -m src.main [--data data/data.csv] [--k 15] [--out outputs]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from . import analysis as an
from . import graphs as gr
from . import preprocess as pp
from . import plots


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def run(data: str, k: int, out: Path, weights: dict[str, float], skip_sens: bool = False) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    (out / "tabelas").mkdir(exist_ok=True)
    (out / "figuras").mkdir(exist_ok=True)

    raw = pp.load_raw(data)
    target = raw["Target"]
    f, schema = pp.build_features(raw)
    onehot = pp.onehot_factors(f, schema)
    y = (target == "Dropout").to_numpy().astype(float)
    log(f"dados: {len(f)} estudantes, {f.shape[1]} atributos, {onehot.shape[1]} fatores", t0)

    pd.DataFrame([{"atributo": a, "grupo": s["group"], "tipo": s["kind"], "categorias": " | ".join(s["labels"]),
                   "ausentes": int(f[a].isna().sum())} for a, s in schema.items()]
                 ).to_csv(out / "tabelas/dicionario_atributos.csv", index=False)

    # ------------------------------------------------------------ Modelo 1
    G = gr.build_similarity_graph(f, schema, target, weights=weights, k=k)
    summ = an.graph_summary(G)
    log(f"Modelo 1: {summ['vertices']} vértices, {summ['arestas']} arestas", t0)
    an.mixing_matrix(G).to_csv(out / "tabelas/m1_matriz_mistura.csv")

    comms = an.detect_communities(G)
    evals = pd.DataFrame({m: an.evaluate_partition(G, c, target.to_numpy()) for m, c in comms.items()}).T
    evals.to_csv(out / "tabelas/m1_comparacao_metodos_comunidades.csv")
    log("comunidades detectadas\n" + evals.round(3).to_string(), t0)

    best_method = evals["modularidade"].astype(float).idxmax()
    profiles = an.community_profiles(comms[best_method], target, onehot)
    profiles.to_csv(out / "tabelas/m1_perfis_comunidades.csv", index=False)

    surp, surp_frac = an.surprising_nodes(G, target)
    log(f"evadidos com vizinhança majoritariamente não-evadida: {100*surp_frac:.1f}%", t0)

    cv = an.cross_validate_risk(G, y, onehot, f)
    cv["precisao_top20_%"] = [100 * an.precision_at_k(s, y) for s in cv.attrs["scores"].values()]
    cv.to_csv(out / "tabelas/m1_validacao_risco.csv", index=False)
    log("validação\n" + cv.round(3).to_string(), t0)

    # cenário de alerta precoce: apenas atributos anteriores ao desempenho em disciplinas
    G_pre = gr.build_similarity_graph(f, schema, target, weights=weights, k=k, attrs=pp.ENTRY_ONLY)
    cv_pre = an.cross_validate_risk(G_pre, y, onehot[[c for c in onehot.columns
                                                      if c.split("=")[0] in pp.ENTRY_ONLY]],
                                    f[pp.ENTRY_ONLY])
    cv_pre["precisao_top20_%"] = [100 * an.precision_at_k(s, y) for s in cv_pre.attrs["scores"].values()]
    cv_pre.to_csv(out / "tabelas/m1_validacao_risco_ingresso.csv", index=False)
    summ_pre = an.graph_summary(G_pre)
    # perfis de ingresso quase idênticos com desfechos opostos
    div, n_div, n_same, thr = an.divergent_pairs(G_pre, f, schema, target)
    div.to_csv(out / "tabelas/m1_pares_divergentes.csv", index=False)
    log("pares divergentes\n" + div.round(1).to_string(), t0)
    log("validação (só ingresso)\n" + cv_pre.round(3).to_string(), t0)

    # sensibilidade aos pesos α, β, γ e a k
    sens = []
    for w in ({"ACAD": 1, "SOCIO": 0, "DEMO": 0}, {"ACAD": 0, "SOCIO": 1, "DEMO": 0},
              {"ACAD": 0, "SOCIO": 0, "DEMO": 1}, {"ACAD": 1 / 3, "SOCIO": 1 / 3, "DEMO": 1 / 3},
              weights, {"ACAD": 0.3, "SOCIO": 0.5, "DEMO": 0.2}):
        if skip_sens:
            break
        for kk in (10, 15, 25):
            Gs = gr.build_similarity_graph(f, schema, target, weights=w, k=kk)
            s = an.graph_summary(Gs)
            louv = an.nxc.louvain_communities(Gs, weight="weight", seed=an.SEED)
            A = an._adj(Gs, len(y))
            known = np.ones(len(y), bool)
            # AUC de voto dos vizinhos em validação cruzada simples (5 folds)
            vote = np.zeros(len(y))
            for tr, te in an.StratifiedKFold(5, shuffle=True, random_state=an.SEED).split(vote, y):
                kn = np.zeros(len(y), bool); kn[tr] = True
                vote[te] = an.neighbor_vote(A, y, kn)[te]
            sens.append({"alpha_ACAD": round(w["ACAD"], 2), "beta_SOCIO": round(w["SOCIO"], 2),
                         "gamma_DEMO": round(w["DEMO"], 2), "k": kk, "arestas": s["arestas"],
                         "assortatividade": s["assortatividade_desfecho"],
                         "modularidade_louvain": an.nxc.modularity(Gs, louv, weight="weight"),
                         "AUC_voto": an.roc_auc_score(y, vote)})
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "tabelas/m1_sensibilidade_pesos_k.csv", index=False)
    log("sensibilidade\n" + sens.round(3).to_string(), t0)

    # ------------------------------------------------------------ Modelo 2
    target_pt = target.map(pp.OUTCOME_PT)
    F = gr.build_factor_graph(onehot, target_pt)
    fsum = an.graph_summary(F, attr=None)
    log(f"Modelo 2: {fsum['vertices']} vértices, {fsum['arestas']} arestas", t0)
    cent = an.factor_centrality(F)
    cent.to_csv(out / "tabelas/m2_centralidade_fatores.csv")
    fcomms = an.nxc.louvain_communities(F, weight="weight", seed=an.SEED)
    fcomm_rows = [{"comunidade_fatores": i, "tamanho": len(c), "contem_desfecho": ", ".join(
        sorted(v for v in c if F.nodes[v]["kind"] == "desfecho")), "fatores": "; ".join(sorted(c))}
        for i, c in enumerate(sorted(fcomms, key=len, reverse=True))]
    pd.DataFrame(fcomm_rows).to_csv(out / "tabelas/m2_comunidades_fatores.csv", index=False)
    egos = {o: an.outcome_neighborhood(F, o) for o in ("Evasão", "Graduado", "Matriculado") if o in F}
    for o, df in egos.items():
        df.to_csv(out / f"tabelas/m2_vizinhanca_{o}.csv", index=False)
    syn = an.synergy(onehot, y, egos["Evasão"]["fator"].tolist())
    syn.to_csv(out / "tabelas/m2_sinergia_pares_evasao.csv", index=False)
    log("vizinhança de Evasão\n" + egos["Evasão"].head(15).round(2).to_string(), t0)
    log("sinergias\n" + syn.head(10).round(1).to_string(), t0)

    # ------------------------------------------------------------ figuras
    plots.all_figures(out / "figuras", G=G, comms=comms[best_method], target=target, profiles=profiles,
                      evals=evals, cv=cv, cv_pre=cv_pre, div=div, F=F, egos=egos, fcomms=fcomms,
                      sens=sens, mixing=an.mixing_matrix(G))
    log("figuras geradas", t0)

    # artefatos para o explorador interativo
    with open(out / "artefatos.pkl", "wb") as fh:
        pickle.dump({"G": G, "F": F, "f": f, "schema": schema, "target": target, "onehot": onehot,
                     "partition": an.partition_vector(comms[best_method], len(y)),
                     "risk": cv.attrs["scores"]["Grafo: propagação de rótulos"]}, fh)

    resumo = {"pesos": weights, "k": k, "modelo1": summ, "modelo1_ingresso": summ_pre, "modelo2": fsum,
              "metodo_comunidades_escolhido": best_method,
              "pares_divergentes": {"limiar_similaridade": float(thr), "evasao_x_graduado": n_div,
                                    "graduado_x_graduado": n_same},
              "evadidos_vizinhanca_nao_evadida_%": 100 * surp_frac,
              "melhor_variavel_isolada": cv.attrs["melhor_variavel"],
              "melhor_variavel_isolada_ingresso": cv_pre.attrs["melhor_variavel"]}
    (out / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=float))
    log("resumo\n" + json.dumps(resumo, indent=2, ensure_ascii=False, default=float), t0)
    return resumo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/data.csv")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--alpha", type=float, default=0.5, help="peso do grupo acadêmico")
    ap.add_argument("--beta", type=float, default=0.3, help="peso do grupo socioeconômico")
    ap.add_argument("--gamma", type=float, default=0.2, help="peso do grupo demográfico")
    ap.add_argument("--rapido", action="store_true", help="pula a análise de sensibilidade (mais lenta)")
    a = ap.parse_args()
    run(a.data, a.k, Path(a.out), {"ACAD": a.alpha, "SOCIO": a.beta, "DEMO": a.gamma}, skip_sens=a.rapido)


if __name__ == "__main__":
    main()
