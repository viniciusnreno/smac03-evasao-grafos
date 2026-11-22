"""Pipeline completo. Classe: ``Pipeline``.

Responsável: Carlos.

Constrói e executa os modelos G1, G2 e G3 (todos ``ModeloGrafo``) e
grava os resultados em outputs/. Segue o protocolo 80/20 do relatório:

1. ``BaseDeDados``: remove os Enrolled e faz a divisão oficial
   (2.904 treino / 726 teste, semente 42), gravada em ``split.csv``.
2. Hiperparâmetros fixos, escolhidos antes por CV de 5 dobras só no
   treino (``K_PADRAO``, λ = 1,0, taxas derivadas incluídas).
3. G1: kNN de Gower + corte mínimo (Edmonds-Karp) classifica o teste,
   que é usado **uma única vez**. Assortatividade do desfecho (H5) no
   kNN só de treino, com teste de permutação.
4. G2: árvore geradora máxima de informação mútua (Kruskal) no treino,
   estabilidade por bootstrap e replicação da árvore no teste (H2).
5. G3: DAG de trajetórias (Kahn + Bellman-Ford), se já implementado.
6. Avaliação (``Avaliador``, ``avaliacao.py``): CV no treino, teste
   com IC por bootstrap, AUC, McNemar com Holm, ablação, eliminação
   backward e robustez em 10 divisões. O ``Avaliador`` devolve as
   tabelas e o ``Pipeline`` grava.
7. Conferências com o networkx (só asserts) e gravação dos arquivos.

As figuras ficam no ``GeradorFiguras`` (``figuras.py``), que lê os
arquivos gravados aqui.

Uso::

    python src/main.py            # completo (~30 min)
    python src/main.py --rapido   # sem ablação, backward e robustez
"""

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from arvore_dependencias import DESFECHO, N_BOOTSTRAP, ArvoreDependencias
from avaliacao import Avaliador
from corte_minimo import ClassificadorCorteMinimo
from dados import SEMENTE, BaseDeDados
from grafo_similaridade import K_PADRAO, GrafoSimilaridade

try:
    from dag_trajetorias import DagTrajetorias
except ImportError:          # G3 ainda não implementado
    DagTrajetorias = None

SAIDA_PADRAO = Path(__file__).resolve().parent.parent / "outputs"
MOMENTOS = ("T0", "T1", "T2")
DOBRAS_CV = 5

# Assortatividade (H5): kNN só de treino, k = 10, pesos iguais e sem as
# taxas derivadas, como no analise_v2; 200 permutações dos rótulos.
K_ASSORTATIVIDADE = 10
N_PERMUTACOES = 200

# Parâmetros do protocolo executados pelo Avaliador (registrados na
# Tab. 2 do relatório).
N_BOOTSTRAP_IC = 1000
N_SPLITS_ROBUSTEZ = 10

TOLERANCIA = 1e-9


class Pipeline:
    """Orquestra os modelos de grafo e grava os resultados.

    Parâmetros:
        saida: pasta dos resultados (``outputs/``).
        momentos: momentos da trajetória a rodar.
        n_bootstrap_arvore: reamostragens da estabilidade do G2.
        n_permutacoes: permutações do teste de assortatividade.
        verificar: confere G2 e assortatividade com o networkx.
        avaliacao_completa: inclui ablação, eliminação backward e
            robustez (as partes demoradas da avaliação).
        progresso: função que recebe mensagens de andamento (``print``
            na linha de comando; a interface gráfica pode passar outra).
    """

    def __init__(self, saida: Path | str = SAIDA_PADRAO,
                 momentos: tuple[str, ...] = MOMENTOS,
                 n_bootstrap_arvore: int = N_BOOTSTRAP,
                 n_permutacoes: int = N_PERMUTACOES,
                 verificar: bool = True,
                 avaliacao_completa: bool = True,
                 progresso: Callable[[str], None] | None = None) -> None:
        self.saida = Path(saida)
        self.momentos = tuple(momentos)
        self.n_bootstrap_arvore = n_bootstrap_arvore
        self.n_permutacoes = n_permutacoes
        self.verificar = verificar
        self.avaliacao_completa = avaliacao_completa
        self._progresso = progresso or (lambda _msg: None)
        self.base: BaseDeDados | None = None
        self.resultados: dict = {}

    # ------------------------------------------------------------------
    # Execução
    # ------------------------------------------------------------------
    def executar(self) -> dict:
        """Roda o protocolo completo e devolve o dict de resultados."""
        inicio = time.perf_counter()
        self._avisar("Carregando dados e divisão 80/20 (semente 42)")
        self.base = BaseDeDados()
        self.resultados = {
            "dados": self._resumo_dados(),
            "hiperparametros": self._hiperparametros(),
        }
        self.saida.mkdir(parents=True, exist_ok=True)
        self._gravar_split()

        self.resultados["g1"] = self._executar_g1()
        self.resultados["assortatividade"] = self._assortatividade()
        self.resultados["g2"] = self._executar_g2()
        self.resultados["g3"] = self._executar_g3()
        self.resultados["avaliacao"] = self._avaliar()
        self.resultados["tempo_s"] = round(time.perf_counter() - inicio, 1)
        self._gravar_json("resultados.json", self.resultados)
        self._gravar_json("hiperparametros.json",
                          self.resultados["hiperparametros"])
        self._avisar(f"Concluído em {self.resultados['tempo_s']} s; "
                     f"resultados em {self.saida}")
        return self.resultados

    # ------------------------------------------------------------------
    # G1 · similaridade + corte mínimo
    # ------------------------------------------------------------------
    def _executar_g1(self) -> dict:
        """Classifica o teste em cada momento pelo corte mínimo."""
        base = self.base
        y_teste = base.y(base.idx_teste)
        previsoes = pd.DataFrame({"aluno": base.idx_teste,
                                  "evadiu": y_teste})
        metricas, saida = [], {}
        for m in self.momentos:
            self._avisar(f"G1 {m}: kNN de Gower (k = {K_PADRAO[m]}) e "
                         "corte mínimo (Edmonds-Karp)")
            modelo = GrafoSimilaridade(m, K_PADRAO[m])
            modelo.construir(base)
            r = modelo.executar()
            y_prev = np.array([r["previsoes"][i] for i in base.idx_teste])
            previsoes[m] = y_prev
            met = self.metricas(y_teste, y_prev)
            metricas.append({"momento": m, "metodo": "corte_minimo", **met})
            metricas.append({"momento": m, "metodo": "classe_majoritaria",
                             **self.metricas(y_teste,
                                             np.zeros_like(y_teste))})
            saida[m] = {
                "k": r["k"], "lam": r["lam"], "fluxo": r["fluxo"],
                "num_caminhos": r["num_caminhos"],
                "num_vertices": r["num_vertices"],
                "num_arestas": r["num_arestas"],
                "num_arestas_cortadas": len(modelo.destaques()["arestas"]),
                "pesos": dict(sorted(r["pesos"].items(),
                                     key=lambda p: -p[1])),
                "teste": met,
            }
        previsoes.to_csv(self.saida / "g1_previsoes_teste.csv", index=False)
        pd.DataFrame(metricas).to_csv(self.saida / "g1_metricas_teste.csv",
                                      index=False)
        return saida

    def _assortatividade(self) -> dict:
        """Assortatividade do desfecho no kNN de treino (H5)."""
        base = self.base
        rng = np.random.default_rng(SEMENTE)
        linhas, saida = [], {}
        for m in self.momentos:
            self._avisar(f"H5 {m}: assortatividade com "
                         f"{self.n_permutacoes} permutações")
            colunas = base.atributos(m)
            modelo = GrafoSimilaridade(
                m, K_ASSORTATIVIDADE, pesos=dict.fromkeys(colunas, 1.0),
                incluir_derivados=False)
            modelo.construir(base, idx_teste=[])
            r = modelo.assortatividade()
            alunos = list(modelo.rotulos_)
            valores = np.array(list(modelo.rotulos_.values()))
            nulos = np.array([
                modelo.assortatividade(
                    dict(zip(alunos, rng.permutation(valores))))
                for _ in range(self.n_permutacoes)])
            p = (1 + int((nulos >= r).sum())) / (self.n_permutacoes + 1)
            if self.verificar:
                self._conferir_assortatividade(modelo, r)
            saida[m] = {"r": r, "p_permutacao": p,
                        "max_permutacao": float(nulos.max()),
                        "num_arestas": modelo.grafo.num_arestas()}
            linhas.append({"momento": m, **saida[m]})
        pd.DataFrame(linhas).to_csv(self.saida / "g1_assortatividade.csv",
                                    index=False)
        return saida

    # ------------------------------------------------------------------
    # G2 · árvore de dependências
    # ------------------------------------------------------------------
    def _executar_g2(self) -> dict:
        """Kruskal no treino, estabilidade e replicação no teste."""
        base = self.base
        im_desfecho, saida = {}, {}
        for m in self.momentos:
            self._avisar(f"G2 {m}: árvore máxima de IM (Kruskal) e "
                         f"{self.n_bootstrap_arvore} bootstraps")
            modelo = ArvoreDependencias(m)
            modelo.construir(base)
            r = modelo.executar()
            if self.verificar:
                self._conferir_kruskal(modelo, r)
            estab = modelo.estabilidade(base, n=self.n_bootstrap_arvore)

            # Replicação: a mesma construção com os alunos do teste.
            # Só mede; nenhuma escolha depende disto.
            modelo_teste = ArvoreDependencias(m)
            modelo_teste.construir(base, idx=base.idx_teste)
            no_teste = ArvoreDependencias.chaves(
                modelo_teste.executar()["arestas"])

            linhas = [{
                "u": u, "v": v, "im_bits": im,
                "toca_desfecho": DESFECHO in (u, v),
                "estabilidade": estab[ArvoreDependencias._ordenar(u, v)],
                "no_teste": ArvoreDependencias._ordenar(u, v) in no_teste,
            } for u, v, im in r["arestas"]]
            pd.DataFrame(linhas).to_csv(self.saida / f"g2_arvore_{m}.csv",
                                        index=False)
            im_desfecho[m] = r["im_desfecho"]
            saida[m] = {
                "num_vertices": r["num_vertices"],
                "num_arestas_arvore": len(r["arestas"]),
                "peso_total": r["peso_total"],
                "vizinhos_desfecho": [
                    {"atributo": v, "im_bits": im,
                     "estabilidade": estab[
                         ArvoreDependencias._ordenar(DESFECHO, v)]}
                    for v, im in r["vizinhos_desfecho"]],
                "sobreposicao_treino_teste": sum(
                    linha["no_teste"] for linha in linhas) / len(linhas),
                "estabilidade_media": float(np.mean(list(estab.values()))),
            }
        pd.DataFrame(im_desfecho).rename_axis("atributo").to_csv(
            self.saida / "g2_im_desfecho.csv")
        return saida

    # ------------------------------------------------------------------
    # G3 · DAG de trajetórias
    # ------------------------------------------------------------------
    def _executar_g3(self) -> dict:
        """Roda o G3 em cada momento, se ``DagTrajetorias`` existir.

        Supõe ``DagTrajetorias(momento)`` e, em ``executar()``, a chave
        opcional ``previsoes`` (``{aluno: bool}`` do teste), como no G1.
        """
        if DagTrajetorias is None:
            self._avisar("G3: dag_trajetorias.py ainda não implementado")
            return {"status": "não implementado"}
        base = self.base
        y_teste = base.y(base.idx_teste)
        saida = {}
        for m in self.momentos:
            self._avisar(f"G3 {m}: DAG de trajetórias (Kahn + "
                         "Bellman-Ford)")
            try:
                modelo = DagTrajetorias(m)
                modelo.construir(base)
                r = modelo.executar()
            except (TypeError, NotImplementedError) as erro:
                saida[m] = {"status": f"erro: {erro}"}
                continue
            if self.verificar and modelo.grafo is not None:
                self._conferir_dag(modelo)
            resumo = {c: v for c, v in r.items() if c != "previsoes"}
            if "previsoes" in r:
                y_prev = np.array([r["previsoes"][i]
                                   for i in base.idx_teste])
                resumo["teste"] = self.metricas(y_teste, y_prev)
            saida[m] = resumo
        return saida

    # ------------------------------------------------------------------
    # Avaliação
    # ------------------------------------------------------------------
    def _avaliar(self) -> dict:
        """Roda o ``Avaliador`` e grava as tabelas ``avaliacao_*``."""
        completa = self.avaliacao_completa
        r = Avaliador().executar(
            self.base, self.momentos, ablacao=completa, backward=completa,
            robustez=completa, progresso=self._avisar)
        for nome in ("cv", "teste", "mcnemar", "previsoes", "ablacao",
                     "robustez"):
            if nome in r:
                r[nome].to_csv(self.saida / f"avaliacao_{nome}.csv",
                               index=False)
        resumo = {
            "regras_1_variavel": r["regras"],
            "teste": r["teste"].to_dict(orient="records"),
            "mcnemar": r["mcnemar"].to_dict(orient="records"),
            "cv": r["cv"].to_dict(orient="records"),
        }
        if "backward" in r:
            linhas = [{"momento": m, **h} for m, res in r["backward"].items()
                      for h in res["historico"]]
            pd.DataFrame(linhas).to_csv(
                self.saida / "avaliacao_backward.csv", index=False)
            resumo["backward"] = {
                m: {"mantidos": res["mantidos"],
                    "acuracia_cv": res["acuracia_cv"]}
                for m, res in r["backward"].items()}
        if "ablacao" in r:
            resumo["ablacao"] = r["ablacao"].to_dict(orient="records")
        if "robustez" in r:
            resumo["robustez"] = r["robustez"].to_dict(orient="records")
        return resumo

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------
    @staticmethod
    def metricas(y_real: np.ndarray, y_prev: np.ndarray) -> dict:
        """Métricas de classificação; evasão (True) é a classe positiva.

        Acurácia balanceada = (recall da evasão + especificidade) / 2,
        a métrica principal (a acurácia simples premia "sempre forma").
        """
        y_real = np.asarray(y_real, dtype=bool)
        y_prev = np.asarray(y_prev, dtype=bool)
        vp = int((y_real & y_prev).sum())
        fp = int((~y_real & y_prev).sum())
        fn = int((y_real & ~y_prev).sum())
        vn = int((~y_real & ~y_prev).sum())
        recall = vp / (vp + fn) if vp + fn else 0.0
        especificidade = vn / (vn + fp) if vn + fp else 0.0
        precisao = vp / (vp + fp) if vp + fp else 0.0
        f1 = (2 * precisao * recall / (precisao + recall)
              if precisao + recall else 0.0)
        return {
            "acuracia_balanceada": (recall + especificidade) / 2,
            "recall_evasao": recall,
            "especificidade": especificidade,
            "precisao_evasao": precisao,
            "f1_evasao": f1,
            "acuracia": (vp + vn) / len(y_real),
            "vp": vp, "fp": fp, "fn": fn, "vn": vn,
        }

    # ------------------------------------------------------------------
    # Conferências com o networkx (só asserts)
    # ------------------------------------------------------------------
    @staticmethod
    def _conferir_kruskal(modelo: ArvoreDependencias, r: dict) -> None:
        """Kruskal próprio × ``maximum_spanning_tree`` do networkx.

        Empates de IM (os três atributos macro são o mesmo código de
        coorte) admitem árvores diferentes; a garantia comparada é o
        peso total máximo e a mesma lista ordenada de pesos.
        """
        import networkx as nx

        arvore_nx = nx.maximum_spanning_tree(modelo.grafo.para_networkx())
        pesos_nx = sorted(d["weight"] for *_, d in
                          arvore_nx.edges(data=True))
        pesos = sorted(p for _, _, p in r["arestas"])
        assert len(pesos) == len(modelo.grafo) - 1, "não é árvore"
        assert np.allclose(pesos, pesos_nx, atol=TOLERANCIA), \
            "Kruskal difere do networkx"

    @staticmethod
    def _conferir_assortatividade(modelo: GrafoSimilaridade,
                                  r: float) -> None:
        """Assortatividade própria × networkx."""
        import networkx as nx

        g = modelo.grafo.para_networkx()
        nx.set_node_attributes(g, modelo.rotulos_, "evadiu")
        r_nx = nx.attribute_assortativity_coefficient(g, "evadiu")
        assert abs(r - r_nx) < 1e-6, "assortatividade difere do networkx"

    @staticmethod
    def _conferir_dag(modelo) -> None:
        """O grafo do G3 precisa ser acíclico (o Kahn garante a ordem)."""
        import networkx as nx

        assert nx.is_directed_acyclic_graph(modelo.grafo.para_networkx()), \
            "o grafo do G3 tem ciclo"

    # ------------------------------------------------------------------
    # Resumos e gravação
    # ------------------------------------------------------------------
    def _resumo_dados(self) -> dict:
        base = self.base
        return {
            "alunos": len(base),
            "treino": len(base.idx_treino),
            "teste": len(base.idx_teste),
            "taxa_evasao": base.taxa_evasao(),
            "taxa_evasao_treino": base.taxa_evasao(base.idx_treino),
            "taxa_evasao_teste": base.taxa_evasao(base.idx_teste),
        }

    def _hiperparametros(self) -> dict:
        """Parâmetros do protocolo (Tab. 2 do relatório)."""
        return {
            "divisao": {"treino": 1 - self.base.fracao_teste,
                        "teste": self.base.fracao_teste,
                        "estratificada": True, "semente": SEMENTE},
            "dobras_cv_treino": DOBRAS_CV,
            "g1": {
                "k": {m: K_PADRAO[m] for m in self.momentos},
                "lam": ClassificadorCorteMinimo().lam,
                "pesos_atributos": "informação mútua com o desfecho "
                                   "(treino)",
                "capacidade": "c(i,j) = 1 - d_Gower(i,j)",
                "taxas_derivadas": True,
            },
            "assortatividade": {"k": K_ASSORTATIVIDADE,
                                "pesos": "iguais",
                                "permutacoes": self.n_permutacoes},
            "g2": {"peso": "informação mútua (bits, treino)",
                   "algoritmo": "Kruskal decrescente + union-find",
                   "bootstrap_estabilidade": self.n_bootstrap_arvore},
            "avaliacao": {"bootstrap_ic95": N_BOOTSTRAP_IC,
                          "splits_robustez": N_SPLITS_ROBUSTEZ},
        }

    def _gravar_split(self) -> None:
        base = self.base
        split = pd.DataFrame({
            "aluno": base.idx_treino + base.idx_teste,
            "conjunto": (["treino"] * len(base.idx_treino)
                         + ["teste"] * len(base.idx_teste)),
        }).sort_values("aluno")
        split.to_csv(self.saida / "split.csv", index=False)

    def _gravar_json(self, nome: str, conteudo: dict) -> None:
        with open(self.saida / nome, "w", encoding="utf-8") as arquivo:
            json.dump(self._para_json(conteudo), arquivo,
                      ensure_ascii=False, indent=2)

    @classmethod
    def _para_json(cls, valor):
        """Converte tipos do numpy, tuplas e conjuntos para JSON."""
        if isinstance(valor, dict):
            return {str(c): cls._para_json(v) for c, v in valor.items()}
        if isinstance(valor, (list, tuple, set)):
            return [cls._para_json(v) for v in valor]
        if isinstance(valor, np.generic):
            return valor.item()
        if isinstance(valor, float):
            return round(valor, 6)
        return valor

    def _avisar(self, mensagem: str) -> None:
        self._progresso(mensagem)


def main() -> None:
    """Ponto de entrada da linha de comando."""
    rapido = "--rapido" in sys.argv[1:]
    resultados = Pipeline(avaliacao_completa=not rapido,
                          progresso=print).executar()
    print("\nAcurácia balanceada no teste (IC 95%):")
    for linha in resultados["avaliacao"]["teste"]:
        print(f"  {linha['momento']} {linha['metodo']:<20} "
              f"{linha['acuracia_balanceada']:.3f} "
              f"({linha['ic_inf']:.3f}–{linha['ic_sup']:.3f})")


if __name__ == "__main__":
    main()
