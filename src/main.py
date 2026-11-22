"""Pipeline completo: roda os três grafos e grava os resultados.

1. Carrega os dados e separa treino e teste (80/20, semente 42).
2. G1: grafo de alunos parecidos, classificado pelo corte mínimo
   (Edmonds-Karp). O teste é usado uma única vez.
3. G2: árvore de dependências entre fatores (Kruskal).
4. G3: DAG de trajetórias (Kahn + Bellman-Ford).
5. Avaliação estatística (``Avaliador``) e gravação em ``outputs/``.

Uso::

    python src/main.py            # completo
    python src/main.py --rapido   # sem ablação, backward e robustez
"""

import json
import os
import platform
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from arvore_dependencias import DESFECHO, N_BOOTSTRAP, ArvoreDependencias
from avaliacao import DOBRAS, N_SPLITS, Avaliador
from avaliacao import N_BOOTSTRAP as N_BOOTSTRAP_IC
from corte_minimo import ClassificadorCorteMinimo
from dados import SEMENTE, BaseDeDados
from dag_trajetorias import (CAMADAS_POR_MOMENTO, MIN_SUPORTE, SUAVIZACAO,
                             DagTrajetorias)
from grafo_similaridade import K_PADRAO, GrafoSimilaridade

SAIDA_PADRAO = Path(__file__).resolve().parent.parent / "outputs"
MOMENTOS = ("T0", "T1", "T2")

# Assortatividade: o quanto alunos ligados no grafo têm o mesmo
# desfecho. Medida num kNN só de treino, com pesos iguais, e comparada
# com rótulos embaralhados (teste de permutação).
K_ASSORTATIVIDADE = 10
N_PERMUTACOES = 200

TOLERANCIA = 1e-9

# Bibliotecas cujas versões vão para o registro do ambiente.
BIBLIOTECAS = ("numpy", "pandas", "scikit-learn", "scipy", "networkx",
               "matplotlib")


class Pipeline:
    """Executa os modelos de grafo e grava os resultados.

    Parâmetros:
        saida: pasta dos resultados.
        momentos: momentos da trajetória a rodar.
        n_bootstrap_arvore: reamostragens da estabilidade do G2.
        n_permutacoes: permutações do teste de assortatividade.
        verificar: confere os resultados com o networkx.
        avaliacao_completa: inclui as partes demoradas da avaliação.
        progresso: recebe as mensagens de andamento (ex.: ``print``).
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
        self._avisar = progresso or (lambda _msg: None)
        self.base: BaseDeDados | None = None
        self.resultados: dict = {}
        self._tempos: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Execução
    # ------------------------------------------------------------------
    def executar(self) -> dict:
        """Roda todas as etapas e devolve o dict de resultados."""
        inicio = time.perf_counter()
        self._tempos = {}
        self._avisar("Carregando dados e divisão 80/20 (semente 42)")
        self.base = self._cronometrar("dados", BaseDeDados)
        self.resultados = {
            "dados": self._resumo_dados(),
            "hiperparametros": self._hiperparametros(),
        }
        self.saida.mkdir(parents=True, exist_ok=True)
        self._gravar_split()

        self.resultados["g1"] = self._cronometrar("g1", self._executar_g1)
        self.resultados["assortatividade"] = self._cronometrar(
            "assortatividade", self._assortatividade)
        self.resultados["g2"] = self._cronometrar("g2", self._executar_g2)
        self.resultados["g3"] = self._cronometrar("g3", self._executar_g3)
        self.resultados["avaliacao"] = self._cronometrar(
            "avaliacao", self._avaliar)
        self._tempos["total"] = time.perf_counter() - inicio

        ambiente = self.ambiente()
        ambiente["avaliacao_completa"] = self.avaliacao_completa
        ambiente["tempo_s"] = {etapa: round(s, 1)
                               for etapa, s in self._tempos.items()}
        self.resultados["hiperparametros"]["ambiente"] = ambiente
        self.resultados["tempo_s"] = ambiente["tempo_s"]["total"]
        self._gravar_json("resultados.json", self.resultados)
        self._gravar_json("hiperparametros.json",
                          self.resultados["hiperparametros"])
        self._avisar(f"Concluído em {self.resultados['tempo_s']} s; "
                     f"resultados em {self.saida}")
        return self.resultados

    def _cronometrar(self, etapa: str, funcao: Callable):
        """Executa ``funcao`` e guarda quanto tempo a etapa levou."""
        inicio = time.perf_counter()
        resultado = funcao()
        self._tempos[etapa] = time.perf_counter() - inicio
        return resultado

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
            met = Avaliador.metricas(y_teste, y_prev)
            metricas.append({"momento": m, "metodo": "corte_minimo", **met})
            # Referência: prever que todo aluno se forma.
            metricas.append({"momento": m, "metodo": "classe_majoritaria",
                             **Avaliador.metricas(y_teste,
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
        """Mede se alunos vizinhos no grafo tendem a ter o mesmo desfecho.

        O valor real é comparado com o de rótulos embaralhados; o p-valor
        é a fração de embaralhamentos que chegam ao valor real.
        """
        base = self.base
        rng = np.random.default_rng(SEMENTE)
        linhas, saida = [], {}
        for m in self.momentos:
            self._avisar(f"Assortatividade {m}: "
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
        """Monta a árvore no treino e mede sua estabilidade.

        A árvore também é refeita com os alunos do teste, só para medir
        se ela se repete; nenhuma escolha depende disso.
        """
        base = self.base
        par = ArvoreDependencias.ordenar_par
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

            modelo_teste = ArvoreDependencias(m)
            modelo_teste.construir(base, idx=base.idx_teste)
            no_teste = ArvoreDependencias.chaves(
                modelo_teste.executar()["arestas"])

            linhas = [{
                "u": u, "v": v, "im_bits": im,
                "toca_desfecho": DESFECHO in (u, v),
                "estabilidade": estab[par(u, v)],
                "no_teste": par(u, v) in no_teste,
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
                     "estabilidade": estab[par(DESFECHO, v)]}
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
        """Monta o DAG, acha as trajetórias e classifica o teste."""
        base = self.base
        y_teste = base.y(base.idx_teste)
        previsoes = pd.DataFrame({"aluno": base.idx_teste,
                                  "evadiu": y_teste})
        metricas, saida = [], {}
        for m in self.momentos:
            self._avisar(f"G3 {m}: DAG de trajetórias (Kahn + "
                         "Bellman-Ford)")
            modelo = DagTrajetorias(m)
            modelo.construir(base)
            r = modelo.executar()
            if self.verificar:
                self._conferir_dag(modelo)
            # Previsão e risco por aluno vão para o CSV, não para o JSON.
            resumo = {c: v for c, v in r.items()
                      if c not in ("previsoes", "risco")}
            y_prev = np.array([r["previsoes"][i] for i in base.idx_teste])
            previsoes[m] = y_prev
            previsoes[f"risco_{m}"] = [r["risco"][i] for i in base.idx_teste]
            resumo["teste"] = Avaliador.metricas(y_teste, y_prev)
            metricas.append({"momento": m, "metodo": "caminho_dag",
                             **resumo["teste"]})
            saida[m] = resumo
        previsoes.to_csv(self.saida / "g3_previsoes_teste.csv", index=False)
        pd.DataFrame(metricas).to_csv(self.saida / "g3_metricas_teste.csv",
                                      index=False)
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
    # Conferências com o networkx (só asserts)
    # ------------------------------------------------------------------
    @staticmethod
    def _conferir_kruskal(modelo: ArvoreDependencias, r: dict) -> None:
        """Compara o nosso Kruskal com o do networkx.

        Com pesos empatados há mais de uma árvore válida, então o que se
        compara é o peso de cada aresta escolhida, não as arestas.
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
        """Compara a nossa assortatividade com a do networkx."""
        import networkx as nx

        g = modelo.grafo.para_networkx()
        nx.set_node_attributes(g, modelo.rotulos_, "evadiu")
        r_nx = nx.attribute_assortativity_coefficient(g, "evadiu")
        assert abs(r - r_nx) < 1e-6, "assortatividade difere do networkx"

    @staticmethod
    def _conferir_dag(modelo: DagTrajetorias) -> None:
        """Confirma com o networkx que o grafo do G3 não tem ciclo."""
        import networkx as nx

        assert nx.is_directed_acyclic_graph(modelo.grafo.para_networkx()), \
            "o grafo do G3 tem ciclo"

    # ------------------------------------------------------------------
    # Registro dos parâmetros e do ambiente
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
        """Parâmetros usados em cada etapa."""
        return {
            "divisao": {"treino": 1 - self.base.fracao_teste,
                        "teste": self.base.fracao_teste,
                        "estratificada": True, "semente": SEMENTE},
            "dobras_cv_treino": DOBRAS,
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
            "g3": {
                "camadas_por_momento": {m: CAMADAS_POR_MOMENTO[m]
                                        for m in self.momentos},
                "min_suporte": MIN_SUPORTE,
                "suavizacao_laplace": SUAVIZACAO,
                "peso_bellman_ford": "-ln(taxa de evasão do arco)",
                "limiar": "taxa de evasão do treino",
            },
            "avaliacao": {"bootstrap_ic95": N_BOOTSTRAP_IC,
                          "splits_robustez": N_SPLITS},
        }

    @staticmethod
    def ambiente() -> dict:
        """Versões, sistema e hardware (sem nome de usuário ou máquina)."""
        bibliotecas = {}
        for nome in BIBLIOTECAS:
            try:
                bibliotecas[nome] = metadata.version(nome)
            except metadata.PackageNotFoundError:
                bibliotecas[nome] = None
        return {
            "python": platform.python_version(),
            "bibliotecas": bibliotecas,
            "sistema": f"{platform.system()} {platform.release()}",
            "processador": Pipeline._processador(),
            "nucleos_logicos": os.cpu_count(),
            "memoria_gb": Pipeline._memoria_gb(),
        }

    @staticmethod
    def _processador() -> str:
        """Modelo da CPU (lido de ``/proc/cpuinfo`` no Linux)."""
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as arquivo:
                for linha in arquivo:
                    if linha.startswith("model name"):
                        return linha.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or platform.machine()

    @staticmethod
    def _memoria_gb() -> float | None:
        """Memória RAM total em GB, quando o sistema informa."""
        try:
            paginas = os.sysconf("SC_PHYS_PAGES")
            tamanho = os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, ValueError, OSError):
            return None
        return round(paginas * tamanho / 1024 ** 3, 1)

    # ------------------------------------------------------------------
    # Gravação
    # ------------------------------------------------------------------
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
