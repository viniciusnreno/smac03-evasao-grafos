"""Demonstrações passo a passo do Edmonds–Karp e do Kruskal.

As duas classes rodam os algoritmos do projeto (não os reimplementam)
e devolvem a lista de passos que a interface anima. Não dependem do
Streamlit.

- ``DemoEdmondsKarp``: rede pequena com alunos reais do treino; usa o
  ``ClassificadorCorteMinimo`` com ``registrar=True`` e refaz a rede
  residual depois de cada caminho aumentante.
- ``DemoKruskal``: lê a árvore devolvida por
  ``ArvoreDependencias.kruskal`` e reconstitui a ordem em que as
  arestas foram examinadas. A decisão (aceita ou rejeita) vem da saída
  oficial; o ``UniaoBusca`` só acompanha os componentes para colorir o
  desenho e confirmar que cada rejeitada fecharia um ciclo.
"""

from collections.abc import Hashable, Mapping

import numpy as np

from arvore_dependencias import ArvoreDependencias, UniaoBusca
from corte_minimo import FONTE, ClassificadorCorteMinimo
from dados import SEMENTE
from grafo import Grafo
from grafo_similaridade import GrafoSimilaridade

Residual = dict[Hashable, dict[Hashable, float]]


class DemoEdmondsKarp:
    """Edmonds–Karp (aula 6) numa rede pequena do G1, passo a passo.

    Sorteia (semente 42) alunos do **treino**: metade dos rotulados
    evadiu e metade formou; os "sem rótulo" também vêm do treino, então
    o desfecho real deles pode ser mostrado sem tocar no teste. A rede
    é montada sem contração (``contrair=False``), para S e T aparecerem
    ligados aos rotulados por arcos de capacidade infinita.

    Garantia mostrada: cada caminho aumentante é o mais curto em número
    de arcos (BFS); o fluxo máximo é igual à capacidade do corte mínimo
    encontrado pela BFS final na rede residual.
    """

    def __init__(self, momento: str = "T1", n_rotulados: int = 8,
                 n_sem_rotulo: int = 6, k: int = 3, lam: float = 1.0,
                 semente: int = SEMENTE) -> None:
        if n_rotulados < 2 or n_sem_rotulo < 1:
            raise ValueError("a demonstração precisa de rotulados das "
                             "duas classes e de um aluno sem rótulo")
        self.momento = momento
        self.n_rotulados = n_rotulados
        self.n_sem_rotulo = n_sem_rotulo
        self.k = k
        self.lam = lam
        self.semente = semente

    def executar(self, base,
                 pesos: Mapping[str, float] | None = None) -> dict:
        """Roda o corte mínimo e devolve a rede e os passos.

        Args:
            base: ``BaseDeDados``.
            pesos: pesos de Gower; use os do G1 oficial (informação
                mútua no treino inteiro). ``None`` recalcula a IM só
                com os poucos alunos sorteados, o que é instável.

        Chaves: ``rede``, ``grafo`` (similaridade), ``rotulos``
        (conhecidos), ``sem_rotulo``, ``real`` (desfecho real de
        todos), ``passos`` (lista de dicts com ``caminho``,
        ``gargalo``, ``fluxo`` acumulado e ``residual``), ``fluxo``,
        ``lado_s``, ``previsoes`` e ``capacidade_corte``.
        """
        rotulados, sem_rotulo = self._sortear(base)
        modelo = GrafoSimilaridade(self.momento, self.k, pesos=pesos,
                                   lam=self.lam)
        grafo = modelo.construir(base, rotulados, sem_rotulo)
        classificador = ClassificadorCorteMinimo(
            lam=self.lam, contrair=False, registrar=True)
        previsoes = classificador.classificar(
            grafo, modelo.rotulos_, sem_rotulo)
        rede = classificador.rede

        residual = self.residual_inicial(rede)
        passos = [{"caminho": [], "gargalo": 0.0, "fluxo": 0.0,
                   "residual": self._copiar(residual)}]
        fluxo = 0.0
        for registro in classificador.historico:
            g = registro["gargalo"]
            for u, v in registro["caminho"]:
                residual[u][v] -= g
                residual[v][u] += g
            fluxo += g
            passos.append({"caminho": registro["caminho"], "gargalo": g,
                           "fluxo": fluxo,
                           "residual": self._copiar(residual)})

        lado_s = self.alcancaveis(residual, FONTE)
        capacidade_corte = sum(
            c for u, v, c in rede.arestas()
            if u in lado_s and v not in lado_s)
        idx = [*rotulados, *sem_rotulo]
        return {
            "rede": rede,
            "grafo": grafo,
            "rotulos": dict(modelo.rotulos_),
            "sem_rotulo": list(sem_rotulo),
            "real": dict(zip(idx, map(bool, base.y(idx)))),
            "passos": passos,
            "fluxo": classificador.fluxo,
            "lado_s": lado_s,
            "previsoes": previsoes,
            "capacidade_corte": capacidade_corte,
        }

    def _sortear(self, base) -> tuple[list[int], list[int]]:
        """Alunos do treino: rotulados equilibrados e os sem rótulo."""
        rng = np.random.default_rng(self.semente)
        treino = np.array(base.idx_treino)
        y = base.y(treino.tolist())
        evadiu = rng.permutation(treino[y])
        formou = rng.permutation(treino[~y])
        meio = self.n_rotulados // 2
        rotulados = [*evadiu[:meio], *formou[:self.n_rotulados - meio]]
        meio_t = self.n_sem_rotulo // 2
        sem_rotulo = [*evadiu[meio:meio + meio_t],
                      *formou[self.n_rotulados - meio:
                              self.n_rotulados - meio
                              + self.n_sem_rotulo - meio_t]]
        return (sorted(int(i) for i in rotulados),
                sorted(int(i) for i in sem_rotulo))

    @staticmethod
    def residual_inicial(rede: Grafo) -> Residual:
        """Folgas iniciais: capacidade em cada arco e 0 no de volta."""
        residual: Residual = {v: {} for v in rede.vertices()}
        for u, v, c in rede.arestas():
            residual[u][v] = residual[u].get(v, 0.0) + c
            residual[v].setdefault(u, 0.0)
        return residual

    @staticmethod
    def alcancaveis(residual: Residual, s: Hashable) -> set[Hashable]:
        """Vértices com folga positiva a partir de ``s`` (lado S)."""
        vistos = {s}
        pilha = [s]
        while pilha:
            u = pilha.pop()
            for v, folga in residual[u].items():
                if folga > 1e-12 and v not in vistos:
                    vistos.add(v)
                    pilha.append(v)
        return vistos

    @staticmethod
    def _copiar(residual: Residual) -> Residual:
        return {u: dict(viz) for u, viz in residual.items()}


class DemoKruskal:
    """Kruskal (aula 5) passo a passo, a partir da árvore oficial.

    A ordem de exame é a do ``ArvoreDependencias.kruskal`` com
    ``maximo=True``: IM decrescente, empates pelo nome dos vértices.
    Cada aresta é marcada como aceita se está na árvore devolvida pelo
    Kruskal; para as rejeitadas, confere-se que as pontas já estavam no
    mesmo componente (fecharia um ciclo).

    Garantia mostrada: a árvore final é geradora (liga todos os
    vértices, sem ciclos) e tem a maior soma de IM possível.
    """

    def executar(self, modelo: ArvoreDependencias,
                 arvore: list[tuple[str, str, float]]) -> dict:
        """Passos do Kruskal sobre o grafo completo de ``modelo``.

        Args:
            modelo: ``ArvoreDependencias`` já construído.
            arvore: ``executar()["arestas"]`` do mesmo modelo.

        Devolve ``vertices``, ``passos`` (``u``, ``v``, ``peso``,
        ``aceita``, ``componentes`` depois do passo e ``num_aceitas``)
        e ``num_examinadas``.
        """
        grafo = modelo.grafo
        if grafo is None:
            raise RuntimeError("construa o modelo antes")
        na_arvore = {tuple(sorted((u, v), key=str)) for u, v, _ in arvore}
        arestas = sorted(
            (tuple(sorted((u, v), key=str)) + (p,)
             for u, v, p in grafo.arestas() if u != v),
            key=lambda a: (-a[2], str(a[0]), str(a[1])))

        vertices = sorted(grafo.vertices(), key=str)
        conjuntos = UniaoBusca(vertices)
        passos = []
        aceitas = 0
        for u, v, p in arestas:
            if aceitas == len(arvore):
                break
            aceita = (u, v) in na_arvore
            if aceita:
                if not conjuntos.unir(u, v):
                    raise AssertionError(f"aresta da árvore fecha ciclo: "
                                         f"{u} — {v}")
                aceitas += 1
            elif conjuntos.buscar(u) != conjuntos.buscar(v):
                raise AssertionError(f"aresta rejeitada não fecharia "
                                     f"ciclo: {u} — {v}")
            passos.append({
                "u": u, "v": v, "peso": p, "aceita": aceita,
                "num_aceitas": aceitas,
                "componentes": {x: conjuntos.buscar(x) for x in vertices},
            })
        return {"vertices": vertices, "passos": passos,
                "num_examinadas": len(passos),
                "num_arestas_completo": len(arestas)}
