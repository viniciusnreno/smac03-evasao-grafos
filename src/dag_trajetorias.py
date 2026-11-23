"""G3: DAG de trajetórias. Classe: ``DagTrajetorias(ModeloGrafo)``.

Cada aluno de treino percorre um caminho por cinco camadas::

    INICIO -> L0 bolsa x faixa etária -> L1 mensalidade x devedor
           -> L2 aprovação no S1 -> L3 aprovação no S2 -> L4 desfecho

Os arcos acumulam a contagem de alunos, P(próximo | atual) e a taxa de
evasão de quem passa por eles. No treino oficial são 23 vértices e 76
arcos.

Três classes:

- ``Kahn``: ordenação topológica (aula 3). Garante uma ordem em que todo
  arco u -> v tem u antes de v, ou detecta ciclo, em O(V + E).
- ``BellmanFord``: caminho mínimo de fonte única (aula 4). Garante as
  distâncias mínimas mesmo com pesos negativos e detecta ciclo negativo,
  em O(V·E). Com os arcos em ordem topológica converge em uma rodada.
- ``DagTrajetorias``: monta o DAG, valida com Kahn e usa Bellman-Ford
  com peso -ln(taxa) para achar a trajetória de maior risco (até a
  evasão) e a mais segura (até a formatura). Como preditor, o risco do
  aluno é a taxa de evasão, no treino, do trecho do caminho que já é
  conhecido no momento.
"""

import math
from collections import Counter, deque
from collections.abc import Hashable, Iterable

import numpy as np
import pandas as pd

from dados import faixa_aprovacao
from grafo import Grafo
from modelo import ModeloGrafo

INICIO = "INICIO"
EVASAO = "L4:Dropout"
FORMATURA = "L4:Graduate"

# Faixas etárias do L0: até 20, de 21 a 23, acima de 23 anos.
LIMITES_IDADE = (20, 23)

# Camadas do caminho conhecidas em cada momento (L0, L1, ...).
CAMADAS_POR_MOMENTO: dict[str, int] = {"T0": 2, "T1": 3, "T2": 4}

# Arco ou trecho só é usado com pelo menos este número de alunos.
MIN_SUPORTE = 20

# Suavização de Laplace da taxa do arco: (evasões + 1) / (n + 2). Evita
# -ln 0 e, entre arcos com a mesma taxa, prefere o de mais alunos.
SUAVIZACAO = 1.0


class Kahn:
    """Ordenação topológica de Kahn (aula 3), em O(V + E).

    Garante uma ordem dos vértices em que todo arco u -> v tem u antes
    de v. Se sobrar vértice sem entrar na ordem, o grafo tem ciclo.
    """

    def ordenar(self, grafo: Grafo) -> list[Hashable]:
        """Devolve a ordem topológica; ``ValueError`` se houver ciclo."""
        if not grafo.dirigido:
            raise ValueError("ordenação topológica exige grafo dirigido")
        grau_entrada = {v: 0 for v in grafo.vertices()}
        for _, v, _ in grafo.arestas():
            grau_entrada[v] += 1

        fila = deque(v for v, g in grau_entrada.items() if g == 0)
        ordem = []
        while fila:
            u = fila.popleft()
            ordem.append(u)
            for v in grafo.vizinhos(u):
                grau_entrada[v] -= 1
                if grau_entrada[v] == 0:
                    fila.append(v)

        if len(ordem) != len(grafo):
            raise ValueError("o grafo tem ciclo: não é um DAG")
        return ordem


class BellmanFord:
    """Caminho mínimo de fonte única por Bellman-Ford (aula 4).

    Garante as menores distâncias a partir da fonte com pesos de
    qualquer sinal, desde que não haja ciclo negativo alcançável (que é
    detectado). Faz até V - 1 rodadas de relaxamento, em O(V·E), e para
    antes se uma rodada não muda nada.

    Uso::

        bf = BellmanFord()
        dist = bf.distancias(grafo, fonte)
        caminho = bf.caminho(destino)
    """

    def __init__(self) -> None:
        self.dist: dict[Hashable, float] = {}
        self.pais: dict[Hashable, Hashable | None] = {}
        self.rodadas = 0
        self._fonte: Hashable = None

    def distancias(self, grafo: Grafo,
                   fonte: Hashable) -> dict[Hashable, float]:
        """Distância mínima da ``fonte`` a cada vértice (inf se não há)."""
        if fonte not in grafo:
            raise KeyError(f"fonte fora do grafo: {fonte!r}")
        arcos = grafo.arestas()
        if not grafo.dirigido:
            arcos += [(v, u, w) for u, v, w in arcos]

        self.dist = {v: math.inf for v in grafo.vertices()}
        self.pais = {v: None for v in grafo.vertices()}
        self.dist[fonte] = 0.0
        self._fonte = fonte
        self.rodadas = 0
        for _ in range(len(grafo) - 1):
            self.rodadas += 1
            if not self._relaxar(arcos):
                break
        if self._relaxar(arcos, apenas_testar=True):
            raise ValueError("ciclo negativo alcançável a partir da fonte")
        return dict(self.dist)

    def caminho(self, destino: Hashable) -> list[Hashable]:
        """Caminho mínimo da fonte até ``destino`` (vazio se não há)."""
        if self._fonte is None:
            raise RuntimeError("chame distancias antes de caminho")
        if self.dist.get(destino, math.inf) == math.inf:
            return []
        caminho = [destino]
        while caminho[-1] != self._fonte:
            caminho.append(self.pais[caminho[-1]])
        caminho.reverse()
        return caminho

    def _relaxar(self, arcos: list[tuple[Hashable, Hashable, float]],
                 apenas_testar: bool = False) -> bool:
        """Uma rodada de relaxamento; indica se alguma distância caiu."""
        mudou = False
        for u, v, w in arcos:
            if self.dist[u] + w < self.dist[v]:
                if apenas_testar:
                    return True
                self.dist[v] = self.dist[u] + w
                self.pais[v] = u
                mudou = True
        return mudou


class DagTrajetorias(ModeloGrafo):
    """G3: DAG de estados da trajetória do aluno.

    Parâmetros:
        momento: ``"T0"``, ``"T1"`` ou ``"T2"``. Define quantas camadas
            do caminho o preditor usa (2, 3 ou 4).
        min_suporte: mínimo de alunos de treino em um arco (trajetórias)
            ou em um trecho de caminho (preditor).
        limiar: risco acima do qual o aluno é previsto como evasão.
            ``None`` usa a taxa de evasão do treino.

    Depois de ``construir`` ficam disponíveis ``grafo`` (peso = número
    de alunos no arco), ``contagem_vertice_``, ``evasoes_vertice_``,
    ``contagem_arco_``, ``evasoes_arco_`` e ``taxa_treino_``.
    """

    nome = "G3 · DAG de trajetórias"
    descricao = ("Camadas ingresso → financeiro → S1 → S2 → desfecho. "
                 "Kahn valida o DAG e Bellman-Ford acha a trajetória "
                 "de maior risco.")

    def __init__(self, momento: str = "T2", min_suporte: int = MIN_SUPORTE,
                 limiar: float | None = None) -> None:
        super().__init__()
        if momento not in CAMADAS_POR_MOMENTO:
            raise ValueError(f"momento inválido: {momento!r}")
        if min_suporte < 1:
            raise ValueError("min_suporte deve ser pelo menos 1")
        self.momento = momento
        self.min_suporte = min_suporte
        self.limiar = limiar

        self.contagem_vertice_: Counter = Counter()
        self.evasoes_vertice_: Counter = Counter()
        self.contagem_arco_: Counter = Counter()
        self.evasoes_arco_: Counter = Counter()
        self.taxa_treino_: float = 0.0
        self.idx_sem_rotulo_: list[int] = []
        self._trechos: Counter = Counter()
        self._evasoes_trecho: Counter = Counter()
        self._estados_sem_rotulo: dict[int, tuple[str, ...]] = {}
        self._resultado: dict | None = None

    # ------------------------------------------------------------------
    # Interface ModeloGrafo
    # ------------------------------------------------------------------
    def construir(self, base, idx_treino: Iterable[int] | None = None,
                  idx_teste: Iterable[int] | None = None) -> Grafo:
        """Monta o DAG com os caminhos dos alunos de treino.

        Os alunos de ``idx_teste`` só têm os estados L0 a L3 guardados
        para o preditor; o desfecho deles nunca é lido.
        """
        idx_treino = sorted(int(i) for i in (
            base.idx_treino if idx_treino is None else idx_treino))
        idx_teste = sorted(int(i) for i in (
            base.idx_teste if idx_teste is None else idx_teste))
        if not idx_treino:
            raise ValueError("o treino não pode ser vazio")
        if set(idx_treino) & set(idx_teste):
            raise ValueError("aluno do teste também está no treino")

        estados = self.estados(base.dados.loc[idx_treino])
        evadiu = base.y(idx_treino)
        self._contar(estados, evadiu)
        self.taxa_treino_ = float(np.mean(evadiu))

        self.idx_sem_rotulo_ = idx_teste
        estados_teste = self.estados(base.dados.loc[idx_teste])
        self._estados_sem_rotulo = {
            i: tuple(linha) for i, linha in
            zip(idx_teste, estados_teste.itertuples(index=False))}

        self.grafo = Grafo(dirigido=True)
        self.grafo.adicionar_vertice(INICIO)
        for (u, v), n in self.contagem_arco_.items():
            self.grafo.adicionar_aresta(u, v, float(n))
        self._resultado = None
        return self.grafo

    def executar(self) -> dict:
        """Ordena com Kahn, busca trajetórias e classifica o teste.

        Devolve ``ordem_topologica``, ``trajetoria_risco`` e
        ``trajetoria_segura`` (caminho, n e evasões no treino),
        ``ponto_de_virada``, ``risco`` e ``previsoes`` por aluno,
        ``limiar`` e o tamanho do DAG.
        """
        if self.grafo is None:
            raise RuntimeError("chame construir antes de executar")
        ordem = Kahn().ordenar(self.grafo)
        limiar = self.taxa_treino_ if self.limiar is None else self.limiar
        risco = {i: self._risco(e)
                 for i, e in self._estados_sem_rotulo.items()}
        self._resultado = {
            "momento": self.momento,
            "ordem_topologica": ordem,
            "trajetoria_risco": self._trajetoria(ordem, EVASAO),
            "trajetoria_segura": self._trajetoria(ordem, FORMATURA),
            "ponto_de_virada": self._ponto_de_virada(),
            "risco": risco,
            "previsoes": {i: r > limiar for i, r in risco.items()},
            "limiar": limiar,
            "num_vertices": len(self.grafo),
            "num_arestas": self.grafo.num_arestas(),
        }
        return self._resultado

    def destaques(self) -> dict[str, list[Hashable]]:
        """Vértices e arcos da trajetória de maior risco."""
        if self._resultado is None:
            raise RuntimeError("chame executar antes de destaques")
        caminho = self._resultado["trajetoria_risco"]["caminho"]
        return {"vertices": list(caminho),
                "arestas": list(zip(caminho, caminho[1:]))}

    # ------------------------------------------------------------------
    # Estados e estatísticas
    # ------------------------------------------------------------------
    @staticmethod
    def estados(dados: pd.DataFrame) -> pd.DataFrame:
        """Estados L0 a L3 de cada aluno (rótulos dos vértices)."""
        idade = dados["Age at enrollment"]
        baixo, alto = LIMITES_IDADE
        faixa_idade = np.select(
            [idade <= baixo, idade <= alto],
            [f"Idade<={baixo}", f"{baixo}<Idade<={alto}"],
            f"Idade>{alto}")
        bolsa = np.where(dados["Scholarship holder"] == 1,
                         "Bolsista", "Sem bolsa")
        mensalidade = np.where(dados["Tuition fees up to date"] == 1,
                               "em dia", "atrasada")
        devedor = np.where(dados["Debtor"] == 1, "devedor", "sem divida")
        camadas = {
            "L0": np.char.add(np.char.add(bolsa, " & "), faixa_idade),
            "L1": np.char.add(np.char.add(mensalidade, ", "), devedor),
            "L2": faixa_aprovacao(dados["taxa_aprov_s1"]),
            "L3": faixa_aprovacao(dados["taxa_aprov_s2"]),
        }
        return pd.DataFrame(
            {c: np.char.add(f"{c}:", v.astype(str))
             for c, v in camadas.items()}, index=dados.index)

    def _contar(self, estados: pd.DataFrame, evadiu: np.ndarray) -> None:
        """Conta alunos e evasões por vértice, arco e trecho de caminho."""
        for contador in (self.contagem_vertice_, self.evasoes_vertice_,
                         self.contagem_arco_, self.evasoes_arco_,
                         self._trechos, self._evasoes_trecho):
            contador.clear()
        for linha, ev in zip(estados.itertuples(index=False), evadiu):
            ev = int(ev)
            caminho = (INICIO, *linha, EVASAO if ev else FORMATURA)
            for v in caminho:
                self.contagem_vertice_[v] += 1
                self.evasoes_vertice_[v] += ev
            for arco in zip(caminho, caminho[1:]):
                self.contagem_arco_[arco] += 1
                self.evasoes_arco_[arco] += ev
            for tamanho in range(1, len(linha) + 1):
                trecho = tuple(linha[:tamanho])
                self._trechos[trecho] += 1
                self._evasoes_trecho[trecho] += ev

    def probabilidade(self, u: str, v: str) -> float:
        """P(próximo = v | atual = u), estimada no treino."""
        return self.contagem_arco_[(u, v)] / self.contagem_vertice_[u]

    def taxa_arco(self, u: str, v: str) -> float:
        """Taxa de evasão de quem passou pelo arco u -> v no treino."""
        return self.evasoes_arco_[(u, v)] / self.contagem_arco_[(u, v)]

    def _risco(self, estados: tuple[str, ...]) -> float:
        """Taxa de evasão do maior trecho conhecido com suporte."""
        conhecidas = CAMADAS_POR_MOMENTO[self.momento]
        for tamanho in range(conhecidas, 0, -1):
            trecho = estados[:tamanho]
            n = self._trechos[trecho]
            if n >= self.min_suporte:
                return self._evasoes_trecho[trecho] / n
        return self.taxa_treino_

    def _trajetoria(self, ordem: list[Hashable], destino: str) -> dict:
        """Caminho de INICIO a ``destino`` que maximiza o produto das
        taxas (de evasão ou de formatura) dos arcos com suporte.

        Bellman-Ford com peso -ln(taxa suavizada): somar -ln equivale a
        multiplicar as taxas. Os arcos entram em ordem topológica.
        """
        evasao = destino == EVASAO
        pesos = Grafo(dirigido=True)
        for v in ordem:
            pesos.adicionar_vertice(v)
        for u in ordem:
            for v in self.grafo.vizinhos(u):
                n = self.contagem_arco_[(u, v)]
                if n < self.min_suporte:
                    continue
                ev = self.evasoes_arco_[(u, v)]
                favoraveis = ev if evasao else n - ev
                taxa = (favoraveis + SUAVIZACAO) / (n + 2 * SUAVIZACAO)
                pesos.adicionar_aresta(u, v, -math.log(taxa))

        bf = BellmanFord()
        bf.distancias(pesos, INICIO)
        caminho = bf.caminho(destino)
        trecho = tuple(caminho[1:-1])
        return {
            "caminho": caminho,
            "n": self._trechos[trecho] if trecho else 0,
            "evasoes": self._evasoes_trecho[trecho] if trecho else 0,
        }

    def _ponto_de_virada(self) -> dict:
        """Arco com o maior salto de risco: taxa do arco - taxa de u."""
        melhor: dict = {}
        for (u, v), n in self.contagem_arco_.items():
            if u == INICIO or v in (EVASAO, FORMATURA):
                continue
            if n < self.min_suporte:
                continue
            antes = self.evasoes_vertice_[u] / self.contagem_vertice_[u]
            depois = self.taxa_arco(u, v)
            if not melhor or depois - antes > melhor["salto"]:
                melhor = {"arco": (u, v), "n": n, "taxa_antes": antes,
                          "taxa_depois": depois, "salto": depois - antes}
        return melhor
