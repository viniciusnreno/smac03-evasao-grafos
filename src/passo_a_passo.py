"""Kruskal na interface: passos, régua de IM e caminhos na árvore.

- ``DemoKruskal`` roda o algoritmo do projeto (não o reimplementa):
  lê a árvore devolvida por ``ArvoreDependencias.kruskal`` e
  reconstitui a ordem em que as arestas foram examinadas. A decisão
  (aceita ou rejeita) vem da saída oficial; o ``UniaoBusca`` só
  acompanha os componentes e confirma que cada rejeitada fecharia um
  ciclo.
- ``ExploradorArvore`` responde às perguntas da tela: o estado do
  Kruskal quando a régua de IM está num ponto (floresta, componentes,
  rejeitadas), o caminho único de um atributo até o Desfecho com o seu
  gargalo e um layout radial com o Desfecho no centro.

Não dependem do Streamlit.
"""

import math
from collections import deque

from arvore_dependencias import ArvoreDependencias, UniaoBusca

Aresta = tuple[str, str, float]


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


class ExploradorArvore:
    """Consultas sobre a árvore geradora máxima do G2.

    Args:
        arvore: arestas ``(u, v, IM)`` devolvidas pelo Kruskal.
        desfecho: vértice do desfecho (raiz do layout e dos caminhos).
    """

    def __init__(self, arvore: list[Aresta], desfecho: str) -> None:
        self.arvore = list(arvore)
        self.desfecho = desfecho
        self._viz: dict[str, dict[str, float]] = {desfecho: {}}
        for u, v, p in self.arvore:
            self._viz.setdefault(u, {})[v] = p
            self._viz.setdefault(v, {})[u] = p
        self._pai, self._profundidade = self._bfs()

    # ------------------------------------------------------------------
    # Régua de IM
    # ------------------------------------------------------------------
    @staticmethod
    def regua(passos: list[dict], k: int) -> dict:
        """Estado do Kruskal logo depois de aceitar a ``k``-ésima aresta.

        ``passos`` vem de ``DemoKruskal.executar``. Como o Kruskal
        examina as arestas da maior para a menor IM, a floresta nesse
        ponto tem as arestas da árvore com IM acima da ``k``-ésima
        aceita (o ``limiar``); nos empates de IM vale a ordem do
        Kruskal (desempate pelo nome).

        Devolve ``limiar``, ``aceitas`` (as ``k`` arestas, na ordem),
        ``componentes`` (vértice → representante no union-find) e
        ``rejeitadas`` (candidatas descartadas até ali por fecharem
        ciclo).
        """
        aceitas: list[Aresta] = []
        rejeitadas = 0
        for passo in passos:
            if passo["aceita"]:
                aceitas.append((passo["u"], passo["v"], passo["peso"]))
                if len(aceitas) == k:
                    return {"limiar": passo["peso"], "aceitas": aceitas,
                            "componentes": passo["componentes"],
                            "rejeitadas": rejeitadas}
            else:
                rejeitadas += 1
        raise ValueError(f"a árvore tem menos de {k} arestas")

    # ------------------------------------------------------------------
    # Caminho até o Desfecho
    # ------------------------------------------------------------------
    def caminho(self, origem: str) -> list[Aresta]:
        """Arestas do caminho único de ``origem`` até o Desfecho.

        Numa árvore o caminho entre dois vértices é único; ele sai dos
        ponteiros de pai da BFS feita a partir do Desfecho.
        """
        if origem not in self._pai:
            raise KeyError(origem)
        arestas = []
        v = origem
        while v != self.desfecho:
            pai = self._pai[v]
            arestas.append((v, pai, self._viz[v][pai]))
            v = pai
        return arestas

    @staticmethod
    def gargalo(caminho: list[Aresta]) -> Aresta | None:
        """Aresta de menor IM do caminho (a ligação mais fraca).

        Garantia da árvore geradora máxima: entre dois vértices, o
        caminho na árvore tem o maior gargalo possível entre todos os
        caminhos do grafo completo (propriedade minimax).
        """
        return min(caminho, key=lambda a: a[2]) if caminho else None

    def profundidade(self, v: str) -> int:
        """Nº de arestas entre ``v`` e o Desfecho na árvore."""
        return self._profundidade[v]

    # ------------------------------------------------------------------
    # Layout radial
    # ------------------------------------------------------------------
    def layout_radial(self, raio: float = 150.0) -> dict[str, tuple]:
        """Posições ``(x, y)`` com o Desfecho no centro.

        Cada anel é uma distância (em arestas) até o Desfecho. Cada
        ramo ocupa uma fatia de ângulo própria (proporcional à raiz do
        tamanho do ramo, para ramos pequenos não ficarem espremidos), e
        os descendentes ficam dentro da fatia: os ramos não se cruzam.
        """
        filhos: dict[str, list[str]] = {v: [] for v in self._pai}
        for v, pai in self._pai.items():
            if pai is not None:
                filhos[pai].append(v)
        for v in filhos:
            filhos[v].sort(key=lambda f: -self._viz[v][f])
        tamanho: dict[str, int] = {}

        def contar(v: str) -> int:
            tamanho[v] = 1 + sum(contar(f) for f in filhos[v])
            return tamanho[v]

        contar(self.desfecho)
        peso = {v: math.sqrt(n) for v, n in tamanho.items()}
        posicoes = {self.desfecho: (0.0, 0.0)}

        def posicionar(v: str, inicio: float, fim: float) -> None:
            angulo = inicio
            soma = sum(peso[f] for f in filhos[v])
            for f in filhos[v]:
                fatia = (fim - inicio) * peso[f] / soma
                meio = angulo + fatia / 2
                r = raio * (self._profundidade[f] + 0.6)
                posicoes[f] = (r * math.cos(meio), r * math.sin(meio))
                posicionar(f, angulo, angulo + fatia)
                angulo += fatia

        posicionar(self.desfecho, -math.pi / 2, 3 * math.pi / 2)
        return posicoes

    def _bfs(self) -> tuple[dict, dict]:
        """Pai e profundidade de cada vértice a partir do Desfecho."""
        pai: dict[str, str | None] = {self.desfecho: None}
        profundidade = {self.desfecho: 0}
        fila = deque([self.desfecho])
        while fila:
            u = fila.popleft()
            for v in self._viz[u]:
                if v not in pai:
                    pai[v] = u
                    profundidade[v] = profundidade[u] + 1
                    fila.append(v)
        return pai, profundidade
