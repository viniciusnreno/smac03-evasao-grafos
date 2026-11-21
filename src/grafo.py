"""Estrutura de dados comum a todos os grafos do projeto.

Lista de adjacência com peso: ``_adj[u][v]`` é o peso (ou a capacidade)
da aresta u -> v. Vértices podem ser de qualquer tipo hashable: ``int``
para alunos (G1) e ``str`` para atributos (G2) e estados (G3).

Responsável: Rafael.
"""

from collections.abc import Hashable, Iterator, Mapping
from types import MappingProxyType


class Grafo:
    """Grafo com peso, dirigido ou não dirigido.

    Em um grafo não dirigido, ``adicionar_aresta`` grava os dois
    sentidos, o que garante a simetria exigida pelo kNN do G1.
    Adicionar uma aresta que já existe substitui o peso anterior.
    """

    def __init__(self, dirigido: bool = False) -> None:
        self.dirigido = dirigido
        self._adj: dict[Hashable, dict[Hashable, float]] = {}

    def __repr__(self) -> str:
        tipo = "dirigido" if self.dirigido else "não dirigido"
        return (f"Grafo({tipo}, {len(self)} vértices, "
                f"{self.num_arestas()} arestas)")

    def __len__(self) -> int:
        """Número de vértices."""
        return len(self._adj)

    def __contains__(self, v: Hashable) -> bool:
        """Indica se ``v`` é vértice do grafo."""
        return v in self._adj

    def adicionar_vertice(self, v: Hashable) -> None:
        """Inclui o vértice ``v`` sem arestas, se ainda não existir."""
        self._adj.setdefault(v, {})

    def adicionar_aresta(self, u: Hashable, v: Hashable,
                         peso: float) -> None:
        """Inclui a aresta u -> v (e v -> u, se não dirigido).

        Os vértices são criados se ainda não existirem.
        """
        self.adicionar_vertice(u)
        self.adicionar_vertice(v)
        self._adj[u][v] = peso
        if not self.dirigido:
            self._adj[v][u] = peso

    def tem_aresta(self, u: Hashable, v: Hashable) -> bool:
        """Indica se existe a aresta u -> v."""
        return u in self._adj and v in self._adj[u]

    def vizinhos(self, u: Hashable) -> Mapping[Hashable, float]:
        """Devolve ``{v: peso}`` para as arestas que saem de ``u``.

        A visão é somente leitura e não copia os dados, para que buscas
        repetidas (BFS do Edmonds-Karp) não paguem cópias.
        """
        if u not in self._adj:
            raise KeyError(f"vértice inexistente: {u!r}")
        return MappingProxyType(self._adj[u])

    def peso(self, u: Hashable, v: Hashable) -> float:
        """Devolve o peso da aresta u -> v."""
        if not self.tem_aresta(u, v):
            raise KeyError(f"aresta inexistente: {u!r} -> {v!r}")
        return self._adj[u][v]

    def vertices(self) -> list[Hashable]:
        """Devolve a lista de vértices, na ordem de inclusão."""
        return list(self._adj)

    def arestas(self) -> list[tuple[Hashable, Hashable, float]]:
        """Devolve ``(u, v, peso)``; uma vez só por aresta não dirigida."""
        return list(self._iterar_arestas())

    def num_arestas(self) -> int:
        """Número de arestas (contando uma vez as não dirigidas)."""
        return sum(1 for _ in self._iterar_arestas())

    def para_networkx(self, atributo: str = "weight"):
        """Converte para networkx (só em asserts de verificação).

        Use ``atributo="capacity"`` para comparar com os algoritmos de
        fluxo do networkx.
        """
        import networkx as nx

        g = nx.DiGraph() if self.dirigido else nx.Graph()
        g.add_nodes_from(self._adj)
        g.add_weighted_edges_from(self._iterar_arestas(), weight=atributo)
        return g

    def _iterar_arestas(
            self) -> Iterator[tuple[Hashable, Hashable, float]]:
        """Percorre as arestas, sem repetir as não dirigidas."""
        visitados: set[Hashable] = set()
        for u, vizinhos in self._adj.items():
            for v, peso in vizinhos.items():
                if self.dirigido or v not in visitados:
                    yield u, v, peso
            visitados.add(u)
