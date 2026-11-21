"""Estrutura de dados comum a todos os grafos do projeto.

Lista de adjacência com peso: ``adj[u][v]`` é o peso (ou a capacidade)
da aresta u -> v. Vértices podem ser de qualquer tipo hashable: ``int``
para alunos (G1) e ``str`` para atributos (G2) e estados (G3).

Responsável: a definir (Carlos ou Rafael). Deve ficar pronta antes dos
demais módulos, que dependem dela.
"""

from collections.abc import Hashable


class Grafo:
    """Grafo com peso, dirigido ou não dirigido.

    Em um grafo não dirigido, ``adicionar_aresta`` grava os dois
    sentidos, o que garante a simetria exigida pelo kNN do G1.
    """

    def __init__(self, dirigido: bool = False) -> None:
        self.dirigido = dirigido
        self._adj: dict[Hashable, dict[Hashable, float]] = {}

    def adicionar_vertice(self, v: Hashable) -> None:
        """Inclui o vértice ``v`` sem arestas, se ainda não existir."""
        raise NotImplementedError

    def adicionar_aresta(self, u: Hashable, v: Hashable,
                         peso: float) -> None:
        """Inclui a aresta u -> v (e v -> u, se não dirigido)."""
        raise NotImplementedError

    def vizinhos(self, u: Hashable) -> dict[Hashable, float]:
        """Devolve ``{v: peso}`` para as arestas que saem de ``u``."""
        raise NotImplementedError

    def peso(self, u: Hashable, v: Hashable) -> float:
        """Devolve o peso da aresta u -> v."""
        raise NotImplementedError

    def vertices(self) -> list[Hashable]:
        """Devolve a lista de vértices."""
        raise NotImplementedError

    def arestas(self) -> list[tuple[Hashable, Hashable, float]]:
        """Devolve ``(u, v, peso)``; uma vez só por aresta não dirigida."""
        raise NotImplementedError

    def para_networkx(self):
        """Converte para networkx (só em asserts de verificação)."""
        raise NotImplementedError
