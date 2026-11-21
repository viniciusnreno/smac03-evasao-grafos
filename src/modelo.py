"""Interface abstrata comum aos modelos de grafo (G1, G2 e G3).

Permite que o ``Pipeline`` e a futura interface gráfica tratem todos os
modelos do mesmo jeito: construir o grafo, executar o algoritmo e
destacar o resultado na tela.

Responsável: Rafael.
"""

from abc import ABC, abstractmethod
from collections.abc import Hashable

from grafo import Grafo


class ModeloGrafo(ABC):
    """Modelo do trabalho: um grafo e o algoritmo que roda sobre ele."""

    nome: str = ""
    descricao: str = ""

    def __init__(self) -> None:
        self.grafo: Grafo | None = None

    @abstractmethod
    def construir(self, base) -> Grafo:
        """Constrói o grafo a partir de uma ``BaseDeDados`` (só treino)."""

    @abstractmethod
    def executar(self) -> dict:
        """Roda o algoritmo do modelo e devolve os resultados em um dict."""

    @abstractmethod
    def destaques(self) -> dict[str, list[Hashable]]:
        """Devolve o que realçar na tela: ``{"vertices": [...],
        "arestas": [(u, v), ...]}``."""
