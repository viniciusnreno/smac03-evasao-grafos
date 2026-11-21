"""G2: árvore de dependências. Classe: ``ArvoreDependencias(ModeloGrafo)``.

Responsável: Carlos.

- ``construir``: grafo completo não dirigido cujos vértices são os
  atributos do momento e o vértice ``Desfecho``; o peso de cada aresta é
  a informação mútua (IM) entre os dois, calculada só no treino.
- ``executar``: Kruskal em ordem decrescente de peso com union-find
  (aula 5). Garante uma árvore geradora de peso **máximo**: liga todos os
  vértices sem ciclos com a maior IM total possível. É a árvore de Chow e
  Liu (1968), a melhor aproximação em árvore da distribuição conjunta.
  Não é caminho mínimo: a árvore não diz nada sobre distâncias.
- ``destaques``: arestas da árvore e vizinhos diretos do ``Desfecho``
  (os fatores com informação própria sobre a evasão).

A IM usa as mesmas faixas do G1 (``GrafoSimilaridade.discretizar``):
taxas de aprovação nas faixas do DAG, nominais raras em "outros",
numéricas com muitos valores em quartis.
"""

from collections.abc import Hashable, Iterable

import numpy as np

from grafo import Grafo
from grafo_similaridade import GrafoSimilaridade
from modelo import ModeloGrafo

DESFECHO = "Desfecho"
SEMENTE = 42
# Reamostragens do treino para medir a estabilidade das arestas.
N_BOOTSTRAP = 100

Aresta = tuple[str, str]


class UniaoBusca:
    """Union-find (conjuntos disjuntos), aula 5.

    União por posto e compressão de caminho: cada operação custa tempo
    quase constante (função inversa de Ackermann).
    """

    def __init__(self, elementos: Iterable[Hashable]) -> None:
        self._pai: dict[Hashable, Hashable] = {e: e for e in elementos}
        self._posto: dict[Hashable, int] = dict.fromkeys(self._pai, 0)

    def buscar(self, x: Hashable) -> Hashable:
        """Representante do conjunto de ``x`` (Find-Set)."""
        raiz = x
        while self._pai[raiz] != raiz:
            raiz = self._pai[raiz]
        while self._pai[x] != raiz:          # compressão de caminho
            self._pai[x], x = raiz, self._pai[x]
        return raiz

    def unir(self, a: Hashable, b: Hashable) -> bool:
        """Une os conjuntos de ``a`` e ``b`` (Union).

        Devolve ``False`` se já estavam no mesmo conjunto, ou seja, se a
        aresta (a, b) fecharia um ciclo.
        """
        ra, rb = self.buscar(a), self.buscar(b)
        if ra == rb:
            return False
        if self._posto[ra] < self._posto[rb]:
            ra, rb = rb, ra
        self._pai[rb] = ra
        if self._posto[ra] == self._posto[rb]:
            self._posto[ra] += 1
        return True


class ArvoreDependencias(ModeloGrafo):
    """G2: árvore geradora máxima de informação mútua (Kruskal).

    Parâmetros:
        momento: ``"T0"``, ``"T1"`` ou ``"T2"``.
        incluir_derivados: inclui ``taxa_aprov_s1``/``taxa_aprov_s2``
            nos momentos em que existem (38 atributos em T2).
        excluir: atributos a retirar.

    Depois de ``construir`` ficam disponíveis ``grafo`` (completo) e
    ``im_desfecho_`` (IM de cada atributo com o desfecho). Depois de
    ``executar``, ``arvore_`` (``Grafo`` só com as arestas da árvore).
    """

    nome = "G2 · Árvore de dependências (Kruskal)"
    descricao = ("Cada atributo é um vértice, mais o Desfecho. O peso é a "
                 "informação mútua. Kruskal liga todos sem ciclos com a "
                 "maior informação total: os vizinhos do Desfecho são os "
                 "fatores com informação própria.")

    def __init__(self, momento: str, incluir_derivados: bool = True,
                 excluir: Iterable[str] = ()) -> None:
        super().__init__()
        self.momento = momento
        self.incluir_derivados = incluir_derivados
        self.excluir = frozenset(excluir)
        self.im_desfecho_: dict[str, float] = {}
        self.arvore_: Grafo | None = None
        self._arestas: list[tuple[str, str, float]] = []
        self._idx: list[int] = []

    # ------------------------------------------------------------------
    # Interface ModeloGrafo
    # ------------------------------------------------------------------
    def construir(self, base, idx: Iterable[int] | None = None) -> Grafo:
        """Grafo completo de IM entre atributos e desfecho.

        Args:
            base: ``BaseDeDados``.
            idx: alunos usados no cálculo da IM; padrão
                ``base.idx_treino``. Passar o teste só para medir a
                replicação da árvore, nunca para escolher algo.
        """
        self._idx = list(base.idx_treino if idx is None else idx)
        if not self._idx:
            raise ValueError("o conjunto de alunos não pode ser vazio")
        self.grafo = self._grafo_im(base, self._idx)
        self.im_desfecho_ = dict(sorted(
            ((v, self.grafo.peso(DESFECHO, v))
             for v in self.grafo.vizinhos(DESFECHO)),
            key=lambda par: -par[1]))
        self.arvore_ = None
        self._arestas = []
        return self.grafo

    def executar(self) -> dict:
        """Roda o Kruskal e devolve a árvore e seus destaques.

        Chaves: ``arestas`` (``(u, v, im)`` em ordem decrescente de IM),
        ``peso_total``, ``vizinhos_desfecho`` (``[(atributo, im)]``),
        ``im_desfecho``, ``num_vertices`` e ``num_arestas_completo``.
        """
        if self.grafo is None:
            raise RuntimeError("chame construir antes de executar")
        self._arestas = self.kruskal(self.grafo)
        self.arvore_ = Grafo(dirigido=False)
        for v in self.grafo.vertices():
            self.arvore_.adicionar_vertice(v)
        for u, v, peso in self._arestas:
            self.arvore_.adicionar_aresta(u, v, peso)
        vizinhos = sorted(self.arvore_.vizinhos(DESFECHO).items(),
                          key=lambda par: -par[1])
        return {
            "momento": self.momento,
            "arestas": list(self._arestas),
            "peso_total": sum(p for _, _, p in self._arestas),
            "vizinhos_desfecho": vizinhos,
            "im_desfecho": dict(self.im_desfecho_),
            "num_vertices": len(self.grafo),
            "num_arestas_completo": self.grafo.num_arestas(),
        }

    def destaques(self) -> dict[str, list[Hashable]]:
        """Desfecho, seus vizinhos na árvore e as arestas da árvore."""
        if self.arvore_ is None:
            raise RuntimeError("chame executar antes de destaques")
        return {
            "vertices": [DESFECHO, *self.arvore_.vizinhos(DESFECHO)],
            "arestas": [(u, v) for u, v, _ in self._arestas],
        }

    # ------------------------------------------------------------------
    # Algoritmo
    # ------------------------------------------------------------------
    @staticmethod
    def kruskal(grafo: Grafo,
                maximo: bool = True) -> list[tuple[str, str, float]]:
        """Kruskal, aula 5: árvore geradora em O(E log E).

        Percorre as arestas em ordem de peso (decrescente se ``maximo``)
        e aceita cada uma que não fecha ciclo, o que o union-find decide.
        Garante uma árvore geradora de peso máximo (ou mínimo); se o
        grafo for desconexo, devolve uma floresta geradora. Empates são
        desfeitos pelo nome dos vértices, para o resultado ser
        reprodutível.
        """
        if grafo.dirigido:
            raise ValueError("Kruskal exige grafo não dirigido")
        sinal = -1.0 if maximo else 1.0
        arestas = sorted(
            (ArvoreDependencias._ordenar(u, v) + (p,)
             for u, v, p in grafo.arestas() if u != v),
            key=lambda a: (sinal * a[2], str(a[0]), str(a[1])))
        conjuntos = UniaoBusca(grafo.vertices())
        arvore: list[tuple[str, str, float]] = []
        limite = len(grafo) - 1
        for u, v, peso in arestas:
            if conjuntos.unir(u, v):
                arvore.append((u, v, peso))
                if len(arvore) == limite:
                    break
        return arvore

    # ------------------------------------------------------------------
    # Análises de robustez (H2)
    # ------------------------------------------------------------------
    def estabilidade(self, base, n: int = N_BOOTSTRAP,
                     semente: int = SEMENTE) -> dict[Aresta, float]:
        """Fração de reamostragens em que cada aresta da árvore volta.

        Reamostra com reposição os mesmos alunos de ``construir`` (o
        treino), refaz a IM e o Kruskal e conta as aparições. Exige
        ``executar`` antes.
        """
        if self.arvore_ is None:
            raise RuntimeError("chame executar antes de estabilidade")
        rng = np.random.default_rng(semente)
        contagem = dict.fromkeys(self.chaves(self._arestas), 0)
        idx = np.asarray(self._idx)
        for _ in range(n):
            amostra = rng.choice(idx, size=len(idx), replace=True)
            grafo = self._grafo_im(base, amostra.tolist())
            for chave in self.chaves(self.kruskal(grafo)):
                if chave in contagem:
                    contagem[chave] += 1
        return {chave: c / n for chave, c in contagem.items()}

    @staticmethod
    def sobreposicao(arestas_a: Iterable[tuple],
                     arestas_b: Iterable[tuple]) -> float:
        """Fração das arestas de ``a`` que também estão em ``b``."""
        a = ArvoreDependencias.chaves(arestas_a)
        b = ArvoreDependencias.chaves(arestas_b)
        if not a:
            raise ValueError("a primeira árvore não tem arestas")
        return len(a & b) / len(a)

    @staticmethod
    def chaves(arestas: Iterable[tuple]) -> set[Aresta]:
        """Arestas como pares ordenados, para comparar árvores."""
        return {ArvoreDependencias._ordenar(a[0], a[1]) for a in arestas}

    # ------------------------------------------------------------------
    # Auxiliares
    # ------------------------------------------------------------------
    def _colunas(self, base) -> list[str]:
        """Atributos do momento, sem os excluídos."""
        disponiveis = base.atributos(self.momento, self.incluir_derivados)
        desconhecidos = self.excluir - set(disponiveis)
        if desconhecidos:
            raise ValueError(f"atributos fora do momento {self.momento}: "
                             f"{sorted(desconhecidos)}")
        return [c for c in disponiveis if c not in self.excluir]

    def _grafo_im(self, base, idx: list[int]) -> Grafo:
        """Grafo completo com peso = IM (bits) entre cada par."""
        colunas = self._colunas(base)
        dados = base.dados.loc[idx, colunas]
        codigos = {c: GrafoSimilaridade.discretizar(dados[c])
                   for c in colunas}
        codigos[DESFECHO] = base.y(idx).astype(int)
        vertices = list(codigos)
        grafo = Grafo(dirigido=False)
        for v in vertices:
            grafo.adicionar_vertice(v)
        for i, a in enumerate(vertices):
            for b in vertices[i + 1:]:
                grafo.adicionar_aresta(
                    a, b, self.informacao_mutua(codigos[a], codigos[b]))
        return grafo

    @staticmethod
    def informacao_mutua(a: np.ndarray, b: np.ndarray) -> float:
        """IM em bits entre dois vetores de códigos inteiros (>= 0).

        I(A;B) = soma p(a,b) · log2[p(a,b) / (p(a) p(b))], pela tabela
        de contingência. Mesmo valor do ``mutual_info_score`` do
        scikit-learn, porém bem mais rápido para as 741 arestas de T2.
        """
        nb = int(b.max()) + 1
        conjunta = np.bincount(a * nb + b, minlength=(int(a.max()) + 1) * nb)
        conjunta = conjunta.reshape(-1, nb) / len(a)
        pa = conjunta.sum(axis=1, keepdims=True)
        pb = conjunta.sum(axis=0, keepdims=True)
        nz = conjunta > 0
        razao = conjunta[nz] / (pa @ pb)[nz]
        return float(max((conjunta[nz] * np.log2(razao)).sum(), 0.0))

    @staticmethod
    def _ordenar(u: str, v: str) -> Aresta:
        """Par (u, v) em ordem alfabética."""
        return (u, v) if str(u) <= str(v) else (v, u)
