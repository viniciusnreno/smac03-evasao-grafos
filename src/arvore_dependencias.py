"""G2: árvore de dependências entre os fatores da evasão.

Cada vértice é um atributo do aluno (bolsa, curso, mensalidade...), mais
um vértice ``Desfecho``. Todo par de vértices é ligado por uma aresta
cujo peso é a informação mútua entre os dois: quanto um ajuda a prever
o outro.

O Kruskal escolhe, entre todas essas arestas, uma árvore que liga todos
os vértices sem ciclos e com a maior informação total (árvore de Chow e
Liu). Os vizinhos do ``Desfecho`` na árvore são os fatores que trazem
informação própria sobre a evasão, sem passar por outro fator.
"""

from collections.abc import Hashable, Iterable

import numpy as np

from dados import SEMENTE
from grafo import Grafo
from grafo_similaridade import GrafoSimilaridade
from modelo import ModeloGrafo

DESFECHO = "Desfecho"
# Reamostragens usadas para medir se a árvore é estável.
N_BOOTSTRAP = 100

Aresta = tuple[str, str]


class UniaoBusca:
    """Union-find (conjuntos disjuntos), aula 5.

    Diz em tempo quase constante se dois vértices já estão ligados. O
    Kruskal usa isso para recusar arestas que fechariam ciclo.
    """

    def __init__(self, elementos: Iterable[Hashable]) -> None:
        self._pai: dict[Hashable, Hashable] = {e: e for e in elementos}
        self._posto: dict[Hashable, int] = dict.fromkeys(self._pai, 0)

    def buscar(self, x: Hashable) -> Hashable:
        """Representante do conjunto de ``x``."""
        raiz = x
        while self._pai[raiz] != raiz:
            raiz = self._pai[raiz]
        while self._pai[x] != raiz:          # compressão de caminho
            self._pai[x], x = raiz, self._pai[x]
        return raiz

    def unir(self, a: Hashable, b: Hashable) -> bool:
        """Une os conjuntos de ``a`` e ``b``.

        Devolve ``False`` se já estavam juntos, ou seja, se a aresta
        (a, b) fecharia um ciclo.
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
        incluir_derivados: inclui as taxas de aprovação nos momentos em
            que já existem.
        excluir: atributos a deixar de fora.
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
        """Monta o grafo completo de informação mútua.

        ``idx`` são os alunos usados no cálculo (padrão: o treino).
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
        """Roda o Kruskal e devolve as arestas da árvore e seu peso."""
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

        Percorre as arestas da mais pesada para a mais leve (ou o
        contrário, com ``maximo=False``) e aceita cada uma que não fecha
        ciclo. Garante uma árvore geradora de peso máximo (ou mínimo).
        Empates são desfeitos pelo nome dos vértices, para o resultado
        ser sempre o mesmo.
        """
        if grafo.dirigido:
            raise ValueError("Kruskal exige grafo não dirigido")
        sinal = -1.0 if maximo else 1.0
        arestas = sorted(
            (ArvoreDependencias.ordenar_par(u, v) + (p,)
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
    # Robustez da árvore
    # ------------------------------------------------------------------
    def estabilidade(self, base, n: int = N_BOOTSTRAP,
                     semente: int = SEMENTE) -> dict[Aresta, float]:
        """Fração das reamostragens em que cada aresta da árvore volta.

        Sorteia os alunos com reposição, refaz a árvore e conta quantas
        vezes cada aresta original reaparece.
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
        return {ArvoreDependencias.ordenar_par(a[0], a[1]) for a in arestas}

    @staticmethod
    def ordenar_par(u: str, v: str) -> Aresta:
        """Par (u, v) em ordem alfabética, para identificar a aresta."""
        return (u, v) if str(u) <= str(v) else (v, u)

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
        """Grafo completo com peso = informação mútua entre cada par."""
        colunas = self._colunas(base)
        dados = base.dados.loc[idx, colunas]
        # Mesmas faixas usadas no G1, para os valores serem comparáveis.
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
        """Informação mútua, em bits, entre dois vetores de códigos.

        I(A;B) = soma p(a,b) · log2[p(a,b) / (p(a) p(b))], calculada pela
        tabela de contingência dos dois vetores.
        """
        nb = int(b.max()) + 1
        conjunta = np.bincount(a * nb + b, minlength=(int(a.max()) + 1) * nb)
        conjunta = conjunta.reshape(-1, nb) / len(a)
        pa = conjunta.sum(axis=1, keepdims=True)
        pb = conjunta.sum(axis=0, keepdims=True)
        nz = conjunta > 0
        razao = conjunta[nz] / (pa @ pb)[nz]
        return float(max((conjunta[nz] * np.log2(razao)).sum(), 0.0))
