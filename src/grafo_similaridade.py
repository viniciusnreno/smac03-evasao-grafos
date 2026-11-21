"""G1: grafo de similaridade. Classe: ``GrafoSimilaridade(ModeloGrafo)``.

Responsável: Carlos (implementado por Rafael enquanto Carlos não
iniciava a etapa).

- ``construir``: distância de Gower ponderada pela informação mútua
  (só treino) e kNN com capacidade c(i, j) = 1 - d(i, j).
- ``executar``: chama ``ClassificadorCorteMinimo`` e devolve as
  previsões.
- ``destaques``: arestas do corte e alunos do lado da evasão.

Distância de Gower (Gower, 1971), Eq. 1 do relatório::

    d(i, j) = soma_a w_a · delta_a(i, j) / soma_a w_a

    delta_a = |x_ia - x_ja| / R_a   (numérica; R_a = amplitude no treino)
    delta_a = [x_ia != x_ja]        (nominal)

Valor ausente (taxa de aprovação sem inscrição) é tratado como uma
categoria própria: dois ausentes têm delta 0, um só ausente tem delta 1.

kNN: a aresta {i, j} existe se j está entre os k mais próximos de i ou
i entre os k mais próximos de j. O grafo não dirigido grava os dois
sentidos, então o kNN sai simétrico. Os vértices são todos os alunos
(treino rotulado e teste sem rótulo, como em Blum e Chawla, 2001), mas
pesos e amplitudes vêm só do treino e os rótulos do teste nunca são
lidos.
"""

from collections.abc import Hashable, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from corte_minimo import ClassificadorCorteMinimo
from dados import DERIVADOS, NOMINAIS, faixa_aprovacao
from grafo import Grafo
from modelo import ModeloGrafo

# k escolhido por CV no treino (plano v2): 25 em T0 e T1, 40 em T2.
K_PADRAO: dict[str, int] = {"T0": 25, "T1": 25, "T2": 40}

# Discretização usada só para calcular a informação mútua (pesos).
# Categoria nominal com menos alunos que isto no treino vira "outros".
MIN_POR_CATEGORIA = 30
# Numérica com mais valores distintos que isto é cortada em quantis.
MAX_VALORES_DISCRETOS = 10
NUM_QUANTIS = 4

# Linhas da matriz de distâncias calculadas de cada vez (memória).
TAMANHO_BLOCO = 512

_OUTROS = "__outros__"


class GrafoSimilaridade(ModeloGrafo):
    """G1: kNN de Gower entre alunos, classificado por corte mínimo.

    Parâmetros:
        momento: ``"T0"``, ``"T1"`` ou ``"T2"``.
        k: vizinhos por aluno (``K_PADRAO`` tem os valores do plano).
        pesos: ``None`` calcula a informação mútua de cada atributo com
            o desfecho, só no treino. Um dict ``{atributo: peso}`` fixa
            os pesos e usa só os atributos listados.
        lam: viés do ``ClassificadorCorteMinimo`` (0 desliga).
        incluir_derivados: inclui ``taxa_aprov_s1``/``taxa_aprov_s2``
            nos momentos em que existem.
        excluir: atributos a retirar (ablação). Com ``pesos=None`` a
            informação mútua continua sendo calculada no treino dado.

    Depois de ``construir`` ficam disponíveis ``grafo``, ``pesos_``
    (normalizados, soma 1), ``amplitudes_``, ``rotulos_`` (treino) e
    ``idx_sem_rotulo_``.
    """

    nome = "G1 · Similaridade + corte mínimo"
    descricao = ("Alunos ligados aos k mais parecidos pela distância de "
                 "Gower. O corte mínimo (Edmonds-Karp) separa o lado da "
                 "evasão do lado da formatura.")

    def __init__(self, momento: str, k: int,
                 pesos: Mapping[str, float] | None = None,
                 lam: float = 1.0, incluir_derivados: bool = True,
                 excluir: Iterable[str] = ()) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("k deve ser pelo menos 1")
        self.momento = momento
        self.k = k
        self.pesos = dict(pesos) if pesos is not None else None
        self.lam = lam
        self.incluir_derivados = incluir_derivados
        self.excluir = frozenset(excluir)

        self.pesos_: dict[str, float] = {}
        self.amplitudes_: dict[str, float] = {}
        self.rotulos_: dict[int, bool] = {}
        self.idx_sem_rotulo_: list[int] = []
        self._vertices: np.ndarray = np.empty(0, dtype=int)
        self._numericas: np.ndarray = np.empty((0, 0))
        self._nominais: np.ndarray = np.empty((0, 0))
        self._w_num: np.ndarray = np.empty(0)
        self._w_nom: np.ndarray = np.empty(0)
        self._amp: np.ndarray = np.empty(0)
        self._posicao: dict[int, int] = {}
        self._classificador: ClassificadorCorteMinimo | None = None

    # ------------------------------------------------------------------
    # Interface ModeloGrafo
    # ------------------------------------------------------------------
    def construir(self, base, idx_treino: Iterable[int] | None = None,
                  idx_teste: Iterable[int] | None = None) -> Grafo:
        """Monta o kNN de Gower com os alunos de treino e de teste.

        Args:
            base: ``BaseDeDados``.
            idx_treino: alunos rotulados; padrão ``base.idx_treino``.
                Na CV, as dobras de treino.
            idx_teste: alunos sem rótulo; padrão ``base.idx_teste``.
                Na CV, a dobra de validação. Pode ser vazio.
        """
        idx_treino = sorted(int(i) for i in (
            base.idx_treino if idx_treino is None else idx_treino))
        idx_teste = sorted(int(i) for i in (
            base.idx_teste if idx_teste is None else idx_teste))
        if not idx_treino:
            raise ValueError("o treino não pode ser vazio")
        if set(idx_treino) & set(idx_teste):
            raise ValueError("aluno do teste também está no treino")

        colunas = self._colunas(base)
        x_treino = base.X(idx_treino, self.momento, self.incluir_derivados)
        y_treino = base.y(idx_treino)
        pesos = self._pesos(x_treino[colunas], y_treino)
        colunas = [c for c in colunas if pesos[c] > 0]
        if not colunas:
            raise ValueError("nenhum atributo com peso positivo")

        self.rotulos_ = base.rotulos(idx_treino)
        self.idx_sem_rotulo_ = idx_teste
        self._preparar(base, idx_treino, idx_teste, colunas, pesos)
        self.grafo = self._knn()
        self._classificador = None
        return self.grafo

    def executar(self) -> dict:
        """Classifica os alunos sem rótulo pelo corte mínimo.

        Devolve ``previsoes`` (``{aluno: True se evasão}``), ``fluxo``,
        ``num_caminhos``, o tamanho do grafo e os parâmetros usados.
        """
        if self.grafo is None:
            raise RuntimeError("chame construir antes de executar")
        if not self.idx_sem_rotulo_:
            raise RuntimeError("não há alunos sem rótulo para classificar")
        classificador = ClassificadorCorteMinimo(lam=self.lam)
        previsoes = classificador.classificar(
            self.grafo, self.rotulos_, self.idx_sem_rotulo_)
        self._classificador = classificador
        return {
            "momento": self.momento,
            "k": self.k,
            "lam": self.lam,
            "previsoes": previsoes,
            "fluxo": classificador.fluxo,
            "num_caminhos": classificador.num_caminhos,
            "num_vertices": len(self.grafo),
            "num_arestas": self.grafo.num_arestas(),
            "pesos": dict(self.pesos_),
        }

    def destaques(self) -> dict[str, list[Hashable]]:
        """Alunos do lado da evasão e arestas cortadas pelo corte mínimo."""
        if self._classificador is None:
            raise RuntimeError("chame executar antes de destaques")
        return {
            "vertices": sorted(self._classificador.lado_evasao),
            "arestas": [(u, v) for u, v, _ in
                        self._classificador.arestas_cortadas()],
        }

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------
    def distancia(self, i: int, j: int) -> float:
        """Distância de Gower entre os alunos ``i`` e ``j`` do grafo."""
        if self.grafo is None:
            raise RuntimeError("chame construir antes")
        pi, pj = self._posicao[i], self._posicao[j]
        return float(self._distancias(np.array([pi]))[0, pj])

    def assortatividade(self,
                        rotulos: Mapping[int, bool] | None = None) -> float:
        """Assortatividade de Newman (2003) do desfecho no grafo.

        r = (soma_i e_ii - soma_i a_i²) / (1 - soma_i a_i²), em que
        e_ij é a fração de pontas de aresta que ligam a classe i à j e
        a_i = soma_j e_ij. Arestas sem peso; só entram as arestas com as
        duas pontas rotuladas. r > 0: alunos parecidos têm o mesmo
        desfecho com mais frequência que o acaso.

        Args:
            rotulos: ``{aluno: evadiu}``; padrão ``rotulos_`` (treino).
                Rótulos embaralhados servem ao teste de permutação.
        """
        if self.grafo is None:
            raise RuntimeError("chame construir antes")
        rotulos = self.rotulos_ if rotulos is None else rotulos
        e = np.zeros((2, 2))
        for u, v, _ in self.grafo.arestas():
            if u in rotulos and v in rotulos:
                a, b = int(rotulos[u]), int(rotulos[v])
                e[a, b] += 1
                e[b, a] += 1
        if e.sum() == 0:
            raise ValueError("nenhuma aresta entre alunos rotulados")
        e /= e.sum()
        esperado = float((e.sum(axis=1) ** 2).sum())
        if esperado == 1.0:
            raise ValueError("todos os rótulos são iguais")
        return (float(np.trace(e)) - esperado) / (1.0 - esperado)

    # ------------------------------------------------------------------
    # Auxiliares
    # ------------------------------------------------------------------
    def _colunas(self, base) -> list[str]:
        """Atributos do momento, sem os excluídos (e só os de ``pesos``)."""
        disponiveis = base.atributos(self.momento, self.incluir_derivados)
        desconhecidos = self.excluir - set(disponiveis)
        if self.pesos is not None:
            desconhecidos |= self.pesos.keys() - set(disponiveis)
        if desconhecidos:
            raise ValueError(f"atributos fora do momento {self.momento}: "
                             f"{sorted(desconhecidos)}")
        colunas = [c for c in disponiveis if c not in self.excluir]
        if self.pesos is not None:
            colunas = [c for c in colunas if c in self.pesos]
        return colunas

    def _pesos(self, x_treino: pd.DataFrame,
               y_treino: np.ndarray) -> dict[str, float]:
        """Peso de cada atributo: o dado ou a informação mútua (treino)."""
        if self.pesos is not None:
            brutos = {c: float(self.pesos[c]) for c in x_treino.columns}
            if any(p < 0 for p in brutos.values()):
                raise ValueError("pesos devem ser não negativos")
        else:
            brutos = {c: self.informacao_mutua(x_treino[c], y_treino)
                      for c in x_treino.columns}
        return brutos

    @staticmethod
    def informacao_mutua(serie: pd.Series, y: np.ndarray) -> float:
        """Informação mútua, em bits, entre um atributo e o desfecho.

        O atributo é discretizado antes: taxas de aprovação usam as
        faixas do DAG (``dados.FAIXAS_APROVACAO``); categorias nominais raras
        (< ``MIN_POR_CATEGORIA``) viram "outros"; numéricas com muitos
        valores viram ``NUM_QUANTIS`` faixas de quantil; ausente é uma
        categoria própria. Chamar só com dados de treino.

        Reproduz os valores do ``analise_v2`` para as taxas de aprovação
        (0,381 e 0,474 bits), a mensalidade (0,149), o curso (0,090), a
        bolsa (0,083) e os atributos binários.
        """
        codigos = GrafoSimilaridade.discretizar(serie)
        return float(mutual_info_score(np.asarray(y, dtype=int), codigos)
                     / np.log(2))

    @staticmethod
    def discretizar(serie: pd.Series) -> np.ndarray:
        """Códigos inteiros das faixas do atributo (base da IM).

        Mesmas regras de ``informacao_mutua``; usado também pelo G2
        (``ArvoreDependencias``) para a IM entre pares de atributos.
        """
        if serie.name in DERIVADOS:
            serie = pd.Series(faixa_aprovacao(serie))
        elif serie.name in NOMINAIS:
            contagem = serie.value_counts()
            raros = contagem.index[contagem < MIN_POR_CATEGORIA]
            serie = serie.astype(object).where(~serie.isin(raros), _OUTROS)
        elif serie.nunique() > MAX_VALORES_DISCRETOS:
            serie = pd.qcut(serie, NUM_QUANTIS, duplicates="drop")
        codigos, _ = pd.factorize(serie, use_na_sentinel=False)
        return codigos

    def _preparar(self, base, idx_treino: list[int], idx_teste: list[int],
                  colunas: list[str], pesos: dict[str, float]) -> None:
        """Guarda as matrizes de atributos, os pesos e as amplitudes."""
        self._vertices = np.array(sorted(idx_treino + idx_teste))
        self._posicao = {int(v): p for p, v in enumerate(self._vertices)}
        dados = base.dados.loc[self._vertices, colunas]
        treino = base.dados.loc[idx_treino, colunas]

        numericas = [c for c in colunas if c not in NOMINAIS]
        nominais = [c for c in colunas if c in NOMINAIS]
        total = sum(pesos[c] for c in colunas)
        self.pesos_ = {c: pesos[c] / total for c in colunas}

        amp = (treino[numericas].max() - treino[numericas].min()).fillna(0)
        self.amplitudes_ = {c: float(amp[c]) for c in numericas}
        self._numericas = dados[numericas].to_numpy(dtype=float)
        self._amp = amp.to_numpy(dtype=float)
        # Códigos inteiros; ausente vira -1 (categoria própria).
        self._nominais = np.column_stack(
            [pd.factorize(dados[c])[0] for c in nominais]
        ) if nominais else np.empty((len(dados), 0), dtype=int)
        self._w_num = np.array([self.pesos_[c] for c in numericas])
        self._w_nom = np.array([self.pesos_[c] for c in nominais])

    def _distancias(self, linhas: np.ndarray) -> np.ndarray:
        """Gower das ``linhas`` (posições) contra todos os vértices."""
        n = len(self._vertices)
        d = np.zeros((len(linhas), n))
        for a, (w, r) in enumerate(zip(self._w_num, self._amp)):
            col = self._numericas[:, a]
            xi, xj = col[linhas, None], col[None, :]
            ausente_i, ausente_j = np.isnan(xi), np.isnan(xj)
            if r > 0:
                delta = np.minimum(np.abs(xi - xj) / r, 1.0)
            else:
                delta = np.zeros((len(linhas), n))
            delta = np.where(ausente_i | ausente_j,
                             (ausente_i != ausente_j).astype(float), delta)
            d += w * delta
        for a, w in enumerate(self._w_nom):
            col = self._nominais[:, a]
            d += w * (col[linhas, None] != col[None, :])
        return d

    def _knn(self) -> Grafo:
        """Liga cada aluno aos k mais próximos (união: kNN simétrico)."""
        n = len(self._vertices)
        k = min(self.k, n - 1)
        grafo = Grafo(dirigido=False)
        for v in self._vertices:
            grafo.adicionar_vertice(int(v))
        for inicio in range(0, n, TAMANHO_BLOCO):
            linhas = np.arange(inicio, min(inicio + TAMANHO_BLOCO, n))
            d = self._distancias(linhas)
            d[np.arange(len(linhas)), linhas] = np.inf
            # Ordenação estável: empate fica com o aluno de menor número.
            vizinhos = np.argsort(d, axis=1, kind="stable")[:, :k]
            for p, viz in zip(linhas, vizinhos):
                u = int(self._vertices[p])
                for q, dist in zip(viz, d[p - inicio, viz]):
                    grafo.adicionar_aresta(u, int(self._vertices[q]),
                                           1.0 - float(dist))
        return grafo
