"""G1: fluxo máximo e classificação pelo corte mínimo.

Responsável: Rafael.

Duas classes:

- ``EdmondsKarp``: algoritmo genérico de fluxo máximo (Ford-Fulkerson
  com caminhos aumentantes escolhidos por BFS, aula 6). Garante o fluxo
  máximo em O(V·E²); pelo teorema fluxo máximo = corte mínimo (Cormen
  et al., cap. 26), uma BFS na rede residual final devolve o lado da
  fonte do corte mínimo.
- ``ClassificadorCorteMinimo``: monta a rede do G1 (super-fonte S ligada
  aos evadidos do treino, super-sumidouro T ligado aos formados) e prevê
  evasão para os alunos do teste que ficam do lado de S (Blum e Chawla,
  2001).
"""

from collections import deque
from collections.abc import Hashable, Iterable

from grafo import Grafo

# Folga abaixo deste valor é tratada como zero (capacidades são floats).
EPSILON = 1e-12


class _Terminal:
    """Vértice artificial (S ou T), que nunca colide com um aluno."""

    def __init__(self, nome: str) -> None:
        self.nome = nome

    def __repr__(self) -> str:
        return self.nome


FONTE = _Terminal("S")
SUMIDOURO = _Terminal("T")


class EdmondsKarp:
    """Fluxo máximo por Edmonds-Karp e extração do corte mínimo.

    Uso::

        ek = EdmondsKarp()
        valor = ek.fluxo_maximo(rede, s, t)
        lado_s = ek.lado_da_fonte()

    Aresta não dirigida de capacidade c vira dois arcos de capacidade c.
    A rede recebida não é alterada: o algoritmo trabalha numa rede
    residual própria.
    """

    def __init__(self) -> None:
        self.valor: float | None = None
        self.num_caminhos = 0
        self._residual: dict[Hashable, dict[Hashable, float]] = {}
        self._s: Hashable = None

    def fluxo_maximo(self, rede: Grafo, s: Hashable,
                     t: Hashable) -> float:
        """Calcula e devolve o valor do fluxo máximo de ``s`` a ``t``."""
        if s not in rede or t not in rede:
            raise KeyError("fonte ou sumidouro fora da rede")
        if s == t:
            raise ValueError("fonte e sumidouro devem ser distintos")

        self._residual = self._montar_residual(rede)
        self._s = s
        self.num_caminhos = 0
        fluxo = 0.0

        while (pais := self._bfs(s, t)) is not None:
            caminho = self._reconstruir(pais, s, t)
            gargalo = min(self._residual[u][v] for u, v in caminho)
            if gargalo == float("inf"):
                raise ValueError("caminho de capacidade infinita entre "
                                 "a fonte e o sumidouro")
            for u, v in caminho:
                self._residual[u][v] -= gargalo
                self._residual[v][u] += gargalo
            fluxo += gargalo
            self.num_caminhos += 1

        self.valor = fluxo
        return fluxo

    def lado_da_fonte(self) -> set[Hashable]:
        """Vértices alcançáveis a partir de ``s`` na rede residual final.

        É o lado da fonte do corte mínimo. Exige ``fluxo_maximo`` antes.
        """
        if self.valor is None:
            raise RuntimeError("chame fluxo_maximo antes de lado_da_fonte")
        alcancados = {self._s}
        fila = deque([self._s])
        while fila:
            u = fila.popleft()
            for v, folga in self._residual[u].items():
                if folga > EPSILON and v not in alcancados:
                    alcancados.add(v)
                    fila.append(v)
        return alcancados

    @staticmethod
    def _montar_residual(
            rede: Grafo) -> dict[Hashable, dict[Hashable, float]]:
        """Copia as capacidades e cria os arcos de volta com folga 0."""
        residual: dict[Hashable, dict[Hashable, float]] = {
            v: {} for v in rede.vertices()}
        for u, v, capacidade in rede.arestas():
            residual[u][v] = residual[u].get(v, 0.0) + capacidade
            residual[v].setdefault(u, 0.0)
            if not rede.dirigido and u != v:
                residual[v][u] += capacidade
        return residual

    def _bfs(self, s: Hashable,
             t: Hashable) -> dict[Hashable, Hashable] | None:
        """Busca o caminho aumentante mais curto; devolve os pais."""
        pais: dict[Hashable, Hashable] = {s: s}
        fila = deque([s])
        while fila:
            u = fila.popleft()
            for v, folga in self._residual[u].items():
                if folga > EPSILON and v not in pais:
                    pais[v] = u
                    if v == t:
                        return pais
                    fila.append(v)
        return None

    @staticmethod
    def _reconstruir(pais: dict[Hashable, Hashable], s: Hashable,
                     t: Hashable) -> list[tuple[Hashable, Hashable]]:
        """Transforma o dicionário de pais na lista de arcos s -> t."""
        caminho = []
        v = t
        while v != s:
            caminho.append((pais[v], v))
            v = pais[v]
        caminho.reverse()
        return caminho


class ClassificadorCorteMinimo:
    """Classifica alunos sem rótulo pelo corte mínimo do G1.

    Rede: S -> evadidos do treino e formados do treino -> T, com
    capacidade infinita, mais as arestas de similaridade do grafo.
    Quem fica do lado de S no corte mínimo é previsto como evasão.

    Parâmetros:
        lam: peso do viés. Cada aluno sem rótulo ganha os arcos
            S -> u com capacidade lam·p e u -> T com lam·(1 - p), em que
            p é a taxa de evasão do treino. Como p < 0,5, o viés puxa
            para o lado de T. ``lam=0`` desliga o viés.
        contrair: funde os rotulados em S e T antes do fluxo. Não muda
            o corte e reduz muito a rede (3.630 alunos com k = 25: 728
            vértices, ~0,2 s). ``contrair=False`` serve só para
            verificação: na escala real leva vários minutos.
    """

    def __init__(self, lam: float = 1.0, contrair: bool = True) -> None:
        if lam < 0:
            raise ValueError("lam deve ser não negativo")
        self.lam = lam
        self.contrair = contrair
        self.fluxo: float | None = None
        self.num_caminhos = 0
        self.lado_evasao: set[Hashable] = set()
        self._grafo: Grafo | None = None

    def classificar(self, grafo: Grafo, rotulos_treino: dict[int, bool],
                    idx_teste: Iterable[int]) -> dict[int, bool]:
        """Devolve ``{aluno: True se previsto como evasão}``.

        Args:
            grafo: grafo de similaridade não dirigido (capacidades).
            rotulos_treino: ``True`` = evadiu, ``False`` = formou.
            idx_teste: alunos sem rótulo a classificar.
        """
        idx_teste = list(idx_teste)
        self._validar(grafo, rotulos_treino, idx_teste)

        rede = self._montar_rede(grafo, rotulos_treino, idx_teste)
        ek = EdmondsKarp()
        self.fluxo = ek.fluxo_maximo(rede, FONTE, SUMIDOURO)
        self.num_caminhos = ek.num_caminhos
        lado_s = ek.lado_da_fonte()

        self._grafo = grafo
        self.lado_evasao = {v for v in grafo.vertices()
                            if self._lado_de(v, lado_s, rotulos_treino)}
        return {u: u in self.lado_evasao for u in idx_teste}

    def arestas_cortadas(self) -> list[tuple[Hashable, Hashable, float]]:
        """Arestas do grafo original com uma ponta em cada lado."""
        if self._grafo is None:
            raise RuntimeError("chame classificar antes")
        return [(u, v, c) for u, v, c in self._grafo.arestas()
                if (u in self.lado_evasao) != (v in self.lado_evasao)]

    @staticmethod
    def _validar(grafo: Grafo, rotulos_treino: dict[int, bool],
                 idx_teste: list[int]) -> None:
        """Confere se rótulos e alunos a classificar fazem sentido."""
        if grafo.dirigido:
            raise ValueError("o grafo de similaridade deve ser não dirigido")
        if not any(rotulos_treino.values()) or all(rotulos_treino.values()):
            raise ValueError("o treino precisa ter evadidos e formados")
        if set(idx_teste) & rotulos_treino.keys():
            raise ValueError("aluno do teste também está no treino")
        fora = [v for v in [*rotulos_treino, *idx_teste] if v not in grafo]
        if fora:
            raise KeyError(f"alunos fora do grafo: {fora[:5]}")

    def _montar_rede(self, grafo: Grafo, rotulos_treino: dict[int, bool],
                     idx_teste: list[int]) -> Grafo:
        """Monta a rede de fluxo dirigida com S e T."""
        capacidades: dict[tuple[Hashable, Hashable], float] = {}

        def somar(a: Hashable, b: Hashable, c: float) -> None:
            if a != b and c > 0:
                capacidades[(a, b)] = capacidades.get((a, b), 0.0) + c

        def representante(v: Hashable) -> Hashable:
            if self.contrair and v in rotulos_treino:
                return FONTE if rotulos_treino[v] else SUMIDOURO
            return v

        for u, v, c in grafo.arestas():
            a, b = representante(u), representante(v)
            somar(a, b, c)
            somar(b, a, c)

        if not self.contrair:
            infinito = float("inf")
            for v, evadiu in rotulos_treino.items():
                if evadiu:
                    somar(FONTE, v, infinito)
                else:
                    somar(v, SUMIDOURO, infinito)

        if self.lam > 0:
            p = sum(rotulos_treino.values()) / len(rotulos_treino)
            for u in idx_teste:
                somar(FONTE, u, self.lam * p)
                somar(u, SUMIDOURO, self.lam * (1 - p))

        rede = Grafo(dirigido=True)
        rede.adicionar_vertice(FONTE)
        rede.adicionar_vertice(SUMIDOURO)
        for u in idx_teste:
            rede.adicionar_vertice(u)
        for (a, b), c in capacidades.items():
            rede.adicionar_aresta(a, b, c)
        return rede

    @staticmethod
    def _lado_de(v: Hashable, lado_s: set[Hashable],
                 rotulos_treino: dict[int, bool]) -> bool:
        """Indica se o vértice original ``v`` ficou do lado de S."""
        if v in rotulos_treino:
            return rotulos_treino[v]
        return v in lado_s
