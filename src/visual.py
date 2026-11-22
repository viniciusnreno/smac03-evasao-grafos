"""Montagem visual dos grafos para a interface (vis-network).

Converte os ``Grafo`` e os resultados dos modelos em listas de vértices
e arestas com cor, tamanho, rótulo e posição, e gera o HTML que a
interface exibe. Não depende do Streamlit: devolve dados e texto HTML.

O vis-network é o que vem empacotado no pyvis, embutido no HTML. Assim
a página funciona sem internet, o que importa na apresentação.
"""

import json
import math
from collections.abc import Hashable, Iterable, Mapping
from functools import lru_cache
from pathlib import Path

import pyvis

from grafo import Grafo

# Paleta (vermelho/azul distinguíveis por daltônicos, mais neutros).
COR_EVASAO = "#d1495b"
COR_FORMATURA = "#2e86ab"
COR_NEUTRA = "#a7b1bc"
COR_DESTAQUE = "#f4a259"
COR_SEGURA = "#3a9e5d"
COR_TEXTO = "#1f2933"
COR_ARESTA = "#c3ccd5"

# Cores de componentes do union-find (ciclo quando acabam).
CORES_COMPONENTES = (
    "#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#edc948",
    "#76b7b2", "#ff9da7", "#9c755f", "#e15759", "#bab0ac",
)

ALTURA_PADRAO = 560
INFINITO = float("inf")


@lru_cache(maxsize=1)
def _javascript_vis() -> str:
    """Código do vis-network empacotado no pyvis."""
    caminho = (Path(pyvis.__file__).parent / "lib" / "vis-9.1.2"
               / "vis-network.min.js")
    return caminho.read_text(encoding="utf-8")


def _rotulo_curto(texto: str, limite: int = 22) -> str:
    """Quebra rótulos longos em duas linhas para caberem no vértice."""
    if len(texto) <= limite:
        return texto
    corte = texto.rfind(" ", 0, limite)
    corte = corte if corte > 0 else limite
    return texto[:corte] + "\n" + texto[corte:].strip()


def _mistura(cor_a: str, cor_b: str, t: float) -> str:
    """Interpola duas cores hexadecimais (t = 0 dá a, t = 1 dá b)."""
    t = min(max(t, 0.0), 1.0)
    a = [int(cor_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(cor_b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}"
                         for x, y in zip(a, b))


def cor_risco(taxa: float) -> str:
    """Cor da taxa de evasão: azul (0) → cinza (0,5) → vermelho (1)."""
    if taxa <= 0.5:
        return _mistura(COR_FORMATURA, COR_NEUTRA, taxa / 0.5)
    return _mistura(COR_NEUTRA, COR_EVASAO, (taxa - 0.5) / 0.5)


def numero(x: float, casas: int = 2) -> str:
    """Número com vírgula decimal (padrão brasileiro)."""
    return f"{x:.{casas}f}".replace(".", ",")


def formatar_capacidade(c: float) -> str:
    """Capacidade com 2 casas, ou ∞."""
    return "∞" if c == INFINITO else numero(c)


class MontadorVisual:
    """Converte os grafos do projeto em HTML interativo.

    Cada método ``*_g1``, ``*_g2``, ``*_g3``, ``rede_fluxo`` e
    ``passo_kruskal`` devolve ``(vertices, arestas, opcoes)``, que
    ``html`` transforma na página. Os vértices e arestas seguem o
    formato do vis-network (``id``, ``label``, ``color``, ``from``,
    ``to``...).
    """

    def __init__(self, altura: int = ALTURA_PADRAO) -> None:
        self.altura = altura

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------
    def html(self, vertices: list[dict], arestas: list[dict],
             opcoes: dict | None = None) -> str:
        """Página com o grafo desenhado pelo vis-network."""
        opcoes = opcoes or {}
        dados = {
            "vertices": vertices,
            "arestas": arestas,
            "opcoes": opcoes,
        }
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<style>*{box-sizing:border-box}html,body{margin:0;"
            "padding:0;background:#fff;overflow:hidden;"
            "font-family:sans-serif}#rede{width:100%;"
            f"height:{self.altura}px;border:1px solid #e3e7eb;"
            "border-radius:8px}</style></head><body>"
            "<div id='rede'></div>"
            f"<script>{_javascript_vis()}</script>"
            "<script>const d = " + json.dumps(dados, ensure_ascii=False,
                                              default=str)
            + ";new vis.Network(document.getElementById('rede'),"
            "{nodes: new vis.DataSet(d.vertices),"
            "edges: new vis.DataSet(d.arestas)}, d.opcoes);"
            "</script></body></html>"
        )

    @staticmethod
    def _opcoes(fisica: bool = True, dirigido: bool = False,
                fixo: bool = False) -> dict:
        """Opções comuns: física, setas e interação."""
        opcoes = {
            "interaction": {"hover": True, "tooltipDelay": 80,
                            "navigationButtons": False},
            "nodes": {"font": {"color": COR_TEXTO, "size": 14},
                      "borderWidth": 1.5},
            "edges": {"smooth": False,
                      "arrows": {"to": {"enabled": dirigido,
                                        "scaleFactor": 0.6}}},
            "physics": {"enabled": fisica},
        }
        if fisica:
            opcoes["physics"].update({
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {"gravitationalConstant": -60,
                                     "springLength": 110,
                                     "avoidOverlap": 0.4},
                "stabilization": {"iterations": 250},
            })
        if fixo:
            opcoes["interaction"]["dragNodes"] = False
        return opcoes

    # ------------------------------------------------------------------
    # G1 · ego-rede de um aluno
    # ------------------------------------------------------------------
    def ego_g1(self, grafo: Grafo, aluno: int,
               rotulos_treino: Mapping[int, bool],
               previsoes: Mapping[int, bool],
               lado_evasao: set[int],
               evadiu_real: Mapping[int, bool]) -> tuple:
        """Aluno no centro, seus vizinhos no kNN e as arestas cortadas.

        Args:
            grafo: kNN de Gower (capacidade = 1 - distância).
            aluno: vértice central (aluno de teste).
            rotulos_treino: rótulos que o algoritmo conhecia.
            previsoes: previsões do corte para os alunos sem rótulo.
            lado_evasao: vértices do lado de S no corte mínimo.
            evadiu_real: desfecho real dos alunos sem rótulo, mostrado
                só depois da classificação (borda do vértice).
        """
        ego = [aluno, *sorted(grafo.vizinhos(aluno))]
        conjunto = set(ego)
        vertices = []
        for v in ego:
            vertices.append(self._vertice_aluno(
                v, v == aluno, rotulos_treino, previsoes, evadiu_real))

        arestas = []
        for i, u in enumerate(ego):
            for v, c in grafo.vizinhos(u).items():
                if v not in conjunto or ego.index(v) <= i:
                    continue
                cortada = (u in lado_evasao) != (v in lado_evasao)
                central = aluno in (u, v)
                arestas.append({
                    "from": u, "to": v,
                    "width": 1 + 5 * c if central else 0.6 + 2 * c,
                    "color": {"color": COR_EVASAO if cortada else (
                        "#8a96a3" if central else COR_ARESTA),
                        "opacity": 1.0 if central or cortada else 0.45},
                    "dashes": cortada,
                    "title": (f"capacidade 1 − d = {c:.3f}"
                              + ("\naresta cortada" if cortada else "")),
                })
        return vertices, arestas, self._opcoes(fisica=True)

    @staticmethod
    def _vertice_aluno(v: int, central: bool,
                       rotulos_treino: Mapping[int, bool],
                       previsoes: Mapping[int, bool],
                       evadiu_real: Mapping[int, bool]) -> dict:
        """Vértice de aluno: cor pelo rótulo (treino) ou pela previsão."""
        if v in rotulos_treino:
            evadiu = rotulos_treino[v]
            cor = COR_EVASAO if evadiu else COR_FORMATURA
            borda = cor
            papel = "treino: " + ("evadiu" if evadiu else "formou")
            forma = "dot"
        else:
            prev = previsoes.get(v, False)
            real = evadiu_real.get(v)
            cor = _mistura(COR_EVASAO if prev else COR_FORMATURA,
                           "#ffffff", 0.35)
            borda = (COR_EVASAO if real else COR_FORMATURA) \
                if real is not None else COR_TEXTO
            papel = ("teste: previsto " + ("evasão" if prev else
                                           "formatura"))
            if real is not None:
                papel += "; real " + ("evasão" if real else "formatura")
            forma = "diamond"
        vertice = {
            "id": v,
            "label": f"aluno {v}" if central else str(v),
            "title": f"aluno {v}\n{papel}",
            "shape": "star" if central else forma,
            "size": 34 if central else 15,
            "color": {"background": cor, "border": borda},
            "borderWidth": 4 if central or v not in rotulos_treino else 1,
            "font": {"size": 18 if central else 11},
        }
        if central:
            vertice.update({"fixed": True, "x": 0, "y": 0})
        return vertice

    # ------------------------------------------------------------------
    # G2 · árvore de dependências
    # ------------------------------------------------------------------
    def arvore_g2(self, arestas_arvore: Iterable[tuple[str, str, float]],
                  vertices: Iterable[str], desfecho: str,
                  estabilidade: Mapping[tuple[str, str], float]
                  | None = None) -> tuple:
        """Árvore de IM: espessura = IM, opacidade = estabilidade."""
        arestas_arvore = list(arestas_arvore)
        estabilidade = estabilidade or {}
        vizinhos = {u if v == desfecho else v for u, v, _ in arestas_arvore
                    if desfecho in (u, v)}
        maior = max((p for _, _, p in arestas_arvore), default=1.0)
        nos = []
        for v in vertices:
            if v == desfecho:
                nos.append({"id": v, "label": "Desfecho", "shape": "box",
                            "color": {"background": COR_EVASAO,
                                      "border": COR_EVASAO},
                            "font": {"color": "#ffffff", "size": 20},
                            "fixed": True, "x": 0, "y": 0,
                            "title": "vértice do desfecho (evadiu?)"})
            else:
                destaque = v in vizinhos
                nos.append({
                    "id": v, "label": _rotulo_curto(v),
                    "shape": "box" if destaque else "ellipse",
                    "color": {"background": COR_DESTAQUE if destaque
                              else "#eef1f4",
                              "border": "#b26b1f" if destaque
                              else "#9aa5b1"},
                    "font": {"size": 15 if destaque else 12},
                    "title": v + ("\nvizinho do Desfecho" if destaque
                                  else ""),
                })
        arestas = []
        for u, v, p in arestas_arvore:
            est = estabilidade.get(tuple(sorted((u, v))))
            toca = desfecho in (u, v)
            titulo = f"{u} — {v}\nIM = {p:.4f} bits"
            if est is not None:
                titulo += f"\nestabilidade (bootstrap) = {est:.0%}"
            arestas.append({
                "from": u, "to": v,
                "width": 1 + 9 * math.sqrt(p / maior),
                "color": {"color": COR_EVASAO if toca else "#6b7785",
                          "opacity": 0.3 + 0.7 * (est if est is not None
                                                  else 1.0)},
                "title": titulo,
            })
        opcoes = self._opcoes(fisica=True)
        opcoes["physics"]["forceAtlas2Based"]["springLength"] = 90
        return nos, arestas, opcoes

    # ------------------------------------------------------------------
    # G2 · um passo do Kruskal
    # ------------------------------------------------------------------
    def passo_kruskal(self, vertices: list[str], desfecho: str,
                      aceitas: Iterable[tuple[str, str, float]],
                      atual: tuple[str, str, float, bool] | None,
                      componentes: Mapping[str, Hashable]) -> tuple:
        """Estado do Kruskal: árvore parcial, aresta da vez e componentes.

        Vértices em círculo fixo (o Desfecho no centro) para o desenho
        não mudar entre os passos. Cada componente do union-find com
        mais de um vértice ganha uma cor.
        """
        tamanhos: dict[Hashable, int] = {}
        for r in componentes.values():
            tamanhos[r] = tamanhos.get(r, 0) + 1
        cores: dict[Hashable, str] = {}
        for v in vertices:
            r = componentes[v]
            if tamanhos[r] > 1 and r not in cores:
                cores[r] = CORES_COMPONENTES[
                    len(cores) % len(CORES_COMPONENTES)]

        outros = [v for v in vertices if v != desfecho]
        raio_x, raio_y = 520, 330
        nos = []
        for i, v in enumerate(outros):
            ang = 2 * math.pi * i / len(outros)
            cor = cores.get(componentes[v], "#eef1f4")
            nos.append({"id": v, "label": _rotulo_curto(v, 18),
                        "x": raio_x * math.cos(ang),
                        "y": raio_y * math.sin(ang), "shape": "dot",
                        "size": 9, "color": {"background": cor,
                                             "border": "#6b7785"},
                        "font": {"size": 11}, "title": v})
        cor_d = cores.get(componentes[desfecho], "#eef1f4")
        nos.append({"id": desfecho, "label": "Desfecho", "x": 0, "y": 0,
                    "shape": "box", "font": {"size": 16},
                    "color": {"background": cor_d,
                              "border": COR_EVASAO},
                    "borderWidth": 3})

        arestas = [{"from": u, "to": v, "width": 2.5,
                    "color": {"color": "#4a5560"},
                    "title": f"{u} — {v}\nIM = {p:.4f}"}
                   for u, v, p in aceitas]
        if atual is not None:
            u, v, p, aceita = atual
            arestas.append({
                "from": u, "to": v, "width": 6,
                "color": {"color": COR_SEGURA if aceita else COR_EVASAO},
                "dashes": not aceita,
                "label": numero(p, 3),
                "font": {"size": 14, "strokeWidth": 4},
                "title": ("aceita: une dois componentes" if aceita else
                          "rejeitada: fecharia um ciclo"),
            })
        return nos, arestas, self._opcoes(fisica=False, fixo=True)

    # ------------------------------------------------------------------
    # G3 · DAG de trajetórias
    # ------------------------------------------------------------------
    def dag_g3(self, grafo: Grafo, ordem: list[str],
               contagem: Mapping[str, int], evasoes: Mapping[str, int],
               min_suporte: int,
               risco: list[str] | None = None,
               segura: list[str] | None = None,
               virada: tuple[str, str] | None = None,
               caminho_usuario: list[str] | None = None) -> tuple:
        """DAG em camadas fixas, da esquerda (INICIO) para a direita.

        A camada vem do prefixo do estado (``L0:``...). Dentro de cada
        camada os estados ficam em ordem de taxa de evasão (baixo =
        mais risco). O número antes do nome é a posição na ordem
        topológica de Kahn. Cor = taxa de evasão do estado no treino.
        """
        posicao_kahn = {v: i for i, v in enumerate(ordem)}
        camadas: dict[int, list[str]] = {}
        for v in grafo.vertices():
            camadas.setdefault(self._camada(v), []).append(v)

        def taxa(v: str) -> float:
            return evasoes[v] / contagem[v] if contagem[v] else 0.0

        pos: dict[str, tuple[float, float]] = {}
        for c, membros in camadas.items():
            membros.sort(key=taxa)
            for i, v in enumerate(membros):
                pos[v] = (c * 250, (i - (len(membros) - 1) / 2) * 95)

        realce: dict[tuple[str, str], str] = {}
        for caminho, cor in ((segura, COR_SEGURA), (risco, COR_EVASAO),
                             (caminho_usuario, COR_DESTAQUE)):
            for a, b in zip(caminho or [], (caminho or [])[1:]):
                realce[(a, b)] = cor
        no_caminho = {v for a, b in realce for v in (a, b)}

        maior = max(contagem.values())
        nos = []
        for v in grafo.vertices():
            x, y = pos[v]
            t = taxa(v)
            nome = v.split(":", 1)[-1]
            nos.append({
                "id": v,
                "label": f"{posicao_kahn[v] + 1}. {_rotulo_curto(nome, 18)}",
                "x": x, "y": y, "shape": "dot",
                "size": 8 + 22 * math.sqrt(contagem[v] / maior),
                "color": {"background": cor_risco(t),
                          "border": COR_TEXTO if v in no_caminho
                          else "#6b7785"},
                "borderWidth": 3 if v in no_caminho else 1,
                "font": {"size": 12, "vadjust": -4},
                "title": (f"{v}\n{contagem[v]} alunos no treino\n"
                          f"evasão {t:.1%}\nordem de Kahn: "
                          f"{posicao_kahn[v] + 1}"),
            })
        arestas = []
        for u, v, n in grafo.arestas():
            cor = realce.get((u, v))
            fraco = n < min_suporte
            e_virada = virada is not None and (u, v) == tuple(virada)
            arestas.append({
                "from": u, "to": v,
                "width": (4.5 if cor else 0.4 + 3 * math.sqrt(n / maior)),
                "color": {"color": cor or ("#e3e7eb" if fraco
                                           else "#9aa5b1"),
                          "opacity": 1.0 if cor else (0.35 if fraco
                                                      else 0.6)},
                "dashes": e_virada,
                "title": f"{u} → {v}\n{int(n)} alunos"
                         + ("\n(abaixo do suporte mínimo)" if fraco
                            else "")
                         + ("\nponto de virada" if e_virada else ""),
            })
        opcoes = self._opcoes(fisica=False, dirigido=True)
        opcoes["edges"]["smooth"] = {"enabled": True, "type": "cubicBezier",
                                     "forceDirection": "horizontal",
                                     "roundness": 0.4}
        return nos, arestas, opcoes

    @staticmethod
    def _camada(estado: str) -> int:
        """INICIO = 0, L0 = 1, ..., L4 = 5."""
        if ":" not in estado:
            return 0
        return int(estado.split(":", 1)[0][1:]) + 1

    # ------------------------------------------------------------------
    # Edmonds–Karp · rede de fluxo e residual
    # ------------------------------------------------------------------
    def rede_fluxo(self, rede: Grafo, fonte: Hashable,
                   sumidouro: Hashable,
                   residual: Mapping[Hashable, Mapping[Hashable, float]],
                   rotulos: Mapping[Hashable, bool],
                   caminho: Iterable[tuple[Hashable, Hashable]] = (),
                   lado_s: set[Hashable] | None = None) -> tuple:
        """Rede de fluxo com o fluxo atual em cada arco (f / c).

        S fica à esquerda e T à direita; os evadidos do treino perto de
        S, os formados perto de T e os alunos sem rótulo no meio. Os
        dois sentidos de uma aresta de similaridade viram um só traço
        com o fluxo líquido. ``lado_s`` (no fim) pinta o corte mínimo.
        """
        colunas = {True: -1, None: 0, False: 1}
        grupos: dict[int, list[Hashable]] = {-1: [], 0: [], 1: []}
        for v in rede.vertices():
            if v in (fonte, sumidouro):
                continue
            grupos[colunas[rotulos.get(v)]].append(v)

        nos = [self._terminal(fonte, "S", -560, COR_EVASAO, lado_s),
               self._terminal(sumidouro, "T", 560, COR_FORMATURA, lado_s)]
        for col, membros in grupos.items():
            membros.sort(key=str)
            for i, v in enumerate(membros):
                rot = rotulos.get(v)
                if rot is None:
                    cor = "#ffffff"
                    papel = "sem rótulo"
                else:
                    cor = COR_EVASAO if rot else COR_FORMATURA
                    papel = "treino: " + ("evadiu" if rot else "formou")
                borda = COR_TEXTO
                if lado_s is not None and rot is None:
                    cor = _mistura(COR_EVASAO if v in lado_s
                                   else COR_FORMATURA, "#ffffff", 0.3)
                nos.append({
                    "id": str(v), "label": str(v), "shape": "circle",
                    "x": col * 300, "y": (i - (len(membros) - 1) / 2) * 110,
                    "color": {"background": cor, "border": borda},
                    "font": {"color": COR_TEXTO if cor == "#ffffff"
                             or rot is None else "#ffffff"},
                    "title": f"aluno {v}\n{papel}",
                })

        no_caminho = {(str(u), str(v)) for u, v in caminho}
        arestas = []
        vistos: set[tuple[Hashable, Hashable]] = set()
        for u, v, c in rede.arestas():
            if (v, u) in vistos:
                continue
            vistos.add((u, v))
            volta = rede.peso(v, u) if rede.tem_aresta(v, u) else 0.0
            # Fluxo líquido u -> v = o que sobrou de folga no arco de
            # volta além da capacidade dele (vale também para c = ∞).
            fluxo = residual[v][u] - volta
            de, para = str(u), str(v)
            if volta and fluxo < 0:          # fluxo líquido no outro sentido
                fluxo, c, de, para = -fluxo, volta, para, de
            ativo = ((de, para) in no_caminho
                     or (para, de) in no_caminho)
            cortado = (lado_s is not None
                       and (u in lado_s) != (v in lado_s))
            cheio = c != INFINITO and fluxo >= c - 1e-9
            terminal = fonte in (u, v) or sumidouro in (u, v)
            if ativo:
                cor = COR_DESTAQUE
            elif cortado:
                cor = COR_EVASAO
            elif fluxo > 1e-9:
                cor = "#4a5560"
            else:
                cor = COR_ARESTA
            rotular = ativo or cortado or fluxo > 1e-9
            arestas.append({
                "from": de, "to": para,
                "label": (f"{numero(fluxo)}/{formatar_capacidade(c)}"
                          if rotular else " "),
                "width": 5 if ativo else (3 if cheio else 1.4),
                "color": {"color": cor},
                "dashes": cortado,
                "arrows": {"to": {"enabled": fluxo > 1e-9 or terminal,
                                  "scaleFactor": 0.5}},
                "font": {"size": 11, "strokeWidth": 3,
                         "color": COR_TEXTO, "align": "middle"},
                "title": (f"{de} → {para}\nfluxo {fluxo:.3f} de "
                          f"{formatar_capacidade(c)}"
                          + ("\nsaturada" if cheio else "")
                          + ("\naresta do corte" if cortado else "")),
                # Viés S/T -> sem rótulo: curva forte, para não passar
                # por cima dos rotulados da coluna do meio.
                "smooth": {"enabled": True, "type": "curvedCW",
                           "roundness": 0.3 if terminal and c != INFINITO
                           else 0.08},
            })
        opcoes = self._opcoes(fisica=False, dirigido=False, fixo=True)
        return nos, arestas, opcoes

    @staticmethod
    def _terminal(v: Hashable, nome: str, x: float, cor: str,
                  lado_s: set[Hashable] | None) -> dict:
        """Vértice S ou T."""
        return {"id": str(v), "label": nome, "shape": "circle",
                "x": x, "y": 0, "size": 30,
                "color": {"background": cor, "border": COR_TEXTO},
                "font": {"color": "#ffffff", "size": 22},
                "title": ("fonte: lado da evasão" if nome == "S"
                          else "sumidouro: lado da formatura")}
