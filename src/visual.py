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

from figuras import ROTULOS
from grafo import Grafo

# Paleta (vermelho/azul distinguíveis por daltônicos, mais neutros).
COR_EVASAO = "#d1495b"
COR_FORMATURA = "#2e86ab"
COR_NEUTRA = "#a7b1bc"
COR_DESTAQUE = "#f4a259"
COR_SEGURA = "#3a9e5d"
COR_TEXTO = "#1f2933"
COR_ARESTA = "#c3ccd5"
COR_FUTURO = "#e9edf1"     # camadas ainda não observadas no momento
FUNDO_ESCURO = "#0f1419"   # mesmo fundo do tema escuro (.streamlit)
TEXTO_ESCURO = "#e6eaee"

# Cores de componentes do union-find (ciclo quando acabam).
CORES_COMPONENTES = (
    "#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#edc948",
    "#76b7b2", "#ff9da7", "#9c755f", "#e15759", "#bab0ac",
)

ALTURA_PADRAO = 560


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


class MontadorVisual:
    """Converte os grafos do projeto em HTML interativo.

    Cada método ``*_g2``, ``*_g3`` e
    ``passo_kruskal`` devolve ``(vertices, arestas, opcoes)``, que
    ``html`` transforma na página. Os vértices e arestas seguem o
    formato do vis-network (``id``, ``label``, ``color``, ``from``,
    ``to``...).
    """

    def __init__(self, altura: int = ALTURA_PADRAO,
                 escuro: bool = False) -> None:
        self.altura = altura
        self.escuro = escuro
        self.fundo = FUNDO_ESCURO if escuro else "#ffffff"
        self.texto = TEXTO_ESCURO if escuro else COR_TEXTO
        self.moldura = "#2a333c" if escuro else "#e3e7eb"
        self.cor_futuro = "#252d35" if escuro else COR_FUTURO

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------
    def html(self, vertices: list[dict], arestas: list[dict],
             opcoes: dict | None = None) -> str:
        """Página com o grafo desenhado pelo vis-network.

        Fundo, texto dos rótulos e contorno dos rótulos de aresta seguem
        o modo (claro ou escuro) escolhido no ``__init__``.
        """
        opcoes = dict(opcoes or {})
        opcoes.setdefault("nodes", {}).setdefault("font", {})
        opcoes["nodes"]["font"]["color"] = self.texto
        opcoes.setdefault("edges", {})["font"] = {
            "color": self.texto, "strokeColor": self.fundo}
        dados = {
            "vertices": vertices,
            "arestas": arestas,
            "opcoes": opcoes,
        }
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<style>*{box-sizing:border-box}html,body{margin:0;"
            f"padding:0;background:{self.fundo};overflow:hidden;"
            "font-family:sans-serif}#rede{width:100%;"
            f"height:{self.altura}px;border:1px solid {self.moldura};"
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
                    "id": v, "label": _rotulo_curto(ROTULOS.get(v, v)),
                    "shape": "box" if destaque else "ellipse",
                    "color": {"background": COR_DESTAQUE if destaque
                              else "#eef1f4",
                              "border": "#b26b1f" if destaque
                              else "#9aa5b1"},
                    "font": {"size": 15 if destaque else 12,
                             "color": COR_TEXTO},
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
                **({"label": f"{numero(p, 3)} bits",
                    "font": {"size": 15, "color": COR_EVASAO,
                             "strokeWidth": 4, "align": "horizontal",
                             "bold": True}}
                   if toca else {}),
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
            rotulo = _rotulo_curto(ROTULOS.get(v, v), 18)
            nos.append({"id": v, "label": rotulo,
                        "x": raio_x * math.cos(ang),
                        "y": raio_y * math.sin(ang), "shape": "dot",
                        "size": 9, "color": {"background": cor,
                                             "border": "#6b7785"},
                        "font": {"size": 11}, "title": v})
        cor_d = cores.get(componentes[desfecho], "#eef1f4")
        nos.append({"id": desfecho, "label": "Desfecho", "x": 0, "y": 0,
                    "shape": "box",
                    "font": {"size": 16, "color": COR_TEXTO},
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
               caminho_usuario: list[str] | None = None,
               conhecidas: int | None = None) -> tuple:
        """DAG em camadas fixas, da esquerda (INICIO) para a direita.

        A camada vem do prefixo do estado (``L0:``...). Dentro de cada
        camada os estados ficam em ordem de taxa de evasão (baixo =
        mais risco). O número antes do nome é a posição na ordem
        topológica de Kahn. Cor = taxa de evasão do estado no treino.

        ``conhecidas`` (camadas L0... já observadas no momento, como em
        ``dag_trajetorias.CAMADAS_POR_MOMENTO``) esmaece as camadas que
        o preditor ainda não enxerga; o desfecho continua visível.
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

        ultima = max(camadas)

        def futuro(v: str) -> bool:
            c = self._camada(v)
            return conhecidas is not None and conhecidas < c < ultima

        maior = max(contagem.values())
        nos = []
        for v in grafo.vertices():
            x, y = pos[v]
            t = taxa(v)
            nome = v.split(":", 1)[-1]
            if futuro(v):
                nos.append({
                    "id": v, "x": x, "y": y, "shape": "dot",
                    "label": _rotulo_curto(nome, 18),
                    "size": 8 + 22 * math.sqrt(contagem[v] / maior),
                    "color": {"background": self.cor_futuro,
                              "border": self.moldura},
                    "borderWidth": 1,
                    "font": {"size": 12, "vadjust": -4,
                             "color": "#aab4be"},
                    "title": f"{v}\nainda não observado neste momento",
                })
                continue
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
            if futuro(u) or futuro(v):
                arestas.append({
                    "from": u, "to": v,
                    "width": 3 if cor else 0.6,
                    "color": {"color": cor or self.cor_futuro,
                              "opacity": 0.3 if cor else 0.5},
                    "dashes": e_virada,
                    "title": f"{u} → {v}\n{int(n)} alunos (ainda não "
                             "observado neste momento)",
                })
                continue
            arestas.append({
                "from": u, "to": v,
                **({"label": f"{int(n)}",
                    "font": {"size": 13, "color": cor,
                             "strokeWidth": 4, "align": "horizontal"}}
                   if cor else {}),
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


class GrafoECharts:
    """Opções do ECharts (``st.echarts_chart``) para desenhos clicáveis.

    Diferente do vis-network, o ECharts devolve o clique ao Python: a
    seleção traz ``data_type`` (``"node"`` ou ``"edge"``) e o índice
    em ``data`` ou ``links``. Por isso cada método devolve, junto com
    a opção, os vértices e as arestas na mesma ordem do desenho.
    """

    def __init__(self, escuro: bool = False) -> None:
        self.contorno = TEXTO_ESCURO if escuro else COR_TEXTO

    CATEGORIAS_G1 = (
        ("treino · evadiu", COR_EVASAO),
        ("treino · formou", COR_FORMATURA),
        ("teste · previsto evasão", _mistura(COR_EVASAO, "#ffffff", 0.45)),
        ("teste · previsto formatura",
         _mistura(COR_FORMATURA, "#ffffff", 0.45)),
    )

    def ego_g1(self, grafo: Grafo, aluno: int,
               rotulos_treino: Mapping[int, bool],
               previsoes: Mapping[int, bool],
               lado_evasao: set[int],
               evadiu_real: Mapping[int, bool]
               ) -> tuple[dict, list[int], list[tuple[int, int]]]:
        """Aluno escolhido, seus vizinhos no kNN e as arestas cortadas.

        Args:
            grafo: kNN de Gower (capacidade = 1 − distância).
            aluno: vértice central (aluno de teste).
            rotulos_treino: rótulos que o algoritmo conhecia.
            previsoes: previsões do corte para os alunos sem rótulo.
            lado_evasao: vértices do lado de S no corte mínimo.
            evadiu_real: desfecho real dos alunos sem rótulo (borda),
                mostrado só porque o teste já foi avaliado.

        Returns:
            ``(opcao, vertices, arestas)``: ``vertices[i]`` é o aluno
            do índice ``i`` em ``data``; ``arestas[i]``, o par ``(u, v)``
            do índice ``i`` em ``links``.
        """
        ego = [aluno, *sorted(grafo.vizinhos(aluno))]
        posicao = {v: i for i, v in enumerate(ego)}
        nos = [self._no_aluno(v, v == aluno, rotulos_treino, previsoes,
                              evadiu_real) for v in ego]
        links, pares = [], []
        for u in ego:
            for v, c in grafo.vizinhos(u).items():
                if v not in posicao or posicao[v] <= posicao[u]:
                    continue
                cortada = (u in lado_evasao) != (v in lado_evasao)
                central = aluno in (u, v)
                cor = COR_EVASAO if cortada else (
                    "#7b8794" if central else COR_ARESTA)
                texto = f"aluno {u} — aluno {v}<br/>similaridade " \
                        f"1 − d = {numero(c, 3)}"
                if cortada:
                    texto += "<br/><b>aresta cortada</b> (lados opostos)"
                links.append({
                    "source": str(u), "target": str(v),
                    "value": round(c, 4),
                    "lineStyle": {
                        "width": 1 + 6 * c if central else 0.6 + 1.5 * c,
                        "color": cor,
                        "type": "dashed" if cortada else "solid",
                        "opacity": 0.9 if central or cortada else 0.3},
                    "tooltip": {"formatter": texto},
                })
                pares.append((u, v))
        serie = {
            "type": "graph", "layout": "force", "roam": True,
            "draggable": True, "selectedMode": "single",
            "categories": [{"name": n, "itemStyle": {"color": c}}
                           for n, c in self.CATEGORIAS_G1],
            "force": {"repulsion": 340, "gravity": 0.12,
                      "edgeLength": [55, 190], "friction": 0.2},
            "label": {"show": True, "position": "right", "fontSize": 10},
            "emphasis": {"focus": "adjacency",
                         "lineStyle": {"width": 5, "opacity": 1}},
            "select": {"itemStyle": {"borderColor": COR_DESTAQUE,
                                     "borderWidth": 5},
                       "lineStyle": {"color": COR_DESTAQUE, "width": 7,
                                     "opacity": 1}},
            "data": nos, "links": links,
        }
        opcao = {
            "tooltip": {"show": True, "confine": True},
            "legend": {"data": [n for n, _ in self.CATEGORIAS_G1],
                       "bottom": 0, "itemGap": 14},
            "series": [serie],
        }
        return opcao, ego, pares

    def _no_aluno(self, v: int, central: bool,
                  rotulos_treino: Mapping[int, bool],
                  previsoes: Mapping[int, bool],
                  evadiu_real: Mapping[int, bool]) -> dict:
        """Vértice de aluno: categoria pelo rótulo ou pela previsão."""
        if v in rotulos_treino:
            categoria = 0 if rotulos_treino[v] else 1
            papel = "treino: " + ("evadiu" if rotulos_treino[v]
                                  else "formou")
            borda, largura, simbolo = "#ffffff", 1, "circle"
        else:
            prev = previsoes.get(v, False)
            real = evadiu_real.get(v)
            categoria = 2 if prev else 3
            papel = "teste: previsto " + ("evasão" if prev
                                          else "formatura")
            if real is not None:
                papel += "<br/>real: " + ("evasão" if real
                                          else "formatura")
            borda = (COR_EVASAO if real else COR_FORMATURA) \
                if real is not None else self.contorno
            largura, simbolo = 3, "diamond"
        return {
            "id": str(v),
            "name": f"aluno {v}" if central else str(v),
            "category": categoria,
            "symbol": simbolo,
            "symbolSize": 42 if central else 16,
            "itemStyle": {"borderColor": self.contorno if central
                          else borda,
                          "borderWidth": 4 if central else largura},
            "label": {"fontSize": 14 if central else 10,
                      "fontWeight": "bold" if central else "normal"},
            "tooltip": {"formatter": f"<b>aluno {v}</b><br/>{papel}"},
        }
