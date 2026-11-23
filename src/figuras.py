"""Figuras do relatório em PDF vetorial. Classe: ``GeradorFiguras``.

- Fig. 3 (``fig-lift.pdf``): lift da evasão no treino (barra) e no
  teste (ponto), por fator (H2).
- Fig. 4 (``fig-ablacao.pdf``): Δ da acurácia balanceada do corte
  mínimo ao retirar cada grupo, com IC 95%, em CV no treino (H3).
- Fig. 5 (``fig-arvore.pdf``): árvore geradora máxima de informação
  mútua (Kruskal) em T1, enraizada no Desfecho (G2).

Os números vêm de ``outputs/`` (gerados por ``python src/main.py``); o
lift vem da ``AnaliseFatores``, gravada aqui em ``outputs/fatores_*``.
O desenho da árvore usa um layout próprio (sem networkx).

Uso::

    python src/figuras.py    # grava em relatorio/figs/
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA_PADRAO = RAIZ / "outputs"
DESTINO_PADRAO = RAIZ / "relatorio" / "figs"

LARGURA_COLUNA = 3.4      # polegadas (coluna do template imc, 2 colunas)
LARGURA_PAGINA = 7.0
ESPACO_RAMOS = 1.0        # folga entre os ramos do Desfecho na árvore

COR_AUMENTA = "#c0392b"
COR_DIMINUI = "#2e6da4"
COR_TEXTO = "#222222"
COR_GRADE = "#d9d9d9"
CORES_MOMENTO = {"T0": "#7f8c8d", "T1": "#2e6da4", "T2": "#c0392b"}
CORES_GRUPO = {
    "desfecho": "#222222", "academico_s1": "#2e6da4",
    "academico_s2": "#174a7c", "financeiro": "#c0392b",
    "ingresso": "#e67e22", "demografico": "#7f8c8d",
    "familia": "#8e6bb8", "macro": "#27ae60",
}
NOMES_GRUPO = {
    "academico_s1": "acadêmico S1", "academico_s2": "acadêmico S2",
    "financeiro": "financeiro", "ingresso": "ingresso",
    "demografico": "demográfico", "familia": "família", "macro": "macro",
}
ROTULOS = {
    "Desfecho": "Desfecho", "Marital status": "Estado civil",
    "Gender": "Gênero", "Age at enrollment": "Idade",
    "Nacionality": "Nacionalidade", "International": "Internacional",
    "Displaced": "Deslocado", "Educational special needs": "Nec. especiais",
    "Mother's qualification": "Escol. mãe",
    "Father's qualification": "Escol. pai",
    "Mother's occupation": "Ocup. mãe", "Father's occupation": "Ocup. pai",
    "Application mode": "Modo de ingresso",
    "Application order": "Ordem de opção", "Course": "Curso",
    "Daytime/evening attendance": "Turno",
    "Previous qualification": "Formação prévia",
    "Previous qualification (grade)": "Nota form. prévia",
    "Admission grade": "Nota admissão", "Unemployment rate": "Desemprego",
    "Inflation rate": "Inflação", "GDP": "PIB", "Debtor": "Devedor",
    "Tuition fees up to date": "Mensalidade em dia",
    "Scholarship holder": "Bolsista",
    "taxa_aprov_s1": "Taxa aprov. S1", "taxa_aprov_s2": "Taxa aprov. S2",
}
for _sem, _s in (("1st", "S1"), ("2nd", "S2")):
    for _campo, _nome in (("credited", "creditadas"),
                          ("enrolled", "inscritas"),
                          ("evaluations", "avaliações"),
                          ("approved", "aprovadas"), ("grade", "média"),
                          ("without evaluations", "sem avaliação")):
        ROTULOS[f"Curricular units {_sem} sem ({_campo})"] = \
            f"{_s} {_nome}"

ESTILO = {
    "font.family": "serif", "font.size": 7, "axes.titlesize": 7,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": "#555555",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "pdf.fonttype": 42,
}


def _grupo_de(atributo: str) -> str:
    """Grupo de um atributo (inclui as taxas derivadas e o Desfecho)."""
    from dados import DERIVADOS, GRUPOS
    if atributo == "Desfecho":
        return "desfecho"
    if atributo in DERIVADOS:
        return DERIVADOS[atributo]
    for grupo, colunas in GRUPOS.items():
        if atributo in colunas:
            return grupo
    return "macro"


class GeradorFiguras:
    """Gera as Figs. 3–5 do relatório a partir de ``outputs/``.

    Parâmetros:
        entrada: pasta com os resultados do pipeline.
        destino: pasta das figuras (``relatorio/figs``).
    """

    def __init__(self, entrada: Path | str = ENTRADA_PADRAO,
                 destino: Path | str = DESTINO_PADRAO) -> None:
        self.entrada = Path(entrada)
        self.destino = Path(destino)

    def gerar_todas(self) -> list[Path]:
        """Gera as três figuras e devolve os caminhos gravados."""
        self.destino.mkdir(parents=True, exist_ok=True)
        return [
            self.lift(pd.read_csv(self.entrada / "fatores_lift.csv")),
            self.ablacao(pd.read_csv(self.entrada /
                                     "avaliacao_ablacao.csv")),
            self.arvore(pd.read_csv(self.entrada / "g2_arvore_T1.csv")),
        ]

    # ------------------------------------------------------------------
    # Fig. 3: lift treino × teste
    # ------------------------------------------------------------------
    def lift(self, tabela: pd.DataFrame) -> Path:
        """Barras divergentes em escala log; ponto = mesmo grupo no teste."""
        t = tabela.sort_values("lift_treino").reset_index(drop=True)
        y = np.arange(len(t))
        with plt.rc_context(ESTILO):
            fig, ax = plt.subplots(figsize=(LARGURA_COLUNA, 4.3))
            cores = np.where(t["lift_treino"] >= 1, COR_AUMENTA,
                             COR_DIMINUI)
            esquerda = np.minimum(t["lift_treino"], 1)
            largura = np.abs(t["lift_treino"] - 1)
            ax.barh(y, largura, left=esquerda, height=0.62, color=cores,
                    alpha=0.85, linewidth=0)
            ax.scatter(t["lift_teste"], y, s=9, color=COR_TEXTO,
                       zorder=3, linewidths=0)
            ax.axvline(1, color=COR_TEXTO, linewidth=0.6)
            ax.set_xscale("log")
            marcas = [0.1, 0.2, 0.5, 1, 2, 3]
            ax.set_xticks(marcas)
            ax.set_xticklabels([f"×{m:g}".replace(".", ",")
                                for m in marcas])
            ax.minorticks_off()
            ax.set_xlim(0.07, 3.3)
            ax.set_yticks(y)
            rotulos = [f"{f}{'' if q < 0.05 else ' (n.s.)'}"
                       for f, q in zip(t["fator"], t["q_treino"])]
            ax.set_yticklabels(rotulos)
            ax.set_ylim(-0.7, len(t) - 0.3)
            ax.grid(axis="x", color=COR_GRADE, linewidth=0.4)
            ax.set_axisbelow(True)
            ax.set_xlabel("Lift da evasão (taxa do grupo ÷ 39,15%), "
                          "escala log")
            for lado in ("top", "right"):
                ax.spines[lado].set_visible(False)
            legenda = [
                Patch(color=COR_AUMENTA, label="aumenta (treino)"),
                Patch(color=COR_DIMINUI, label="diminui (treino)"),
                Line2D([], [], marker="o", linestyle="", markersize=3,
                       color=COR_TEXTO, label="teste (20%)")]
            ax.legend(handles=legenda, loc="upper left", frameon=False,
                      handlelength=1.2)
            return self._salvar(fig, "fig-lift.pdf")

    # ------------------------------------------------------------------
    # Fig. 4: ablação por grupo
    # ------------------------------------------------------------------
    def ablacao(self, tabela: pd.DataFrame) -> Path:
        """Δ por grupo e momento com IC 95%; cheio = essencial."""
        grupos = ["academico_s2", "academico_s1", "financeiro",
                  "ingresso", "demografico", "familia", "macro"]
        momentos = ["T0", "T1", "T2"]
        deslocamento = {"T0": 0.22, "T1": 0.0, "T2": -0.22}
        with plt.rc_context(ESTILO):
            fig, ax = plt.subplots(figsize=(LARGURA_COLUNA, 2.5))
            for i, grupo in enumerate(grupos):
                if i % 2 == 0:
                    ax.axhspan(len(grupos) - i - 1.5, len(grupos) - i - 0.5,
                               color="#f4f4f4", linewidth=0, zorder=0)
            for m in momentos:
                linhas = tabela[tabela["momento"] == m]
                for i, grupo in enumerate(grupos):
                    r = linhas[linhas["grupo"] == grupo]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    yv = len(grupos) - i - 1 + deslocamento[m]
                    cor = CORES_MOMENTO[m]
                    ax.plot([r["ic_inf"] * 100, r["ic_sup"] * 100],
                            [yv, yv], color=cor, linewidth=1)
                    essencial = r["classe"] == "essencial"
                    ax.plot(r["delta"] * 100, yv, marker="o",
                            markersize=3.6, color=cor,
                            markerfacecolor=cor if essencial else "white",
                            markeredgewidth=0.9, zorder=3)
            ax.axvline(0, color=COR_TEXTO, linewidth=0.6)
            ax.set_yticks(range(len(grupos)))
            ax.set_yticklabels([NOMES_GRUPO[g] for g in reversed(grupos)])
            ax.set_ylim(-0.5, len(grupos) - 0.5)
            ax.set_xlabel("Δ acurácia balanceada ao retirar o grupo "
                          "(p.p., CV no treino)")
            ax.grid(axis="x", color=COR_GRADE, linewidth=0.4)
            ax.set_axisbelow(True)
            for lado in ("top", "right"):
                ax.spines[lado].set_visible(False)
            legenda = [Line2D([], [], marker="o", color=CORES_MOMENTO[m],
                              markersize=3.6, linewidth=1, label=m)
                       for m in momentos]
            legenda += [
                Line2D([], [], marker="o", color=COR_TEXTO, linestyle="",
                       markersize=3.6, label="essencial"),
                Line2D([], [], marker="o", color=COR_TEXTO, linestyle="",
                       markerfacecolor="white", markersize=3.6,
                       label="neutro (IC cruza 0)")]
            ax.xaxis.set_major_formatter(
                lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
            ax.legend(handles=legenda, loc="lower center",
                      bbox_to_anchor=(0.45, 1.0), frameon=False, ncol=5,
                      handlelength=1.4, columnspacing=0.9)
            return self._salvar(fig, "fig-ablacao.pdf")

    # ------------------------------------------------------------------
    # Fig. 5: árvore de Kruskal
    # ------------------------------------------------------------------
    def arvore(self, tabela: pd.DataFrame, raiz: str = "Desfecho") -> Path:
        """Árvore enraizada no Desfecho; espessura ∝ IM.

        Aresta tracejada: presente em menos de 50% dos bootstraps.
        """
        filhos, im, estab = self._enraizar(tabela, raiz)
        pos = self._layout(filhos, im, raiz)
        with plt.rc_context(ESTILO):
            fig, ax = plt.subplots(figsize=(LARGURA_PAGINA, 3.9))
            maximo = max(im.values())
            for (u, v), peso in im.items():
                (x0, y0), (x1, y1) = pos[u], pos[v]
                ax.plot([x0, x1], [y0, y1], color="#666666",
                        linewidth=0.5 + 3.5 * np.sqrt(peso / maximo),
                        linestyle="-" if estab[(u, v)] >= 0.5 else "--",
                        zorder=1, solid_capstyle="round")
            for no, (x, y) in pos.items():
                cor = CORES_GRUPO[_grupo_de(no)]
                rotulo = ROTULOS.get(no, no)
                if (raiz, no) in im:
                    rotulo += f"\n{im[(raiz, no)]:.3f} bits".replace(
                        ".", ",")
                ax.text(x, y, rotulo, ha="center",
                        va="center", fontsize=5.6,
                        color="white" if no == raiz else COR_TEXTO,
                        zorder=3, bbox={
                            "boxstyle": "round,pad=0.25",
                            "facecolor": cor if no == raiz else "white",
                            "edgecolor": cor, "linewidth": 0.9})
            ax.set_axis_off()
            ax.margins(x=0.04, y=0.08)
            usados = {_grupo_de(n) for n in pos} - {"desfecho"}
            ax.legend(handles=[
                Patch(facecolor="white", edgecolor=CORES_GRUPO[g],
                      label=NOMES_GRUPO[g])
                for g in NOMES_GRUPO if g in usados] + [
                Line2D([], [], color="#666666", linestyle="--",
                       label="estabilidade < 50%")],
                loc="lower right", frameon=False, ncol=2,
                handlelength=1.4)
            return self._salvar(fig, "fig-arvore.pdf")

    @staticmethod
    def _enraizar(tabela: pd.DataFrame, raiz: str):
        """Filhos de cada vértice (BFS a partir da raiz), IM e estabilidade."""
        adj = defaultdict(list)
        im, estab = {}, {}
        for u, v, w, e in zip(tabela["u"], tabela["v"], tabela["im_bits"],
                              tabela["estabilidade"]):
            adj[u].append(v)
            adj[v].append(u)
            im[(u, v)] = im[(v, u)] = w
            estab[(u, v)] = estab[(v, u)] = e
        filhos = defaultdict(list)
        visto, fila = {raiz}, [raiz]
        while fila:
            u = fila.pop(0)
            for v in sorted(adj[u], key=lambda v: -im[(u, v)]):
                if v not in visto:
                    visto.add(v)
                    filhos[u].append(v)
                    fila.append(v)
        arestas = {(u, v): im[(u, v)] for u in filhos for v in filhos[u]}
        estab_arestas = {k: estab[k] for k in arestas}
        return filhos, arestas, estab_arestas

    @staticmethod
    def _layout(filhos: dict, im: dict, raiz: str) -> dict:
        """Layout em camadas: x = profundidade, y = ordem das folhas."""
        pos, proxima = {}, [0.0]

        def visitar(u: str, prof: int) -> float:
            if not filhos[u]:
                x = proxima[0]
                proxima[0] += 1
            else:
                xs = []
                for v in filhos[u]:
                    xs.append(visitar(v, prof + 1))
                    if u == raiz:
                        proxima[0] += ESPACO_RAMOS
                x = (min(xs) + max(xs)) / 2
            pos[u] = (prof, -x)
            return x

        visitar(raiz, 0)
        return pos

    def _salvar(self, fig, nome: str) -> Path:
        """Grava a figura em PDF vetorial e fecha."""
        caminho = self.destino / nome
        fig.savefig(caminho, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return caminho


def main() -> None:
    """Calcula o lift (H2), grava ``outputs/fatores_*`` e as figuras."""
    from dados import BaseDeDados
    from fatores import AnaliseFatores

    r = AnaliseFatores().executar(BaseDeDados())
    for nome in ("lift", "faixas", "im"):
        r[nome].to_csv(ENTRADA_PADRAO / f"fatores_{nome}.csv", index=False)
    with open(ENTRADA_PADRAO / "fatores_replicacao.json", "w",
              encoding="utf-8") as arquivo:
        json.dump(r["replicacao"], arquivo, ensure_ascii=False, indent=2)
    for caminho in GeradorFiguras().gerar_todas():
        print(caminho.relative_to(RAIZ))


if __name__ == "__main__":
    main()
