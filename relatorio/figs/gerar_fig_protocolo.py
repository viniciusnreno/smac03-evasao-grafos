"""Fig. 2 do relatório: diagrama operacional segundo o CRISP-NET.

Responsável: Carlos (seção 3.2). Rótulo LaTeX: ``fig:protocolo``.

Segue o estilo do diagrama operacional dos trabalhos de referência da
disciplina: grade 3 x 3 de etapas numeradas, caixas azuis sólidas com
texto branco e setas finas em zigue-zague. Ocupa uma coluna do artigo.

Os números das caixas vêm de ``outputs/`` (gerados por ``src/main.py``)
e do CSV original, nunca digitados à mão.

Uso::

    python src/main.py                      # gera outputs/
    python relatorio/figs/gerar_fig_protocolo.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

matplotlib.use("Agg")

RAIZ = Path(__file__).resolve().parents[2]
OUTPUTS = RAIZ / "outputs"
CSV_DADOS = RAIZ / "data" / "data.csv"
SAIDA = Path(__file__).resolve().parent / "fig-protocolo.pdf"

# Largura de uma coluna do template (A4, margens de 15 mm, 13 pt entre
# colunas) em polegadas.
LARGURA_POL = 3.29
FONTES = ("Carlito", "Lato", "DejaVu Sans")
TAMANHO = 6.1
AZUL = "#156082"

COLUNAS, LINHAS = 3, 3
LARGURA_CAIXA, ALTURA_CAIXA = 0.98, 0.50      # polegadas
FOLGA_X, FOLGA_Y = 0.175, 0.20                 # polegadas


def br(numero: float) -> str:
    """Inteiro no padrão brasileiro: 4.424."""
    return f"{numero:,.0f}".replace(",", ".")


class FiguraProtocolo:
    """Diagrama operacional do trabalho, a partir de ``outputs/``."""

    def __init__(self, outputs: Path = OUTPUTS,
                 csv_dados: Path = CSV_DADOS) -> None:
        with open(outputs / "resultados.json", encoding="utf-8") as arq:
            self.resultados = json.load(arq)
        self.total_csv = len(pd.read_csv(csv_dados, sep=";"))
        self._ax = None

    def etapas(self) -> list[str]:
        """Texto das nove etapas, na ordem de leitura."""
        d = self.resultados["dados"]
        h = self.resultados["hiperparametros"]
        divisao = h["divisao"]
        return [
            "Estudo do problema:\nevasão e hipóteses",
            f"Entendimento dos dados\n(UCI 697, "
            f"{br(self.total_csv)} alunos)",
            f"Preparação: remove\n{br(self.total_csv - d['alunos'])} "
            "matriculados;\nmomentos T0–T2",
            f"Divisão "
            f"{round(divisao['treino'] * 100)}/"
            f"{round(divisao['teste'] * 100)}\nestratificada\n"
            f"({br(d['treino'])} treino, {br(d['teste'])} teste)",
            f"Pesos, faixas e limiares\ne CV de "
            f"{h['dobras_cv_treino']} dobras\n(só no treino)",
            "Modelagem em rede:\nG1, G2 e G3",
            "Algoritmos de aula:\nEdmonds–Karp, Kruskal,\n"
            "Kahn e Bellman-Ford",
            "Avaliação no teste,\numa única vez",
            "Interpretação e\nconsolidação dos\nresultados",
        ]

    def desenhar(self):
        """Devolve a figura do matplotlib."""
        self._configurar_fonte()
        largura = COLUNAS * LARGURA_CAIXA + (COLUNAS - 1) * FOLGA_X
        altura = LINHAS * ALTURA_CAIXA + (LINHAS - 1) * FOLGA_Y
        margem = 0.02
        fig = plt.figure(figsize=(LARGURA_POL, altura + 2 * margem))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(-(LARGURA_POL - largura) / 2,
                    largura + (LARGURA_POL - largura) / 2)
        ax.set_ylim(-margem, altura + margem)
        ax.axis("off")
        self._ax = ax

        posicoes = []
        for i, texto in enumerate(self.etapas()):
            linha, coluna = divmod(i, COLUNAS)
            x = coluna * (LARGURA_CAIXA + FOLGA_X)
            y = altura - ALTURA_CAIXA - linha * (ALTURA_CAIXA + FOLGA_Y)
            posicoes.append((x, y))
            self._caixa(x, y, f"[{i + 1}]\n{texto}")
        self._setas(posicoes)
        return fig

    def salvar(self, caminho: Path = SAIDA) -> Path:
        """Grava o PDF vetorial e devolve o caminho."""
        fig = self.desenhar()
        fig.savefig(caminho, format="pdf")
        plt.close(fig)
        return caminho

    # ------------------------------------------------------------------
    # Desenho
    # ------------------------------------------------------------------
    def _caixa(self, x: float, y: float, texto: str) -> None:
        self._ax.add_patch(Rectangle((x, y), LARGURA_CAIXA, ALTURA_CAIXA,
                                     facecolor=AZUL, edgecolor="none"))
        self._ax.text(x + LARGURA_CAIXA / 2, y + ALTURA_CAIXA / 2, texto,
                      ha="center", va="center", color="white",
                      fontsize=TAMANHO, linespacing=1.1)

    def _setas(self, posicoes: list[tuple[float, float]]) -> None:
        estilo = {"arrowstyle": "-|>", "mutation_scale": 5,
                  "color": AZUL, "linewidth": 0.5, "shrinkA": 0,
                  "shrinkB": 0}
        meio_y = ALTURA_CAIXA / 2
        for i in range(len(posicoes) - 1):
            (x0, y0), (x1, y1) = posicoes[i], posicoes[i + 1]
            if (i + 1) % COLUNAS:
                # Mesma linha: da borda direita à borda esquerda.
                self._ax.add_patch(FancyArrowPatch(
                    (x0 + LARGURA_CAIXA + 0.02, y0 + meio_y),
                    (x1 - 0.02, y1 + meio_y), **estilo))
            else:
                # Fim da linha: desce, volta à esquerda e desce.
                xa = x0 + LARGURA_CAIXA / 2
                xb = x1 + LARGURA_CAIXA * 0.12
                ym = y0 - FOLGA_Y / 2
                self._ax.plot([xa, xa, xb], [y0 - 0.02, ym, ym],
                              color=AZUL, linewidth=0.5,
                              solid_joinstyle="miter")
                self._ax.add_patch(FancyArrowPatch(
                    (xb, ym), (xb, y1 + ALTURA_CAIXA + 0.02), **estilo))

    @staticmethod
    def _configurar_fonte() -> None:
        nomes = {f.name for f in font_manager.fontManager.ttflist}
        fonte = next((f for f in FONTES if f in nomes), "sans-serif")
        plt.rcParams.update({"font.family": fonte, "font.size": TAMANHO,
                             "pdf.fonttype": 42})


def main() -> None:
    caminho = FiguraProtocolo().salvar()
    print(f"Figura gravada em {caminho}")


if __name__ == "__main__":
    main()
