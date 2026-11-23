"""Explicação das arestas do G1 para a interface.

Decompõe a similaridade de Gower entre dois alunos atributo por
atributo, com os mesmos pesos e amplitudes que o ``GrafoSimilaridade``
aprendeu no treino. Para cada atributo ``a`` com peso ``w_a``:

    d(i, j) = soma_a w_a · δ_a(i, j)        (os pesos somam 1)
    1 − d   = soma_a w_a · (1 − δ_a(i, j))  (capacidade da aresta)

δ_a é a diferença normalizada: ``|x_i − x_j| / amplitude`` (no máximo
1) para numéricos, 0/1 para nominais; ausente contra ausente vale 0 e
ausente contra presente vale 1. Assim a barra de cada atributo mostra
quanto do peso dele os dois alunos "ganharam" por serem parecidos.

Não depende do Streamlit: devolve tabelas e números.
"""

import pandas as pd

from dados import DERIVADOS, GRUPOS, NOMINAIS
from figuras import NOMES_GRUPO, ROTULOS
from grafo_similaridade import GrafoSimilaridade

# Dicionário dos cursos (plano v2, seção "Dados e curadoria").
CURSOS: dict[int, str] = {
    33: "Tec. Produção de Biocombustíveis",
    171: "Design de Animação e Multimédia",
    8014: "Serviço Social (noturno)",
    9003: "Agronomia",
    9070: "Design de Comunicação",
    9085: "Enfermagem Veterinária",
    9119: "Engenharia Informática",
    9130: "Equinicultura",
    9147: "Gestão",
    9238: "Serviço Social",
    9254: "Turismo",
    9500: "Enfermagem",
    9556: "Higiene Oral",
    9670: "Publicidade e Marketing",
    9773: "Jornalismo e Comunicação",
    9853: "Educação Básica",
    9991: "Gestão (noturno)",
}
BINARIOS: frozenset[str] = frozenset({
    "Gender", "International", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Scholarship holder",
    "Daytime/evening attendance",
})
TOLERANCIA = 1e-9


def grupo_do_atributo(atributo: str) -> str:
    """Grupo (``dados.GRUPOS``) a que o atributo pertence."""
    if atributo in DERIVADOS:
        return DERIVADOS[atributo]
    return next(g for g, colunas in GRUPOS.items() if atributo in colunas)


def formatar_valor(atributo: str, valor) -> str:
    """Valor legível de um atributo (curso pelo nome, taxa em %...)."""
    if pd.isna(valor):
        return "—"
    if atributo == "Course":
        return CURSOS.get(int(valor), f"código {int(valor)}")
    if atributo == "Gender":
        return "masculino" if int(valor) == 1 else "feminino"
    if atributo == "Daytime/evening attendance":
        return "diurno" if int(valor) == 1 else "noturno"
    if atributo in BINARIOS:
        return "sim" if int(valor) == 1 else "não"
    if atributo in DERIVADOS:
        return f"{100 * float(valor):.0f}%"
    if atributo in NOMINAIS:
        return f"código {int(valor)}"
    if float(valor).is_integer():
        return str(int(valor))
    return f"{float(valor):.1f}".replace(".", ",")


class ExplicadorGower:
    """Decompõe a similaridade de Gower de um ``GrafoSimilaridade``.

    Args:
        modelo: ``GrafoSimilaridade`` já construído (usa ``pesos_`` e
            ``amplitudes_``, aprendidos só no treino).
        base: ``BaseDeDados`` de onde saem os valores dos alunos.
    """

    def __init__(self, modelo: GrafoSimilaridade, base) -> None:
        if not modelo.pesos_:
            raise ValueError("construa o modelo antes de explicá-lo")
        self.modelo = modelo
        self.pesos = dict(modelo.pesos_)
        self.amplitudes = dict(modelo.amplitudes_)
        self.colunas = list(self.pesos)
        self.dados = base.dados[self.colunas]

    def contribuicoes(self, i: int, j: int) -> pd.DataFrame:
        """Tabela atributo a atributo da similaridade entre ``i`` e ``j``.

        Colunas: ``atributo``, ``rotulo``, ``grupo``, ``peso``,
        ``diferenca`` (δ), ``semelhanca`` (w·(1 − δ)), ``distancia``
        (w·δ), ``valor_i`` e ``valor_j`` (legíveis). A soma de
        ``semelhanca`` é a capacidade 1 − d da aresta; confere com
        ``GrafoSimilaridade.distancia``.
        """
        xi, xj = self.dados.loc[i], self.dados.loc[j]
        linhas = []
        for c in self.colunas:
            delta = self._diferenca(c, xi[c], xj[c])
            w = self.pesos[c]
            linhas.append({
                "atributo": c,
                "rotulo": ROTULOS.get(c, c),
                "grupo": NOMES_GRUPO[grupo_do_atributo(c)],
                "peso": w,
                "diferenca": delta,
                "semelhanca": w * (1.0 - delta),
                "distancia": w * delta,
                "valor_i": formatar_valor(c, xi[c]),
                "valor_j": formatar_valor(c, xj[c]),
            })
        tabela = pd.DataFrame(linhas).sort_values("peso", ascending=False)
        d = float(tabela.distancia.sum())
        if abs(d - self.modelo.distancia(i, j)) > TOLERANCIA:
            raise AssertionError("decomposição difere da distância do G1")
        return tabela.reset_index(drop=True)

    def perfil(self, aluno: int, limite: int = 8) -> pd.DataFrame:
        """Os ``limite`` atributos de maior peso, com o valor do aluno."""
        principais = sorted(self.colunas, key=self.pesos.get,
                            reverse=True)[:limite]
        return pd.DataFrame([
            {"atributo": ROTULOS.get(c, c),
             "valor": formatar_valor(c, self.dados.at[aluno, c]),
             "peso": self.pesos[c]}
            for c in principais])

    def _diferenca(self, coluna: str, a, b) -> float:
        """δ de Gower de um atributo (regra do ``GrafoSimilaridade``)."""
        ausente_a, ausente_b = pd.isna(a), pd.isna(b)
        if ausente_a or ausente_b:
            return float(ausente_a != ausente_b)
        if coluna in NOMINAIS:
            return float(a != b)
        r = self.amplitudes[coluna]
        if r <= 0:
            return 0.0
        return min(abs(float(a) - float(b)) / r, 1.0)
