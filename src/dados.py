"""Carga e preparação dos dados. Classe: ``BaseDeDados``.

Responsável: Rafael.

- Lê data/data.csv (UCI 697) e remove os alunos "Enrolled".
- Separa treino/teste 80/20 estratificado, semente 42.
- Entrega os atributos de cada momento: T0, T1 e T2.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CAMINHO_PADRAO = Path(__file__).resolve().parent.parent / "data" / "data.csv"
SEMENTE = 42
FRACAO_TESTE = 0.2
ALVO = "Target"

GRUPOS: dict[str, list[str]] = {
    "demografico": [
        "Marital status", "Gender", "Age at enrollment", "Nacionality",
        "International", "Displaced", "Educational special needs",
    ],
    "familia": [
        "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation",
    ],
    "ingresso": [
        "Application mode", "Application order", "Course",
        "Daytime/evening attendance", "Previous qualification",
        "Previous qualification (grade)", "Admission grade",
    ],
    "macro": ["Unemployment rate", "Inflation rate", "GDP"],
    "financeiro": [
        "Debtor", "Tuition fees up to date", "Scholarship holder",
    ],
    "academico_s1": [
        f"Curricular units 1st sem ({c})"
        for c in ("credited", "enrolled", "evaluations", "approved",
                  "grade", "without evaluations")
    ],
    "academico_s2": [
        f"Curricular units 2nd sem ({c})"
        for c in ("credited", "enrolled", "evaluations", "approved",
                  "grade", "without evaluations")
    ],
}

# Grupos disponíveis em cada momento da trajetória.
MOMENTOS: dict[str, list[str]] = {
    "T0": ["demografico", "familia", "ingresso", "macro", "financeiro"],
    "T1": ["demografico", "familia", "ingresso", "macro", "financeiro",
           "academico_s1"],
    "T2": ["demografico", "familia", "ingresso", "macro", "financeiro",
           "academico_s1", "academico_s2"],
}

# Atributos derivados e o semestre de que dependem.
DERIVADOS: dict[str, str] = {
    "taxa_aprov_s1": "academico_s1",
    "taxa_aprov_s2": "academico_s2",
}

# Códigos sem ordem (comparados por igualdade na distância de Gower).
# Os demais atributos são numéricos ou binários 0/1.
NOMINAIS: frozenset[str] = frozenset({
    "Marital status", "Application mode", "Course", "Nacionality",
    "Previous qualification", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
})

# Faixas da taxa de aprovação de um semestre (camadas L2/L3 do G3 e
# discretização da informação mútua no G1).
FAIXAS_APROVACAO: tuple[str, ...] = (
    "sem inscricao", "0%", "(0,50%)", "[50,100%)", "100%")


def faixa_aprovacao(taxa: pd.Series) -> np.ndarray:
    """Faixa de cada taxa de aprovação; ``NaN`` vira "sem inscricao"."""
    return np.select([taxa.isna(), taxa == 0, taxa < 0.5, taxa < 1],
                     FAIXAS_APROVACAO[:4], FAIXAS_APROVACAO[4])


class BaseDeDados:
    """Alunos com desfecho conhecido, divididos em treino e teste.

    Ao ser criada, a base lê o CSV, remove os "Enrolled", numera os
    alunos de 0 a n - 1 (o número é o vértice do aluno no G1) e faz a
    divisão 80/20 estratificada pelo desfecho.

    Atributos:
        dados: DataFrame com os 36 atributos, os derivados
            ``taxa_aprov_s1`` e ``taxa_aprov_s2``, ``Target`` e
            ``evadiu`` (True = Dropout).
        idx_treino, idx_teste: posições dos alunos de cada conjunto.
    """

    def __init__(self, caminho: Path | str = CAMINHO_PADRAO,
                 semente: int = SEMENTE,
                 fracao_teste: float = FRACAO_TESTE) -> None:
        self.caminho = Path(caminho)
        self.semente = semente
        self.fracao_teste = fracao_teste
        self.dados = self._carregar()
        self.idx_treino, self.idx_teste = self.dividir(semente)

    def __len__(self) -> int:
        return len(self.dados)

    def dividir(self, semente: int) -> tuple[list[int], list[int]]:
        """Divisão estratificada treino/teste com a semente dada.

        A semente padrão (42) gera a divisão oficial; outras sementes
        servem para os testes de robustez (10 divisões).
        """
        idx_treino, idx_teste = train_test_split(
            np.arange(len(self.dados)), test_size=self.fracao_teste,
            stratify=self.dados["evadiu"], random_state=semente)
        return sorted(idx_treino.tolist()), sorted(idx_teste.tolist())

    @staticmethod
    def atributos(momento: str,
                  incluir_derivados: bool = False) -> list[str]:
        """Colunas disponíveis no momento ``T0``, ``T1`` ou ``T2``."""
        if momento not in MOMENTOS:
            raise ValueError(f"momento inválido: {momento!r} "
                             f"(use {', '.join(MOMENTOS)})")
        grupos = MOMENTOS[momento]
        colunas = [c for g in grupos for c in GRUPOS[g]]
        if incluir_derivados:
            colunas += [d for d, g in DERIVADOS.items() if g in grupos]
        return colunas

    def X(self, idx: list[int], momento: str,
          incluir_derivados: bool = False) -> pd.DataFrame:
        """Atributos do momento para os alunos ``idx``."""
        return self.dados.loc[idx, self.atributos(momento,
                                                  incluir_derivados)]

    def y(self, idx: list[int]) -> np.ndarray:
        """Desfecho dos alunos ``idx`` (True = evadiu)."""
        return self.dados.loc[idx, "evadiu"].to_numpy()

    def rotulos(self, idx: list[int]) -> dict[int, bool]:
        """``{aluno: evadiu}``, no formato do ``ClassificadorCorteMinimo``."""
        return {int(i): bool(v)
                for i, v in self.dados.loc[idx, "evadiu"].items()}

    def taxa_evasao(self, idx: list[int] | None = None) -> float:
        """Fração de evadidos em ``idx`` (todos os alunos, se omitido)."""
        serie = self.dados["evadiu"] if idx is None else \
            self.dados.loc[idx, "evadiu"]
        return float(serie.mean())

    def _carregar(self) -> pd.DataFrame:
        """Lê o CSV, remove os Enrolled e cria as colunas derivadas."""
        dados = pd.read_csv(self.caminho, sep=";")
        # A coluna "Daytime/evening attendance" vem com um tab no nome.
        dados.columns = dados.columns.str.strip()
        esperadas = {c for g in GRUPOS.values() for c in g} | {ALVO}
        faltando = esperadas - set(dados.columns)
        if faltando:
            raise ValueError(f"colunas ausentes no CSV: {sorted(faltando)}")

        dados = dados[dados[ALVO] != "Enrolled"].reset_index(drop=True)
        dados["evadiu"] = dados[ALVO] == "Dropout"
        for semestre, rotulo in (("1st", "s1"), ("2nd", "s2")):
            inscritas = dados[f"Curricular units {semestre} sem (enrolled)"]
            aprovadas = dados[f"Curricular units {semestre} sem (approved)"]
            # Sem inscrição a taxa fica indefinida (NaN): é a faixa
            # "sem inscrição" do plano, tratada por quem usa a base.
            dados[f"taxa_aprov_{rotulo}"] = (
                aprovadas / inscritas.where(inscritas > 0))
        return dados
