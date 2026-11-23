"""Análise de fatores (H1 e H2). Classe: ``AnaliseFatores``.

- Lift da evasão: taxa de evasão do grupo ÷ taxa geral. As faixas e os
  limiares saem só do treino; o mesmo grupo é medido no teste, sem nada
  reajustado (H2: a direção aprendida nos 80% vale nos 20%?).
- Significância no treino: teste exato de Fisher (grupo × resto) com
  correção de Benjamini-Hochberg sobre a família de faixas.
- Replicação: entre as faixas significativas no treino com pelo menos
  ``min_teste`` alunos no teste, a fração que mantém a direção (lift > 1
  ou < 1) no teste; e o Spearman do ranking de informação mútua.

A classe não lê nem grava arquivos: devolve DataFrames e dicts.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from dados import DERIVADOS, GRUPOS, NOMINAIS, faixa_aprovacao
from grafo_similaridade import GrafoSimilaridade

S1 = "Curricular units 1st sem"
MIN_TESTE = 20
ALFA = 0.05
MIN_POR_CATEGORIA = 30
MAX_VALORES_DISCRETOS = 10
NUM_QUANTIS = 4
TOPO = 10

# Fatores da Fig. 3: (rótulo, grupo, regra). A regra recebe o DataFrame
# e o treino (limiares de quantil vêm só do treino) e devolve a máscara.
Regra = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
FATORES_FIGURA: list[tuple[str, str, Regra]] = [
    ("S1: inscrito e 0% aprovado", "academico_s1",
     lambda x, t: (x[f"{S1} (enrolled)"] > 0)
     & (x[f"{S1} (approved)"] == 0)),
    ("S1: aprovou < 50%", "academico_s1",
     lambda x, t: (x["taxa_aprov_s1"] > 0) & (x["taxa_aprov_s1"] < 0.5)),
    ("S2: aprovou < 50%", "academico_s2",
     lambda x, t: (x["taxa_aprov_s2"] > 0) & (x["taxa_aprov_s2"] < 0.5)),
    ("Mensalidade atrasada", "financeiro",
     lambda x, t: x["Tuition fees up to date"] == 0),
    ("Curso: Eng. Informática", "ingresso",
     lambda x, t: x["Course"] == 9119),
    ("Devedor", "financeiro", lambda x, t: x["Debtor"] == 1),
    ("Escolaridade da mãe desconhecida", "familia",
     lambda x, t: x["Mother's qualification"] == 34),
    ("Formação prévia: básico 3º ciclo", "ingresso",
     lambda x, t: x["Previous qualification"] == 19),
    ("Ingresso “maiores de 23 anos”", "ingresso",
     lambda x, t: x["Application mode"] == 39),
    ("Idade no ingresso ≥ 28", "demografico",
     lambda x, t: x["Age at enrollment"] >= 28),
    ("Curso: Gestão (noturno)", "ingresso",
     lambda x, t: x["Course"] == 9991),
    ("Idade no ingresso 23–27", "demografico",
     lambda x, t: x["Age at enrollment"].between(23, 27)),
    ("Gênero masculino", "demografico", lambda x, t: x["Gender"] == 1),
    ("Casado(a)", "demografico", lambda x, t: x["Marital status"] == 2),
    ("Nota de admissão no 1º quartil", "ingresso",
     lambda x, t: x["Admission grade"]
     < t["Admission grade"].quantile(0.25)),
    ("Turno noturno", "ingresso",
     lambda x, t: x["Daytime/evening attendance"] == 0),
    ("Necessidades educativas especiais", "demografico",
     lambda x, t: x["Educational special needs"] == 1),
    ("Internacional", "demografico", lambda x, t: x["International"] == 1),
    ("Deslocado (mudou de cidade)", "demografico",
     lambda x, t: x["Displaced"] == 1),
    ("Coorte com desemprego de 12,4%", "macro",
     lambda x, t: x["Unemployment rate"] == 12.4),
    ("Idade no ingresso 17–19", "demografico",
     lambda x, t: x["Age at enrollment"] <= 19),
    ("Ingresso na 1ª fase geral", "ingresso",
     lambda x, t: x["Application mode"] == 1),
    ("Curso: Enfermagem", "ingresso", lambda x, t: x["Course"] == 9500),
    ("Bolsista", "financeiro", lambda x, t: x["Scholarship holder"] == 1),
    ("S1: média ≥ 14 (escala 0–20)", "academico_s1",
     lambda x, t: x[f"{S1} (grade)"] >= 14),
    ("S1: aprovou 100%", "academico_s1",
     lambda x, t: x["taxa_aprov_s1"] == 1),
    ("S2: aprovou 100%", "academico_s2",
     lambda x, t: x["taxa_aprov_s2"] == 1),
]


class AnaliseFatores:
    """Lift treino × teste, replicação das direções e ranking de IM.

    Parâmetros:
        min_teste: tamanho mínimo do grupo no teste para contar na
            replicação (grupos menores têm taxa instável).
        alfa: nível da correção de Benjamini-Hochberg no treino.
    """

    def __init__(self, min_teste: int = MIN_TESTE,
                 alfa: float = ALFA) -> None:
        self.min_teste = min_teste
        self.alfa = alfa

    def executar(self, base) -> dict:
        """Devolve ``lift``, ``faixas``, ``im`` e ``replicacao``."""
        treino = base.dados.loc[base.idx_treino]
        teste = base.dados.loc[base.idx_teste]
        faixas = self.faixas(treino, teste)
        im = self.ranking_im(treino, teste)
        return {"lift": self.lift_figura(treino, teste),
                "faixas": faixas, "im": im,
                "replicacao": self.replicacao(faixas, im)}

    # ------------------------------------------------------------------
    # Lift
    # ------------------------------------------------------------------
    def lift_figura(self, treino: pd.DataFrame,
                    teste: pd.DataFrame) -> pd.DataFrame:
        """Lift dos fatores da Fig. 3, com q (BH) do treino."""
        linhas = []
        for rotulo, grupo, regra in FATORES_FIGURA:
            linhas.append({"fator": rotulo, "grupo": grupo,
                           **self._comparar(regra(treino, treino),
                                            regra(teste, treino),
                                            treino, teste)})
        tabela = pd.DataFrame(linhas)
        tabela["q_treino"] = stats.false_discovery_control(
            tabela["p_treino"])
        return tabela

    def faixas(self, treino: pd.DataFrame,
               teste: pd.DataFrame) -> pd.DataFrame:
        """Lift de todas as faixas de todos os atributos (base da H2)."""
        linhas = []
        for grupo, colunas in self._atributos().items():
            for coluna in colunas:
                rot_tr, rot_te = self._faixas_de(treino[coluna],
                                                 teste[coluna])
                for faixa in pd.unique(rot_tr):
                    linhas.append({
                        "atributo": coluna, "grupo": grupo,
                        "faixa": str(faixa),
                        **self._comparar(pd.Series(rot_tr == faixa),
                                         pd.Series(rot_te == faixa),
                                         treino, teste)})
        tabela = pd.DataFrame(linhas)
        tabela["q_treino"] = stats.false_discovery_control(
            tabela["p_treino"])
        return tabela

    def ranking_im(self, treino: pd.DataFrame,
                   teste: pd.DataFrame) -> pd.DataFrame:
        """IM com o desfecho (bits) no treino e no teste, por atributo."""
        linhas = []
        for grupo, colunas in self._atributos().items():
            for coluna in colunas:
                linhas.append({
                    "atributo": coluna, "grupo": grupo,
                    "im_treino": GrafoSimilaridade.informacao_mutua(
                        treino[coluna], treino["evadiu"].to_numpy()),
                    "im_teste": GrafoSimilaridade.informacao_mutua(
                        teste[coluna], teste["evadiu"].to_numpy())})
        return (pd.DataFrame(linhas)
                .sort_values("im_treino", ascending=False)
                .reset_index(drop=True))

    def replicacao(self, faixas: pd.DataFrame, im: pd.DataFrame) -> dict:
        """Resumo da H2: direções que se repetem e Spearman da IM."""
        validas = faixas[(faixas["n_teste"] >= self.min_teste)
                         & (faixas["n_treino"] > 0)]
        mesma = (np.sign(validas["lift_treino"] - 1)
                 == np.sign(validas["lift_teste"] - 1))
        sig = validas["q_treino"] < self.alfa
        topo_tr = set(im.nlargest(TOPO, "im_treino")["atributo"])
        topo_te = set(im.nlargest(TOPO, "im_teste")["atributo"])
        return {
            "faixas_significativas": int(sig.sum()),
            "mesma_direcao_significativas": float(mesma[sig].mean()),
            "faixas_avaliadas": int(len(validas)),
            "mesma_direcao_todas": float(mesma.mean()),
            "spearman_im": float(stats.spearmanr(
                im["im_treino"], im["im_teste"]).statistic),
            "topo_im_em_comum": len(topo_tr & topo_te),
            "topo": TOPO, "min_teste": self.min_teste, "alfa": self.alfa,
        }

    # ------------------------------------------------------------------
    # Auxiliares
    # ------------------------------------------------------------------
    @staticmethod
    def _atributos() -> dict[str, list[str]]:
        """Atributos por grupo, com as taxas derivadas."""
        saida = {g: list(c) for g, c in GRUPOS.items()}
        for derivado, grupo in DERIVADOS.items():
            saida[grupo].append(derivado)
        return saida

    @staticmethod
    def _faixas_de(tr: pd.Series,
                   te: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Rótulos de faixa no treino e no teste, cortes só do treino.

        Mesmas regras da IM: taxas de aprovação nas faixas do DAG;
        nominais raras (< 30 no treino) em "outros"; numéricas com mais
        de 10 valores em quartis do treino; ausente é faixa própria.
        """
        if tr.name in DERIVADOS:
            return faixa_aprovacao(tr), faixa_aprovacao(te)
        if tr.name in NOMINAIS:
            contagem = tr.value_counts()
            comuns = set(contagem.index[contagem >= MIN_POR_CATEGORIA])

            def rotular(s: pd.Series) -> np.ndarray:
                return np.where(s.isin(comuns), s.astype(str), "outros")
            return rotular(tr), rotular(te)
        if tr.nunique() > MAX_VALORES_DISCRETOS:
            _, cortes = pd.qcut(tr, NUM_QUANTIS, retbins=True,
                                duplicates="drop")
            cortes[0], cortes[-1] = -np.inf, np.inf
            return (pd.cut(tr, cortes).astype(str).to_numpy(),
                    pd.cut(te, cortes).astype(str).to_numpy())
        return tr.astype(str).to_numpy(), te.astype(str).to_numpy()

    @staticmethod
    def _comparar(m_tr: pd.Series, m_te: pd.Series, treino: pd.DataFrame,
                  teste: pd.DataFrame) -> dict:
        """Taxa, lift e n do grupo no treino e no teste; p de Fisher."""
        m_tr = np.asarray(m_tr, dtype=bool)
        m_te = np.asarray(m_te, dtype=bool)
        y_tr = treino["evadiu"].to_numpy(dtype=bool)
        y_te = teste["evadiu"].to_numpy(dtype=bool)
        tabela = [[(y_tr & m_tr).sum(), (~y_tr & m_tr).sum()],
                  [(y_tr & ~m_tr).sum(), (~y_tr & ~m_tr).sum()]]
        taxa_tr = y_tr[m_tr].mean() if m_tr.any() else np.nan
        taxa_te = y_te[m_te].mean() if m_te.any() else np.nan
        return {
            "n_treino": int(m_tr.sum()), "taxa_treino": taxa_tr,
            "lift_treino": taxa_tr / y_tr.mean(),
            "n_teste": int(m_te.sum()), "taxa_teste": taxa_te,
            "lift_teste": taxa_te / y_te.mean(),
            "p_treino": float(stats.fisher_exact(tabela).pvalue),
        }
