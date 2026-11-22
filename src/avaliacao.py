"""Avaliação. Classe: ``Avaliador``.

Responsável: Rafael.

Protocolo do relatório (seções 3 e 4):

- CV estratificada de 5 dobras **só no treino** para comparar métodos,
  ablação por grupo (H3) e eliminação backward de atributos.
- Teste usado uma única vez, com a configuração já fixada: acurácia
  balanceada (métrica principal), IC 95% por bootstrap (1.000
  reamostras), AUC quando há nota contínua, McNemar exato com correção
  de Holm (H4).
- Robustez: 10 divisões 80/20 com sementes sorteadas a partir da 42.

Comparadores da Tab. 3: classe majoritária, regra de 1 variável
(``RegraUmaVariavel``), caminho no DAG (G3), corte mínimo (G1) e
regressão logística (``RegressaoLogisticaReferencia``, só referência
externa, não faz parte da solução).

As classes não leem nem gravam arquivos: devolvem dicts e DataFrames,
que o ``Pipeline`` grava em ``outputs/``.
"""

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dados import DERIVADOS, GRUPOS, MOMENTOS, NOMINAIS, SEMENTE
from dag_trajetorias import DagTrajetorias
from grafo_similaridade import K_PADRAO, GrafoSimilaridade

DOBRAS = 5
N_BOOTSTRAP = 1000
N_SPLITS = 10
CONFIANCA = 0.95

# Classificador genérico usado na CV, na ablação e na robustez:
# classificar(idx_treino, idx_teste, excluir=()) -> {aluno: evasão?}.
Classificador = Callable[..., dict[int, bool]]


def _acuracia_balanceada(y: np.ndarray, prev: np.ndarray) -> float:
    """(recall da evasão + especificidade) / 2."""
    y = np.asarray(y, dtype=bool)
    prev = np.asarray(prev, dtype=bool)
    return (prev[y].mean() + (~prev[~y]).mean()) / 2


class RegraUmaVariavel:
    """Melhor regra de um atributo só, aprendida no treino (H1 e H4).

    Para cada atributo numérico ou binário e cada limiar (ponto médio
    entre valores vizinhos), testa "x >= limiar" e "x < limiar" e fica
    com a regra de maior acurácia balanceada no treino. Valor ausente
    (taxa sem inscrição) recebe a previsão da maioria ponderada do grupo
    ausente no treino: evasão se a taxa dele passa a taxa geral.
    """

    def __init__(self) -> None:
        self.atributo: str | None = None
        self.sentido: str | None = None
        self.limiar: float | None = None
        self.ausente_evasao = False
        self.acuracia_treino: float | None = None

    def ajustar(self, x: pd.DataFrame,
                y: np.ndarray) -> "RegraUmaVariavel":
        """Escolhe atributo, sentido e limiar no treino."""
        y = np.asarray(y, dtype=bool)
        taxa = y.mean()
        melhor = -1.0
        for coluna in x.columns:
            if coluna in NOMINAIS:
                continue
            valores = x[coluna].to_numpy(dtype=float)
            ausente = np.isnan(valores)
            ausente_ev = bool(ausente.any() and y[ausente].mean() > taxa)
            distintos = np.unique(valores[~ausente])
            if len(distintos) < 2:
                continue
            for limiar in (distintos[:-1] + distintos[1:]) / 2:
                acima = valores >= limiar
                for sentido, prev in ((">=", acima), ("<", ~acima)):
                    prev = np.where(ausente, ausente_ev, prev)
                    acuracia = _acuracia_balanceada(y, prev)
                    if acuracia > melhor:
                        melhor = acuracia
                        self.atributo, self.sentido = coluna, sentido
                        self.limiar = float(limiar)
                        self.ausente_evasao = ausente_ev
        if self.atributo is None:
            raise ValueError("nenhum atributo numérico com dois valores")
        self.acuracia_treino = float(melhor)
        return self

    def prever(self, x: pd.DataFrame) -> np.ndarray:
        """Previsão (True = evasão) para os alunos de ``x``."""
        if self.atributo is None:
            raise RuntimeError("chame ajustar antes de prever")
        valores = x[self.atributo].to_numpy(dtype=float)
        acima = valores >= self.limiar
        prev = acima if self.sentido == ">=" else ~acima
        return np.where(np.isnan(valores), self.ausente_evasao, prev)

    def descricao(self) -> str:
        """Texto da regra, ex.: ``taxa_aprov_s1 < 0.782``."""
        return f"{self.atributo} {self.sentido} {self.limiar:.3f}"


class RegressaoLogisticaReferencia:
    """Regressão logística: só referência externa, não é a solução.

    Nominais em one-hot; numéricas padronizadas, com ausente trocado por
    0 e um indicador de ausência; classes balanceadas, porque a métrica
    principal é a acurácia balanceada.
    """

    def __init__(self, semente: int = SEMENTE) -> None:
        self.semente = semente
        self._modelo = None

    def ajustar(self, x: pd.DataFrame,
                y: np.ndarray) -> "RegressaoLogisticaReferencia":
        nominais = [c for c in x.columns if c in NOMINAIS]
        numericas = [c for c in x.columns if c not in NOMINAIS]
        preparo = ColumnTransformer([
            ("nominais", OneHotEncoder(handle_unknown="ignore"), nominais),
            ("numericas", make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0,
                              add_indicator=True),
                StandardScaler()), numericas),
        ])
        self._modelo = make_pipeline(preparo, LogisticRegression(
            class_weight="balanced", max_iter=5000,
            random_state=self.semente))
        self._modelo.fit(x, np.asarray(y, dtype=bool))
        return self

    def nota(self, x: pd.DataFrame) -> np.ndarray:
        """Probabilidade estimada de evasão."""
        if self._modelo is None:
            raise RuntimeError("chame ajustar antes")
        return self._modelo.predict_proba(x)[:, 1]

    def prever(self, x: pd.DataFrame) -> np.ndarray:
        return self.nota(x) >= 0.5


class Avaliador:
    """CV, teste, IC, McNemar com Holm, ablação, backward e robustez.

    Parâmetros:
        dobras: dobras da CV estratificada (só treino).
        n_bootstrap: reamostras do IC no teste.
        n_splits: divisões 80/20 da análise de robustez.
        semente: semente de todos os sorteios (regra 5).
    """

    def __init__(self, dobras: int = DOBRAS, n_bootstrap: int = N_BOOTSTRAP,
                 n_splits: int = N_SPLITS, semente: int = SEMENTE) -> None:
        self.dobras = dobras
        self.n_bootstrap = n_bootstrap
        self.n_splits = n_splits
        self.semente = semente

    # ------------------------------------------------------------------
    # Métricas e testes
    # ------------------------------------------------------------------
    @staticmethod
    def metricas(y_real: np.ndarray, y_prev: np.ndarray) -> dict:
        """Métricas de classificação; evasão (True) é a classe positiva."""
        y_real = np.asarray(y_real, dtype=bool)
        y_prev = np.asarray(y_prev, dtype=bool)
        vp = int((y_real & y_prev).sum())
        fp = int((~y_real & y_prev).sum())
        fn = int((y_real & ~y_prev).sum())
        vn = int((~y_real & ~y_prev).sum())
        recall = vp / (vp + fn) if vp + fn else 0.0
        especificidade = vn / (vn + fp) if vn + fp else 0.0
        precisao = vp / (vp + fp) if vp + fp else 0.0
        f1 = (2 * precisao * recall / (precisao + recall)
              if precisao + recall else 0.0)
        return {
            "acuracia_balanceada": (recall + especificidade) / 2,
            "recall_evasao": recall,
            "especificidade": especificidade,
            "precisao_evasao": precisao,
            "f1_evasao": f1,
            "acuracia": (vp + vn) / len(y_real),
            "vp": vp, "fp": fp, "fn": fn, "vn": vn,
        }

    @staticmethod
    def auc(y_real: np.ndarray, nota: np.ndarray) -> float:
        """Área sob a curva ROC de uma nota contínua de risco."""
        return float(roc_auc_score(np.asarray(y_real, dtype=bool), nota))

    def ic_bootstrap(self, y_real: np.ndarray,
                     y_prev: np.ndarray) -> tuple[float, float]:
        """IC 95% (percentil) da acurácia balanceada por bootstrap."""
        y_real = np.asarray(y_real, dtype=bool)
        y_prev = np.asarray(y_prev, dtype=bool)
        rng = np.random.default_rng(self.semente)
        n = len(y_real)
        valores = []
        for _ in range(self.n_bootstrap):
            amostra = rng.integers(0, n, n)
            y, p = y_real[amostra], y_prev[amostra]
            if y.all() or not y.any():
                continue
            valores.append(_acuracia_balanceada(y, p))
        alfa = (1 - CONFIANCA) / 2
        inf, sup = np.quantile(valores, [alfa, 1 - alfa])
        return float(inf), float(sup)

    @staticmethod
    def mcnemar(y_real: np.ndarray, prev_a: np.ndarray,
                prev_b: np.ndarray) -> dict:
        """McNemar exato (binomial) entre dois métodos nos mesmos alunos.

        ``so_a``: alunos que só A acerta; ``so_b``: só B acerta.
        """
        y_real = np.asarray(y_real, dtype=bool)
        acerto_a = np.asarray(prev_a, dtype=bool) == y_real
        acerto_b = np.asarray(prev_b, dtype=bool) == y_real
        so_a = int((acerto_a & ~acerto_b).sum())
        so_b = int((~acerto_a & acerto_b).sum())
        p = (stats.binomtest(so_a, so_a + so_b, 0.5).pvalue
             if so_a + so_b else 1.0)
        return {"so_a": so_a, "so_b": so_b, "p": float(p)}

    @staticmethod
    def holm(pvalores: Sequence[float]) -> list[float]:
        """P-valores corrigidos por Holm-Bonferroni (mesma ordem)."""
        m = len(pvalores)
        ordem = np.argsort(pvalores, kind="stable")
        corrigidos = [0.0] * m
        maximo = 0.0
        for posicao, i in enumerate(ordem):
            maximo = max(maximo, min(1.0, (m - posicao) * pvalores[i]))
            corrigidos[i] = maximo
        return corrigidos

    # ------------------------------------------------------------------
    # Classificadores (fábricas para CV, ablação e robustez)
    # ------------------------------------------------------------------
    @staticmethod
    def classificador_g1(base, momento: str, **parametros) -> Classificador:
        """Corte mínimo no kNN de Gower (``GrafoSimilaridade``)."""
        k = parametros.pop("k", K_PADRAO[momento])

        def classificar(idx_treino, idx_teste, excluir=()):
            modelo = GrafoSimilaridade(momento, k, excluir=excluir,
                                       **parametros)
            modelo.construir(base, idx_treino, idx_teste)
            return modelo.executar()["previsoes"]
        return classificar

    @staticmethod
    def classificador_dag(base, momento: str) -> Classificador:
        """Caminho no DAG de trajetórias (``DagTrajetorias``)."""
        def classificar(idx_treino, idx_teste, excluir=()):
            modelo = DagTrajetorias(momento)
            modelo.construir(base, idx_treino, idx_teste)
            return modelo.executar()["previsoes"]
        return classificar

    @staticmethod
    def classificador_regra(base, momento: str) -> Classificador:
        """Regra de 1 variável (com as taxas derivadas)."""
        def classificar(idx_treino, idx_teste, excluir=()):
            regra = RegraUmaVariavel().ajustar(
                base.X(idx_treino, momento, True).drop(columns=list(excluir)),
                base.y(idx_treino))
            prev = regra.prever(base.X(idx_teste, momento, True))
            return dict(zip(idx_teste, map(bool, prev)))
        return classificar

    # ------------------------------------------------------------------
    # Validação cruzada (só treino)
    # ------------------------------------------------------------------
    def particoes(self, base,
                  idx: Iterable[int] | None = None
                  ) -> list[tuple[list[int], list[int]]]:
        """Dobras estratificadas (treino, validação) dos alunos ``idx``."""
        idx = np.array(sorted(base.idx_treino if idx is None else idx))
        cv = StratifiedKFold(self.dobras, shuffle=True,
                             random_state=self.semente)
        return [(sorted(idx[a].tolist()), sorted(idx[b].tolist()))
                for a, b in cv.split(idx, base.y(idx.tolist()))]

    def validacao_cruzada(self, base, classificar: Classificador,
                          excluir: Iterable[str] = (),
                          idx: Iterable[int] | None = None) -> dict:
        """Acurácia balanceada por dobra, média e desvio padrão."""
        excluir = tuple(excluir)
        por_dobra = []
        for treino, validacao in self.particoes(base, idx):
            prev = classificar(treino, validacao, excluir=excluir)
            por_dobra.append(_acuracia_balanceada(
                base.y(validacao), [prev[i] for i in validacao]))
        return {"media": float(np.mean(por_dobra)),
                "dp": float(np.std(por_dobra, ddof=1)),
                "dobras": por_dobra}

    def ablacao_grupos(self, base, momento: str,
                       classificar: Classificador) -> pd.DataFrame:
        """Δ acurácia balanceada (sem − com) ao tirar cada grupo (H3).

        Dobras pareadas; IC 95% de t com ``dobras - 1`` graus de
        liberdade. Essencial: IC todo abaixo de 0 (tirar piora).
        Prejudicial: IC todo acima de 0. Neutro: o IC cruza o 0.
        """
        com = self.validacao_cruzada(base, classificar)["dobras"]
        linhas = []
        for grupo in MOMENTOS[momento]:
            excluir = self.atributos_do_grupo(grupo)
            sem = self.validacao_cruzada(base, classificar, excluir)
            delta = np.array(sem["dobras"]) - np.array(com)
            inf, sup = self._ic_t(delta)
            classe = ("essencial" if sup < 0 else
                      "prejudicial" if inf > 0 else "neutro")
            linhas.append({
                "momento": momento, "grupo": grupo,
                "num_atributos": len(excluir),
                "acuracia_com": float(np.mean(com)),
                "acuracia_sem": sem["media"],
                "delta": float(delta.mean()), "ic_inf": inf, "ic_sup": sup,
                "classe": classe,
            })
        return pd.DataFrame(linhas)

    def eliminacao_backward(self, base, momento: str,
                            classificar: Classificador,
                            tolerancia: float = 0.0,
                            progresso: Callable[[str], None] | None = None
                            ) -> dict:
        """Tira atributos, do de menor IM para o de maior, enquanto a CV
        não piorar mais que ``tolerancia`` (só treino).

        Versão gulosa ordenada: um teste por atributo, em vez de testar
        todos a cada passo, para caber no tempo do trabalho.
        """
        avisar = progresso or (lambda _m: None)
        treino = base.idx_treino
        y = base.y(treino)
        x = base.X(treino, momento, True)
        ordem = sorted(x.columns, key=lambda c: (
            GrafoSimilaridade.informacao_mutua(x[c], y), c))
        atual = self.validacao_cruzada(base, classificar)["media"]
        removidos: list[str] = []
        historico = [{"atributo": None, "acuracia": atual,
                      "removido": False}]
        for coluna in ordem[:-1]:
            tentativa = self.validacao_cruzada(
                base, classificar, removidos + [coluna])["media"]
            removido = tentativa >= atual - tolerancia
            if removido:
                removidos.append(coluna)
                atual = tentativa
            historico.append({"atributo": coluna, "acuracia": tentativa,
                              "removido": removido})
            avisar(f"  backward {momento}: {coluna} -> {tentativa:.3f} "
                   f"({'sai' if removido else 'fica'})")
        mantidos = [c for c in x.columns if c not in removidos]
        return {"momento": momento, "mantidos": mantidos,
                "removidos": removidos, "acuracia_cv": atual,
                "historico": historico}

    def robustez(self, base, classificar: Classificador) -> dict:
        """Acurácia balanceada em ``n_splits`` divisões 80/20."""
        rng = np.random.default_rng(self.semente)
        sementes = rng.integers(0, 2**31 - 1, self.n_splits).tolist()
        acuracias = []
        for semente in sementes:
            treino, teste = base.dividir(semente)
            prev = classificar(treino, teste)
            acuracias.append(float(_acuracia_balanceada(
                base.y(teste), [prev[i] for i in teste])))
        return {"sementes": sementes, "acuracias": acuracias,
                "media": float(np.mean(acuracias)),
                "dp": float(np.std(acuracias, ddof=1))}

    # ------------------------------------------------------------------
    # Teste (uma única vez, configuração fixada)
    # ------------------------------------------------------------------
    def avaliar_teste(self, y_real: np.ndarray,
                      previsoes: dict[str, np.ndarray],
                      notas: dict[str, np.ndarray] | None = None
                      ) -> pd.DataFrame:
        """Métricas, IC 95% e AUC de cada método no teste."""
        notas = notas or {}
        linhas = []
        for metodo, prev in previsoes.items():
            inf, sup = self.ic_bootstrap(y_real, prev)
            linhas.append({
                "metodo": metodo, **self.metricas(y_real, prev),
                "ic_inf": inf, "ic_sup": sup,
                "auc": (self.auc(y_real, notas[metodo])
                        if metodo in notas else np.nan),
            })
        return pd.DataFrame(linhas)

    def comparar(self, y_por_momento: dict[str, np.ndarray],
                 previsoes: dict[str, dict[str, np.ndarray]],
                 pares: Sequence[tuple[str, str]]) -> pd.DataFrame:
        """McNemar de cada par em cada momento; Holm sobre a família."""
        linhas = []
        for momento, y in y_por_momento.items():
            for a, b in pares:
                r = self.mcnemar(y, previsoes[momento][a],
                                 previsoes[momento][b])
                linhas.append({"momento": momento, "metodo_a": a,
                               "metodo_b": b, **r})
        tabela = pd.DataFrame(linhas)
        tabela["p_holm"] = self.holm(tabela["p"].tolist())
        return tabela

    # ------------------------------------------------------------------
    # Protocolo completo
    # ------------------------------------------------------------------
    def executar(self, base, momentos: Sequence[str] = ("T0", "T1", "T2"),
                 ablacao: bool = True, backward: bool = True,
                 robustez: bool = True,
                 progresso: Callable[[str], None] | None = None) -> dict:
        """Roda o protocolo de avaliação e devolve as tabelas.

        Chaves: ``cv``, ``teste``, ``previsoes``, ``mcnemar``,
        ``regras``, e, se pedidos, ``ablacao``, ``backward`` e
        ``robustez``.
        """
        avisar = progresso or (lambda _m: None)
        teste = base.idx_teste
        y_teste = base.y(teste)
        cv, tabelas, regras, backward_res, robusto = [], [], {}, {}, []
        ablacoes = []
        previsoes = pd.DataFrame({"aluno": teste, "evadiu": y_teste})
        prev_momento: dict[str, dict[str, np.ndarray]] = {}

        for m in momentos:
            g1 = self.classificador_g1(base, m)
            fabricas = {"corte_minimo": g1,
                        "caminho_dag": self.classificador_dag(base, m),
                        "regra_1_variavel": self.classificador_regra(base, m)}

            avisar(f"Avaliação {m}: CV de {self.dobras} dobras no treino")
            for metodo, fabrica in fabricas.items():
                r = self.validacao_cruzada(base, fabrica)
                cv.append({"momento": m, "metodo": metodo,
                           "media": r["media"], "dp": r["dp"]})

            avisar(f"Avaliação {m}: teste (uma vez) e bootstrap")
            prev, notas = self._prever_teste(base, m)
            if backward:
                avisar(f"Avaliação {m}: eliminação backward (CV)")
                res = self.eliminacao_backward(base, m, g1,
                                               progresso=avisar)
                backward_res[m] = res
                prev["corte_essenciais"] = np.array(
                    [g1(base.idx_treino, teste, res["removidos"])[i]
                     for i in teste])
            tabela = self.avaliar_teste(y_teste, prev, notas)
            tabela.insert(0, "momento", m)
            tabelas.append(tabela)
            prev_momento[m] = prev
            for metodo, valores in prev.items():
                previsoes[f"{m}_{metodo}"] = valores
            regra = RegraUmaVariavel().ajustar(
                base.X(base.idx_treino, m, True), base.y(base.idx_treino))
            regras[m] = {"regra": regra.descricao(),
                         "acuracia_treino": regra.acuracia_treino}

            if ablacao:
                avisar(f"Avaliação {m}: ablação por grupo (CV)")
                ablacoes.append(self.ablacao_grupos(base, m, g1))
            if robustez:
                avisar(f"Avaliação {m}: {self.n_splits} divisões 80/20")
                r = self.robustez(base, g1)
                robusto.append({"momento": m, "metodo": "corte_minimo",
                                "media": r["media"], "dp": r["dp"],
                                "acuracias": r["acuracias"],
                                "sementes": r["sementes"]})

        saida = {
            "cv": pd.DataFrame(cv),
            "teste": pd.concat(tabelas, ignore_index=True),
            "previsoes": previsoes,
            "mcnemar": self.comparar(
                {m: y_teste for m in momentos}, prev_momento,
                [("corte_minimo", "regra_1_variavel"),
                 ("corte_minimo", "caminho_dag")]),
            "regras": regras,
        }
        if ablacao:
            saida["ablacao"] = pd.concat(ablacoes, ignore_index=True)
        if backward:
            saida["backward"] = backward_res
        if robustez:
            saida["robustez"] = pd.DataFrame(robusto)
        return saida

    # ------------------------------------------------------------------
    # Auxiliares
    # ------------------------------------------------------------------
    def _prever_teste(self, base, momento: str
                      ) -> tuple[dict[str, np.ndarray],
                                 dict[str, np.ndarray]]:
        """Previsões (e notas) de cada método no teste oficial."""
        treino, teste = base.idx_treino, base.idx_teste
        x_treino = base.X(treino, momento, True)
        x_teste = base.X(teste, momento, True)
        y_treino = base.y(treino)

        g1 = GrafoSimilaridade(momento, K_PADRAO[momento])
        g1.construir(base)
        prev_g1 = g1.executar()["previsoes"]

        dag = DagTrajetorias(momento)
        dag.construir(base)
        r_dag = dag.executar()

        regra = RegraUmaVariavel().ajustar(x_treino, y_treino)
        logistica = RegressaoLogisticaReferencia(self.semente).ajustar(
            x_treino, y_treino)

        previsoes = {
            "classe_majoritaria": np.zeros(len(teste), dtype=bool),
            "regra_1_variavel": regra.prever(x_teste).astype(bool),
            "caminho_dag": np.array([r_dag["previsoes"][i] for i in teste]),
            "corte_minimo": np.array([prev_g1[i] for i in teste]),
            "regressao_logistica": logistica.prever(x_teste),
        }
        notas = {
            "caminho_dag": np.array([r_dag["risco"][i] for i in teste]),
            "regressao_logistica": logistica.nota(x_teste),
        }
        return previsoes, notas

    @staticmethod
    def atributos_do_grupo(grupo: str) -> list[str]:
        """Colunas do grupo, com a taxa derivada dele, se houver."""
        return GRUPOS[grupo] + [d for d, g in DERIVADOS.items()
                                if g == grupo]

    def _ic_t(self, valores: np.ndarray) -> tuple[float, float]:
        """IC de t para a média de ``valores`` (dobras pareadas)."""
        n = len(valores)
        media = float(np.mean(valores))
        erro = float(np.std(valores, ddof=1)) / np.sqrt(n)
        t = stats.t.ppf((1 + CONFIANCA) / 2, n - 1)
        return media - t * erro, media + t * erro
