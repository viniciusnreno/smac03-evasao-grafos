"""Pré-processamento do dataset UCI 697 (Predict Students' Dropout and Academic Success).

Transforma os atributos brutos em atributos categóricos/ordinais ("fatores"),
agrupados em três dimensões usadas pelo modelo multirrelacional:
  - ACAD  : desempenho acadêmico (ingresso e 1º/2º semestres)
  - SOCIO : situação socioeconômica e financeira
  - DEMO  : perfil demográfico e forma de ingresso
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- dicionários
COURSES = {
    33: "Tec. Biocombustíveis", 171: "Animação e Multimídia", 8014: "Serviço Social (noturno)",
    9003: "Agronomia", 9070: "Design de Comunicação", 9085: "Enfermagem Veterinária",
    9119: "Eng. Informática", 9130: "Equinicultura", 9147: "Gestão", 9238: "Serviço Social",
    9254: "Turismo", 9500: "Enfermagem", 9556: "Higiene Oral", 9670: "Publicidade e Marketing",
    9773: "Jornalismo", 9853: "Educação Básica", 9991: "Gestão (noturno)",
}

# Escolaridade dos pais -> nível ordinal (0 = fundamental ou menos ... 3 = pós-graduação)
_PARENT_EDU = {
    **{c: 0 for c in (9, 10, 11, 12, 14, 19, 26, 29, 30, 35, 36, 37, 38)},
    **{c: 1 for c in (1, 18, 22, 27, 39, 42)},
    **{c: 2 for c in (2, 3, 6, 40, 41)},
    **{c: 3 for c in (4, 5, 43, 44)},
    34: np.nan,  # desconhecido
}
PARENT_EDU_LABELS = {0: "Fundamental", 1: "Médio/Técnico", 2: "Superior", 3: "Pós-graduação"}

_APP_MODE = {
    **{c: "1ª fase" for c in (1, 5, 16)},
    **{c: "2ª/3ª fase" for c in (17, 18)},
    39: "Maiores de 23",
    **{c: "Transferência/mudança" for c in (42, 43, 51, 57)},
    **{c: "Já diplomado" for c in (7, 44, 53)},
}

OUTCOME_PT = {"Dropout": "Evasão", "Enrolled": "Matriculado", "Graduate": "Graduado"}


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def _ordinal_bins(series: pd.Series, labels: list[str], bins: list[float]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=range(len(labels)), include_lowest=True).astype(float)


def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Retorna (fatores, esquema).

    fatores : DataFrame com um código inteiro (ou NaN) por atributo.
    esquema : {atributo: {"group", "kind" ('ord'|'nom'), "labels"}}
    """
    r = raw
    f = pd.DataFrame(index=r.index)
    schema: dict[str, dict] = {}

    def add(name, values, group, kind, labels):
        f[name] = values
        schema[name] = {"group": group, "kind": kind, "labels": labels}

    # ------------------------------------------------------------- ACAD
    grade_lbl = ["<120", "120-140", "140-160", ">=160"]
    add("Nota de admissão", _ordinal_bins(r["Admission grade"], grade_lbl, [0, 120, 140, 160, 200]),
        "ACAD", "ord", grade_lbl)
    add("Nota formação anterior",
        _ordinal_bins(r["Previous qualification (grade)"], grade_lbl, [0, 120, 140, 160, 200]),
        "ACAD", "ord", grade_lbl)

    rate_lbl = ["0%", "1-49%", "50-99%", "100%"]
    for sem, tag in (("1st", "1º sem"), ("2nd", "2º sem")):
        enr = r[f"Curricular units {sem} sem (enrolled)"]
        app = r[f"Curricular units {sem} sem (approved)"]
        rate = np.where(enr > 0, app / enr.replace(0, np.nan), np.nan)
        rate = pd.Series(rate, index=r.index)
        code = pd.Series(np.select([rate == 0, rate < 0.5, rate < 1, rate >= 1], [0, 1, 2, 3], np.nan),
                         index=r.index).where(rate.notna())
        add(f"Aprovação {tag}", code, "ACAD", "ord", rate_lbl)

        # média das disciplinas aprovadas; indefinida se não houve aprovação
        # (evita duplicar a informação já capturada por "Aprovação = 0%")
        g = r[f"Curricular units {sem} sem (grade)"]
        g_lbl = ["<12", "12-14", ">=14"]
        gcode = pd.Series(np.select([g < 12, g < 14], [0, 1], 2), index=r.index).astype(float)
        gcode = gcode.where(app > 0)
        add(f"Média {tag}", gcode, "ACAD", "ord", g_lbl)

    woe = (r["Curricular units 1st sem (without evaluations)"] + r["Curricular units 2nd sem (without evaluations)"]) > 0
    add("Disciplina sem avaliação", woe.astype(float), "ACAD", "nom", ["Não", "Sim"])

    # ------------------------------------------------------------- SOCIO
    yn = ["Não", "Sim"]
    add("Devedor", r["Debtor"].astype(float), "SOCIO", "nom", yn)
    add("Mensalidade em dia", r["Tuition fees up to date"].astype(float), "SOCIO", "nom", yn)
    add("Bolsista", r["Scholarship holder"].astype(float), "SOCIO", "nom", yn)
    add("Deslocado", r["Displaced"].astype(float), "SOCIO", "nom", yn)
    plab = [PARENT_EDU_LABELS[i] for i in range(4)]
    add("Escolaridade mãe", r["Mother's qualification"].map(_PARENT_EDU).astype(float), "SOCIO", "ord", plab)
    add("Escolaridade pai", r["Father's qualification"].map(_PARENT_EDU).astype(float), "SOCIO", "ord", plab)

    # ------------------------------------------------------------- DEMO
    age_lbl = ["17-19", "20-22", "23-29", "30+"]
    add("Idade no ingresso", _ordinal_bins(r["Age at enrollment"], age_lbl, [0, 19, 22, 29, 100]),
        "DEMO", "ord", age_lbl)
    add("Gênero", r["Gender"].astype(float), "DEMO", "nom", ["Feminino", "Masculino"])
    add("Casado/união", (r["Marital status"] != 1).astype(float), "DEMO", "nom", yn)
    add("Turno noturno", (r["Daytime/evening attendance"] == 0).astype(float), "DEMO", "nom", yn)
    add("Internacional", r["International"].astype(float), "DEMO", "nom", yn)

    modes = sorted(set(_APP_MODE.values()) | {"Outros"})
    add("Forma de ingresso", r["Application mode"].map(_APP_MODE).fillna("Outros").map(modes.index).astype(float),
        "DEMO", "nom", modes)
    clist = [COURSES[c] for c in sorted(COURSES)]
    add("Curso", r["Course"].map(lambda c: clist.index(COURSES[c])).astype(float), "DEMO", "nom", clist)

    return f, schema


def factor_label(attr: str, code: float, schema: dict) -> str:
    return f"{attr}={schema[attr]['labels'][int(code)]}"


def onehot_factors(f: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Matriz binária estudante x fator (atributo=valor), ignorando NaN."""
    cols = {}
    for attr in f.columns:
        for code in sorted(f[attr].dropna().unique()):
            cols[factor_label(attr, code, schema)] = (f[attr] == code).astype(np.int8)
    return pd.DataFrame(cols, index=f.index)


ENTRY_ONLY = [  # atributos conhecidos já no ingresso (cenário de alerta precoce)
    "Nota de admissão", "Nota formação anterior", "Devedor", "Mensalidade em dia", "Bolsista",
    "Deslocado", "Escolaridade mãe", "Escolaridade pai", "Idade no ingresso", "Gênero",
    "Casado/união", "Turno noturno", "Internacional", "Forma de ingresso", "Curso",
]
