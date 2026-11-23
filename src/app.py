"""Interface gráfica do trabalho (Streamlit).

Rodar a partir da raiz do repositório::

    source .venv/bin/activate
    streamlit run src/app.py

Este módulo só cuida da tela: os grafos vêm das classes do projeto
(``GrafoSimilaridade``, ``ArvoreDependencias``, ``DagTrajetorias``), o
desenho vem de ``visual`` e o Kruskal passo a passo de
``passo_a_passo``. Os números oficiais (métricas, IC, McNemar) são
lidos de ``outputs/``; nada é digitado à mão.

Quatro páginas: resumo (avaliação e o que mais pesa) e uma por grafo,
cada uma com o desenho, o quadro "como ler" e o simulador do grafo.
Sem vazamento: os modelos usam os hiperparâmetros gravados em
``outputs/hiperparametros.json``.
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from arvore_dependencias import DESFECHO, ArvoreDependencias
from dados import MOMENTOS as MOMENTOS_DADOS
from dados import BaseDeDados
from dag_trajetorias import CAMADAS_POR_MOMENTO, INICIO, DagTrajetorias
from explicacao import ExplicadorGower, grupo_do_atributo
from figuras import CORES_GRUPO, NOMES_GRUPO, ROTULOS
from grafo_similaridade import GrafoSimilaridade
from passo_a_passo import DemoKruskal, ExploradorArvore
from visual import (COR_EVASAO, COR_SEGURA, GrafoECharts,
                    MontadorVisual)

SAIDA = Path(__file__).resolve().parent.parent / "outputs"
MOMENTOS = ("T0", "T1", "T2")
NOME_MOMENTO = {
    "T0": "T0 · ingresso",
    "T1": "T1 · após o 1º semestre",
    "T2": "T2 · após o 2º semestre",
}
NOME_METODO = {
    "corte_minimo": "Corte mínimo (G1)",
    "corte_essenciais": "Corte mínimo, só variáveis essenciais",
    "caminho_dag": "Caminho no DAG (G3)",
    "regra_1_variavel": "Regra de 1 variável",
    "regressao_logistica": "Regressão logística (referência externa)",
    "classe_majoritaria": "Classe majoritária",
}
TOLERANCIA = 1e-4
TOPO_ATRIBUTOS = 10          # atributos no painel "por que estão ligados"
CAMADAS_SIMULADOR = {"L0": "Ingresso", "L1": "Financeiro",
                     "L2": "Aprovação no S1", "L3": "Aprovação no S2"}
ORDEM_GRUPOS = ("academico_s2", "academico_s1", "financeiro", "ingresso",
                "demografico", "familia", "macro")


# ----------------------------------------------------------------------
# Dados e modelos (cache)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando a base...")
def carregar_base() -> BaseDeDados:
    return BaseDeDados()


@st.cache_data
def ler_json(nome: str) -> dict:
    return json.loads((SAIDA / nome).read_text(encoding="utf-8"))


@st.cache_data
def ler_csv(nome: str) -> pd.DataFrame:
    return pd.read_csv(SAIDA / nome)


def hiperparametros_g1(momento: str) -> tuple[int, float]:
    """k e λ oficiais, como gravados pelo pipeline."""
    h = ler_json("hiperparametros.json")["g1"]
    return int(h["k"][momento]), float(h["lam"])


@st.cache_resource(show_spinner="Construindo o kNN de Gower e rodando "
                                "o Edmonds–Karp...")
def modelo_g1(momento: str) -> tuple[GrafoSimilaridade, dict, dict]:
    k, lam = hiperparametros_g1(momento)
    modelo = GrafoSimilaridade(momento, k, lam=lam)
    modelo.construir(carregar_base())
    resultado = modelo.executar()
    return modelo, resultado, modelo.destaques()


@st.cache_resource
def explicador_g1(momento: str) -> ExplicadorGower:
    return ExplicadorGower(modelo_g1(momento)[0], carregar_base())


@st.cache_resource(show_spinner="Rodando o Kruskal...")
def modelo_g2(momento: str) -> tuple[ArvoreDependencias, dict]:
    modelo = ArvoreDependencias(momento)
    modelo.construir(carregar_base())
    return modelo, modelo.executar()


@st.cache_resource(show_spinner="Rodando Kahn e Bellman-Ford...")
def modelo_g3(momento: str) -> tuple[DagTrajetorias, dict]:
    modelo = DagTrajetorias(momento)
    modelo.construir(carregar_base())
    return modelo, modelo.executar()


@st.cache_resource
def demo_kruskal(momento: str) -> dict:
    modelo, resultado = modelo_g2(momento)
    return DemoKruskal().executar(modelo, resultado["arestas"])


@st.cache_resource
def explorador_g2(momento: str) -> ExploradorArvore:
    return ExploradorArvore(modelo_g2(momento)[1]["arestas"], DESFECHO)


@st.cache_data
def estados_treino() -> pd.DataFrame:
    """Estados L0–L3 dos alunos de treino, com o desfecho."""
    base = carregar_base()
    estados = DagTrajetorias.estados(base.dados.loc[base.idx_treino])
    estados["evadiu"] = base.y(base.idx_treino)
    return estados


# ----------------------------------------------------------------------
# Formatação (padrão brasileiro: vírgula decimal, ponto no milhar)
# ----------------------------------------------------------------------
def br(x: float, casas: int = 3) -> str:
    texto = f"{x:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x: float, casas: int = 0) -> str:
    return br(100 * x, casas) + "%"


# ----------------------------------------------------------------------
# Componentes de tela
# ----------------------------------------------------------------------
def modo_escuro() -> bool:
    """Modo ativo (claro/escuro), para os desenhos fora do tema."""
    return st.context.theme.type == "dark"


def montador() -> MontadorVisual:
    return MontadorVisual(escuro=modo_escuro())


def desenhar(desenho: tuple, altura: int = 560) -> None:
    vertices, arestas, opcoes = desenho
    px = px_desenho(altura)
    html = MontadorVisual(px, modo_escuro()).html(vertices, arestas, opcoes)
    st.iframe(html, height=px + 12)


def escolher_momento(chave: str, padrao: str = "T1") -> str:
    return st.segmented_control(
        "Momento", MOMENTOS, default=padrao, required=True,
        format_func=NOME_MOMENTO.get, key=chave)


def apresentacao() -> bool:
    """Modo apresentação: sem quadros de leitura nem detalhes, grafos
    maiores (para projetor e vídeo). Ligado na barra lateral."""
    return bool(st.session_state.get("apresentacao", False))


def px_desenho(px: int) -> int:
    """Altura de um desenho; 30% maior no modo apresentação."""
    return round(px * 1.3) if apresentacao() else px


def nota(texto: str) -> None:
    """Texto explicativo curto; some no modo apresentação."""
    if not apresentacao():
        st.caption(texto)


def como_ler(vertice: str, aresta: str, leitura: list[str]) -> None:
    """Quadro fixo com o que são vértice e aresta e como ler o desenho."""
    if apresentacao():
        return
    with st.container(border=True):
        c1, c2 = st.columns([1, 1.5])
        c1.markdown(f"**:material/circle: Vértice** · {vertice}\n\n"
                    f"**:material/timeline: Aresta** · {aresta}")
        c2.markdown("\n".join(f"- {t}" for t in leitura))


def avisar_divergencia(confere: bool) -> None:
    """Só aparece se o grafo refeito na tela diferir de ``outputs/``."""
    if not confere:
        st.warning("O resultado na tela difere de `outputs/`. Rode "
                   "`python src/main.py` de novo.")


# ----------------------------------------------------------------------
# Páginas
# ----------------------------------------------------------------------
def pagina_resumo() -> None:
    st.title("Evasão estudantil como grafos")

    dados = ler_json("resultados.json")["dados"]
    c = st.columns(4)
    c[0].metric("Alunos (sem Enrolled)", br(dados["alunos"], 0))
    c[1].metric("Treino / teste", f"{br(dados['treino'], 0)} / "
                f"{br(dados['teste'], 0)}")
    c[2].metric("Taxa de evasão", pct(dados['taxa_evasao'], 1))
    c[3].metric("Semente", "42")

    st.subheader("Os três grafos")
    st.table(pd.DataFrame([
        ["G1 · Similaridade", "alunos", "kNN de Gower",
         "Edmonds–Karp + BFS do corte",
         "fluxo máximo = corte mínimo"],
        ["G2 · Dependências", "atributos + Desfecho",
         "informação mútua", "Kruskal + union-find",
         "árvore geradora de peso máximo"],
        ["G3 · Trajetórias", "estados (ingresso → S2 → desfecho)",
         "alunos que passam pelo arco", "Kahn + Bellman-Ford",
         "ordem topológica; caminho de maior risco"],
    ], columns=["Grafo", "Vértices", "Arestas", "Algoritmo",
                "O que garante"]).set_index("Grafo"))

    st.subheader("Acurácia balanceada no teste (IC 95%, bootstrap)")
    teste = ler_csv("avaliacao_teste.csv")
    teste["valor"] = teste.apply(
        lambda r: f"{br(r.acuracia_balanceada, 3)} "
                  f"[{br(r.ic_inf, 3)}; {br(r.ic_sup, 3)}]", axis=1)
    tabela = teste.pivot(index="metodo", columns="momento",
                         values="valor")
    tabela = tabela.reindex([m for m in NOME_METODO if m in tabela.index])
    tabela.index = tabela.index.map(NOME_METODO)
    tabela.index.name = "Método"
    tabela.columns = [NOME_MOMENTO[m] for m in tabela.columns]
    st.dataframe(tabela, width="stretch")
    st.caption("Fonte: `outputs/avaliacao_teste.csv`.")

    st.subheader("Corte mínimo contra os outros métodos (McNemar + Holm)")
    mc = ler_csv("avaliacao_mcnemar.csv").copy()
    mc["comparação"] = (mc.metodo_a.map(NOME_METODO) + " × "
                        + mc.metodo_b.map(NOME_METODO))
    mc["momento"] = mc.momento.map(NOME_MOMENTO)
    mc["significativo (5%)"] = np.where(mc.p_holm < 0.05, "sim", "não")
    st.dataframe(
        mc[["momento", "comparação", "so_a", "so_b", "p_holm",
            "significativo (5%)"]].rename(columns={
                "so_a": "só A acerta", "so_b": "só B acerta",
                "p_holm": "p (Holm)"}),
        hide_index=True, width="stretch",
        column_config={"p (Holm)": st.column_config.NumberColumn(
            format="%.2e")})
    st.caption("Fonte: `outputs/avaliacao_mcnemar.csv`.")

    secao_fatores()


def tabela_pesos() -> pd.DataFrame:
    """Pesos do G1 por momento: IM com o desfecho (treino), soma 1."""
    g1 = ler_json("resultados.json")["g1"]
    linhas = [
        {"momento": m, "atributo": a, "peso": p,
         "rotulo": ROTULOS.get(a, a),
         "grupo": NOMES_GRUPO[grupo_do_atributo(a)]}
        for m in MOMENTOS for a, p in g1[m]["pesos"].items()
    ]
    return pd.DataFrame(linhas)


def escala_grupos() -> alt.Scale:
    return alt.Scale(domain=[NOMES_GRUPO[g] for g in ORDEM_GRUPOS],
                     range=[CORES_GRUPO[g] for g in ORDEM_GRUPOS])


def secao_fatores() -> None:
    st.header("O que mais pesa", divider="gray")
    momento = escolher_momento("momento_fatores")

    pesos = tabela_pesos()
    do_momento = pesos[pesos.momento == momento]
    por_grupo = (do_momento.groupby("grupo").peso.sum()
                 .sort_values(ascending=False))
    ablacao = ler_csv("avaliacao_ablacao.csv")
    ablacao = ablacao[ablacao.momento == momento].copy()
    ablacao["nome"] = ablacao.grupo.map(NOMES_GRUPO)
    ablacao["delta_pp"] = 100 * ablacao.delta
    ablacao["ic_inf_pp"] = 100 * ablacao.ic_inf
    ablacao["ic_sup_pp"] = 100 * ablacao.ic_sup
    pior = ablacao.loc[ablacao.delta.idxmin()]
    lift = ler_csv("fatores_lift.csv")
    lift = lift[lift.grupo.isin(MOMENTOS_DADOS[momento])]
    topo = lift.loc[lift.lift_treino.idxmax()]
    taxa_base = ler_json("resultados.json")["dados"]["taxa_evasao"]

    c = st.columns(3)
    c[0].metric("Grupo de maior peso no grafo", por_grupo.index[0],
                f"{pct(por_grupo.iloc[0])} do peso total",
                delta_color="off", delta_arrow="off", border=True)
    c[1].metric("Maior queda ao remover", pior.nome,
                f"{br(pior.delta_pp, 1)} p.p. de acurácia",
                delta_color="off", delta_arrow="off", border=True)
    c[2].metric("Perfil de maior risco", topo.fator,
                f"{pct(topo.taxa_treino)} evadem "
                f"({br(topo.lift_treino, 1)}× a média)",
                delta_color="off", delta_arrow="off", border=True)

    # 1. Peso no grafo ------------------------------------------------
    st.subheader("1 · Peso de cada atributo no grafo de similaridade")
    nota(
        "Na distância de Gower do G1, cada atributo entra com peso igual "
        "à sua informação mútua com o desfecho, calculada só no treino e "
        "normalizada para somar 100%. Atributo de peso maior aproxima "
        "mais quem tem o mesmo valor nele.")
    esq, dir_ = st.columns([1, 1.3])
    with esq:
        st.markdown("**Como o peso migra ao longo do curso**")
        migracao = (pesos.groupby(["momento", "grupo"]).peso.sum()
                    .reset_index())
        grafico = alt.Chart(migracao).mark_bar().encode(
            x=alt.X("peso:Q", stack="normalize", title="parcela do peso",
                    axis=alt.Axis(format="%")),
            y=alt.Y("momento:N", title=None, sort=list(MOMENTOS)),
            color=alt.Color("grupo:N", scale=escala_grupos(),
                            title=None,
                            legend=alt.Legend(orient="bottom",
                                              columns=3)),
            order=alt.Order("peso:Q", sort="descending"),
            tooltip=[alt.Tooltip("momento:N"), alt.Tooltip("grupo:N"),
                     alt.Tooltip("peso:Q", format=".1%", title="peso")],
        ).properties(height=260)
        st.altair_chart(grafico, alt="Parcela do peso de cada grupo de "
                        "atributos no G1 em T0, T1 e T2")
    with dir_:
        st.markdown(f"**Os 10 atributos de maior peso em {momento}**")
        top10 = do_momento.nlargest(10, "peso")
        grafico = alt.Chart(top10).mark_bar().encode(
            x=alt.X("peso:Q", title="peso na distância",
                    axis=alt.Axis(format="%")),
            y=alt.Y("rotulo:N", sort="-x", title=None,
                    axis=alt.Axis(labelOverlap=False, labelLimit=200)),
            color=alt.Color("grupo:N", scale=escala_grupos(),
                            legend=None),
            tooltip=[alt.Tooltip("rotulo:N", title="atributo"),
                     alt.Tooltip("grupo:N"),
                     alt.Tooltip("peso:Q", format=".1%", title="peso")],
        ).properties(height=320)
        st.altair_chart(grafico, alt=f"Os 10 atributos de maior peso "
                        f"no G1 em {momento}")
    st.caption("Fonte: `outputs/resultados.json` (`g1.<momento>.pesos`).")

    # 2. Ablação ------------------------------------------------------
    st.subheader("2 · Quanto a acurácia cai sem cada grupo")
    nota(
        "Ablação na validação cruzada do treino: o corte mínimo é "
        "refeito sem o grupo e comparado com o modelo completo. Barra "
        "à esquerda do zero = o grupo faz falta. É **essencial** quando "
        "o IC 95% inteiro fica abaixo de zero.")
    ordem_ab = ablacao.sort_values("delta").nome.tolist()
    base_ab = alt.Chart(ablacao).encode(
        y=alt.Y("nome:N", sort=ordem_ab, title=None))
    barras = base_ab.mark_bar().encode(
        x=alt.X("delta_pp:Q",
                title="variação da acurácia balanceada (p.p.)"),
        color=alt.Color("classe:N", title=None,
                        scale=alt.Scale(domain=["essencial", "neutro"],
                                        range=[COR_EVASAO, "#9aa0a6"]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("nome:N", title="grupo"),
                 alt.Tooltip("num_atributos:Q", title="atributos"),
                 alt.Tooltip("delta_pp:Q", format=".1f",
                             title="Δ (p.p.)"),
                 alt.Tooltip("ic_inf_pp:Q", format=".1f",
                             title="IC inf."),
                 alt.Tooltip("ic_sup_pp:Q", format=".1f",
                             title="IC sup."),
                 alt.Tooltip("classe:N")])
    erros = base_ab.mark_rule().encode(x="ic_inf_pp:Q", x2="ic_sup_pp:Q")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        strokeDash=[4, 3]).encode(x="x:Q")
    st.altair_chart(
        (barras + erros + zero).properties(
            height=40 * len(ablacao) + 40),
        alt=f"Variação da acurácia do corte mínimo sem cada grupo "
            f"em {momento}, com IC 95%")
    st.caption("Fonte: `outputs/avaliacao_ablacao.csv`.")

    # 3. Perfis de risco ----------------------------------------------
    st.subheader("3 · Perfis que mais elevam (ou reduzem) a evasão")
    nota(
        f"Lift = taxa de evasão do perfil ÷ taxa geral "
        f"({pct(taxa_base, 1)}). Lift 2× = evadem o dobro da média. "
        f"Calculado no treino; a última coluna repete a taxa no teste "
        f"para mostrar que o padrão se mantém. Só aparecem perfis "
        f"conhecidos em {momento}.")
    lado = st.segmented_control(
        "Mostrar", ["Risco", "Proteção"], default="Risco",
        required=True, key="lado_fatores",
        format_func={"Risco": "Elevam a evasão (lift > 1)",
                     "Proteção": "Reduzem a evasão (lift < 1)"}.get)
    if lado == "Risco":
        perfis = lift[lift.lift_treino > 1].sort_values(
            "lift_treino", ascending=False)
    else:
        perfis = lift[lift.lift_treino < 1].sort_values("lift_treino")
    perfis = perfis.assign(grupo=perfis.grupo.map(NOMES_GRUPO))
    st.dataframe(
        perfis[["fator", "grupo", "n_treino", "taxa_treino",
                "lift_treino", "taxa_teste"]],
        hide_index=True, width="stretch",
        alt=f"Perfis com lift de evasão em {momento}",
        column_config={
            "fator": st.column_config.TextColumn("perfil", width="large"),
            "grupo": "grupo",
            "n_treino": st.column_config.NumberColumn(
                "alunos (treino)", format="localized"),
            "taxa_treino": st.column_config.ProgressColumn(
                "evasão (treino)", min_value=0, max_value=1,
                format="percent"),
            "lift_treino": st.column_config.NumberColumn(
                "lift", format="%.2f×"),
            "taxa_teste": st.column_config.NumberColumn(
                "evasão (teste)", format="percent"),
        })
    st.caption("Fonte: `outputs/fatores_lift.csv`.")


def pagina_g1() -> None:
    st.title("G1 · Similaridade e corte mínimo")
    momento = escolher_momento("g1_momento")
    modelo, resultado, destaques = modelo_g1(momento)
    oficial = ler_json("resultados.json")["g1"][momento]
    avisar_divergencia(
        abs(resultado["fluxo"] - oficial["fluxo"]) < TOLERANCIA
        and resultado["num_arestas"] == oficial["num_arestas"])

    base = carregar_base()
    teste = oficial["teste"]
    assort = ler_json("resultados.json")["assortatividade"][momento]
    c = st.columns(5)
    c[0].metric("Acurácia balanceada", br(teste['acuracia_balanceada'], 3))
    c[1].metric("Recall da evasão", br(teste['recall_evasao'], 3))
    c[2].metric("Especificidade", br(teste['especificidade'], 3))
    c[3].metric("Fluxo máximo", br(resultado["fluxo"], 1),
                help=f"{resultado['num_caminhos']} caminhos aumentantes")
    c[4].metric("Assortatividade r", br(assort['r'], 3),
                help=f"p = {br(assort['p_permutacao'], 3)} "
                "(teste de permutação, k = 10)")

    st.subheader("Vizinhança de um aluno de teste")
    como_ler(
        f"um aluno. O grafo tem {br(resultado['num_vertices'], 0)}; aqui "
        "aparecem o aluno escolhido e os vizinhos dele.",
        f"liga cada aluno aos k = {modelo.k} mais parecidos pela "
        "distância de Gower (kNN simétrico).",
        ["**Espessura** = similaridade 1 − d, que é a capacidade da "
         "aresta no corte mínimo.",
         "**Círculo** = treino (rótulo conhecido). **Losango** = teste: "
         "cor = previsão, borda = desfecho real.",
         "**Vermelho** = evasão, **azul** = formatura. **Tracejado "
         "vermelho** = aresta cortada pelo corte mínimo.",
         "**Clique** num vizinho ou numa aresta para ver *por que* os "
         "dois alunos estão ligados. Passe o mouse para destacar os "
         "vizinhos de um vértice."])

    previsoes = resultado["previsoes"]
    idx_teste = modelo.idx_sem_rotulo_
    real = dict(zip(idx_teste, map(bool, base.y(idx_teste))))
    categorias = {
        "Todos": lambda i: True,
        "Acerto · evasão (VP)": lambda i: previsoes[i] and real[i],
        "Acerto · formatura (VN)":
            lambda i: not previsoes[i] and not real[i],
        "Falso positivo": lambda i: previsoes[i] and not real[i],
        "Falso negativo": lambda i: not previsoes[i] and real[i],
    }
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    categoria = c1.segmented_control(
        "Filtrar", list(categorias), default="Todos", required=True,
        key="g1_filtro")
    candidatos = [i for i in idx_teste if categorias[categoria](i)]
    aluno = c2.selectbox(f"Aluno ({len(candidatos)})", candidatos,
                         key="g1_aluno")
    if aluno is None:
        st.info("Nenhum aluno nesta categoria.")
        return

    lado_evasao = set(destaques["vertices"])
    vizinhos = modelo.grafo.vizinhos(aluno)
    linhas = []
    for v, cap in sorted(vizinhos.items(), key=lambda x: -x[1]):
        linhas.append({"aluno": v, "papel": papel_aluno(
                           v, modelo.rotulos_, previsoes, real),
                       "lado do corte": "S (evasão)" if v in lado_evasao
                       else "T (formatura)",
                       "distância de Gower": 1 - cap,
                       "capacidade": cap})
    tabela = pd.DataFrame(linhas)
    para_s = tabela.loc[tabela["lado do corte"].str.startswith("S"),
                        "capacidade"].sum()
    para_t = tabela.capacidade.sum() - para_s
    acertou = previsoes[aluno] == real[aluno]

    c = st.columns(5)
    c[0].metric("Previsão", "evasão" if previsoes[aluno] else "formatura",
                border=True)
    c[1].metric("Desfecho real", "evasão" if real[aluno] else "formatura",
                "acertou" if acertou else "errou",
                delta_color="normal" if acertou else "inverse",
                delta_arrow="off", border=True)
    c[2].metric("Vizinhos no kNN", len(vizinhos), border=True)
    c[3].metric("Capacidade → lado S", br(para_s, 2), border=True,
                help="Soma das capacidades das arestas até vizinhos que "
                "ficaram do lado da evasão.")
    c[4].metric("Capacidade → lado T", br(para_t, 2), border=True,
                help="Soma das capacidades das arestas até vizinhos que "
                "ficaram do lado da formatura.")

    opcao, ordem, pares = GrafoECharts(modo_escuro()).ego_g1(
        modelo.grafo, aluno, modelo.rotulos_, previsoes, lado_evasao, real)
    col_g, col_i = st.columns([1.35, 1])
    with col_g:
        evento = st.echarts_chart(
            opcao, on_select="rerun", key=f"ego_{momento}_{aluno}",
            height=px_desenho(560),
            alt=f"Aluno {aluno} e seus {len(vizinhos)} vizinhos no grafo "
            "kNN de Gower")
    par = par_selecionado(evento, ordem, pares, aluno)
    explicador = explicador_g1(momento)
    with col_i:
        if par is None:
            painel_aluno(explicador, aluno, papel_aluno(
                aluno, modelo.rotulos_, previsoes, real))
        else:
            painel_aresta(explicador, *par, modelo.rotulos_, previsoes,
                          real)

    if not apresentacao():
        with st.expander("Todos os vizinhos do aluno"):
            st.dataframe(tabela, hide_index=True, width="stretch",
                         alt=f"Vizinhos do aluno {aluno} no kNN",
                         column_config={
                             "distância de Gower":
                                 st.column_config.NumberColumn(format="%.3f"),
                             "capacidade":
                                 st.column_config.NumberColumn(format="%.3f")})
    nota(
        f"Grafo completo: {br(resultado['num_vertices'], 0)} vértices e "
        f"{br(resultado['num_arestas'], 0)} arestas (k = {modelo.k}, "
        f"λ = {br(modelo.lam, 1)}); "
        f"{br(len(destaques['arestas']), 0)} arestas cortadas. O "
        "desfecho real só aparece aqui porque o teste já foi avaliado "
        "uma vez pelo pipeline.")


def papel_aluno(v: int, rotulos: dict, previsoes: dict,
                real: dict) -> str:
    if v in rotulos:
        return "treino: " + ("evadiu" if rotulos[v] else "formou")
    texto = "teste: previsto " + ("evasão" if previsoes[v]
                                  else "formatura")
    return texto + " (real: " + ("evasão" if real[v]
                                 else "formatura") + ")"


def par_selecionado(evento, ordem: list[int],
                    pares: list[tuple[int, int]],
                    aluno: int) -> tuple[int, int] | None:
    """Par (u, v) clicado: um vizinho vale a aresta até o aluno central."""
    for grupo in evento.selection.selected:
        indices = grupo.get("data_indices") or []
        if not indices:
            continue
        if grupo.get("data_type") == "edge":
            return pares[indices[0]]
        if grupo.get("data_type") == "node":
            v = ordem[indices[0]]
            return None if v == aluno else (aluno, v)
    return None


def painel_aluno(explicador: ExplicadorGower, aluno: int,
                 papel: str) -> None:
    st.markdown(f"#### Aluno {aluno}")
    st.caption(papel)
    st.markdown("**Atributos de maior peso no grafo**")
    st.dataframe(
        explicador.perfil(aluno), hide_index=True, width="stretch",
        alt=f"Atributos de maior peso do aluno {aluno}",
        column_config={
            "atributo": "atributo", "valor": "valor do aluno",
            "peso": st.column_config.ProgressColumn(
                "peso na distância", min_value=0,
                max_value=max(explicador.pesos.values()),
                format="percent")})
    st.info("Clique num vizinho ou numa aresta do desenho para ver por "
            "que os dois alunos estão ligados.",
            icon=":material/touch_app:")


def painel_aresta(explicador: ExplicadorGower, u: int, v: int,
                  rotulos: dict, previsoes: dict, real: dict) -> None:
    tabela = explicador.contribuicoes(u, v)
    similaridade = float(tabela.semelhanca.sum())
    st.markdown(f"#### Por que os alunos {u} e {v} estão ligados")
    st.caption(f"{u} · {papel_aluno(u, rotulos, previsoes, real)}  \n"
               f"{v} · {papel_aluno(v, rotulos, previsoes, real)}")
    st.metric("Similaridade 1 − d (capacidade da aresta)",
              br(similaridade, 3), border=True)

    topo = tabela.head(TOPO_ATRIBUTOS)
    partes = topo.melt(
        id_vars=["rotulo", "peso", "valor_i", "valor_j"],
        value_vars=["semelhanca", "distancia"], var_name="parte",
        value_name="valor")
    partes["parte"] = partes.parte.map(
        {"semelhanca": "em comum", "distancia": "diferença"})
    grafico = alt.Chart(partes).mark_bar().encode(
        x=alt.X("valor:Q", stack="zero", title="peso na distância",
                axis=alt.Axis(format="%")),
        y=alt.Y("rotulo:N", title=None,
                sort=alt.EncodingSortField("peso", order="descending")),
        color=alt.Color("parte:N", title=None,
                        scale=alt.Scale(domain=["em comum", "diferença"],
                                        range=[COR_SEGURA, "#d5dbe1"]),
                        legend=alt.Legend(orient="bottom")),
        order=alt.Order("parte:N", sort="descending"),
        tooltip=[alt.Tooltip("rotulo:N", title="atributo"),
                 alt.Tooltip("valor_i:N", title=f"aluno {u}"),
                 alt.Tooltip("valor_j:N", title=f"aluno {v}"),
                 alt.Tooltip("parte:N"),
                 alt.Tooltip("valor:Q", format=".1%")],
    ).properties(height=28 * len(topo) + 30)
    st.altair_chart(grafico, alt=f"Contribuição de cada atributo para a "
                    f"similaridade entre os alunos {u} e {v}")
    resto = float(tabela.semelhanca.iloc[TOPO_ATRIBUTOS:].sum())
    nota(
        "Barra inteira = peso do atributo. Parte verde = quanto desse "
        "peso os dois ganharam por serem parecidos nele. A soma das "
        "partes verdes de todos os atributos é a similaridade "
        f"({br(similaridade, 3)}); os demais "
        f"{len(tabela) - len(topo)} atributos somam {br(resto, 3)}.")
    if not apresentacao():
        with st.expander("Valores dos dois alunos"):
            st.dataframe(
                topo[["rotulo", "valor_i", "valor_j", "peso",
                      "semelhanca"]],
                hide_index=True, width="stretch",
                alt=f"Valores dos alunos {u} e {v} nos atributos de "
                "maior peso",
                column_config={
                    "rotulo": "atributo",
                    "valor_i": f"aluno {u}", "valor_j": f"aluno {v}",
                    "peso": st.column_config.NumberColumn(
                        "peso", format="percent"),
                    "semelhanca": st.column_config.NumberColumn(
                        "em comum", format="percent")})


def pagina_g2() -> None:
    st.title("G2 · Árvore de dependências")
    momento = escolher_momento("g2_momento")
    modelo, resultado = modelo_g2(momento)
    oficial = ler_json("resultados.json")["g2"][momento]
    avisar_divergencia(
        abs(resultado["peso_total"] - oficial["peso_total"]) < TOLERANCIA)

    c = st.columns(4)
    c[0].metric("Vértices", oficial["num_vertices"])
    c[1].metric("Peso total (bits)", br(oficial['peso_total'], 2))
    c[2].metric("Estabilidade média", pct(oficial['estabilidade_media'], 0),
                help="Fração de 100 bootstraps do treino em que cada "
                "aresta reaparece")
    c[3].metric("Sobreposição treino × teste",
                pct(oficial['sobreposicao_treino_teste'], 0))

    como_ler(
        "um atributo do aluno; o **Desfecho** (evadiu ou não) fica no "
        "centro.",
        "dependência entre dois atributos, aceita pelo Kruskal na "
        "árvore geradora de peso máximo (peso = informação mútua).",
        ["**Anéis** = distância até o Desfecho na árvore; no 1º anel "
         "estão os vizinhos diretos.",
         "**Régua**: desça para o Kruskal aceitar ligações cada vez mais "
         "fracas e veja os blocos se formarem e se ligarem ao Desfecho "
         "(▶ anima). **Vermelho claro** = já ligado ao Desfecho; outras "
         "cores = blocos ainda soltos (componentes do union-find); "
         "cinza = sozinho.",
         "**Espessura** = informação mútua; **opacidade** = estabilidade "
         "em 100 bootstraps.",
         "**Clique num atributo** para ver o caminho até o Desfecho "
         "(laranja) e o **gargalo**, a ligação mais fraca. Tracejado = "
         "trecho que a régua ainda não aceitou. Arraste os vértices se "
         "os rótulos se sobrepuserem."])

    total = len(resultado["arestas"])
    chave, anima = f"g2_regua_{momento}", f"g2_anima_{momento}"
    st.session_state.setdefault(chave, total)
    animando = st.session_state.get(anima, False)
    if st.button(":material/pause: Parar" if animando
                 else ":material/play_arrow: Animar o Kruskal",
                 key=f"g2_botao_{momento}"):
        st.session_state[anima] = not animando
        if not animando:
            st.session_state[chave] = 1
        st.rerun()
    corpo = REGUA_ANIMADA if st.session_state.get(anima) else REGUA
    corpo(momento)

    if not apresentacao():
        with st.expander("Todas as arestas da árvore"):
            st.dataframe(ler_csv(f"g2_arvore_{momento}.csv"),
                         hide_index=True, width="stretch")


def regua_e_caminho(momento: str) -> None:
    """Régua de IM (estado do Kruskal), desenho e caminho clicado.

    Roda como fragmento: mexer na régua ou clicar num atributo refaz só
    esta parte; animando, avança uma aresta por vez sozinha.
    """
    _, resultado = modelo_g2(momento)
    demo = demo_kruskal(momento)
    explorador = explorador_g2(momento)
    total = len(resultado["arestas"])
    chave, anima = f"g2_regua_{momento}", f"g2_anima_{momento}"
    if st.session_state.get(anima):
        if st.session_state[chave] >= total:
            st.session_state[anima] = False
            st.rerun(scope="app")
        st.session_state[chave] += 1

    limiares = [p["peso"] for p in demo["passos"] if p["aceita"]]
    k = st.select_slider(
        "Régua de informação mútua · dependência mínima já aceita",
        options=list(range(1, total + 1)), key=chave,
        format_func=lambda i: f"{br(limiares[i - 1], 3)} bits · "
        f"{i}/{total}")
    estado = explorador.regua(demo["passos"], k)
    componentes = estado["componentes"]
    raiz = componentes[DESFECHO]
    ligados = sum(r == raiz for r in componentes.values()) - 1

    c = st.columns(4)
    c[0].metric("Arestas aceitas", f"{k} de {total}", border=True)
    c[1].metric("Rejeitadas (fechariam ciclo)", estado["rejeitadas"],
                border=True)
    c[2].metric("Blocos soltos", len(set(componentes.values())),
                border=True, help="Componentes do union-find nesse "
                "ponto; 1 = árvore completa.")
    c[3].metric("Atributos ligados ao Desfecho",
                f"{ligados} de {len(componentes) - 1}", border=True)

    csv = ler_csv(f"g2_arvore_{momento}.csv")
    estabilidade = {tuple(sorted((r.u, r.v))): r.estabilidade
                    for r in csv.itertuples()}
    chave_grafico = f"g2_arvore_{momento}"
    selecionado = atributo_selecionado(chave_grafico)
    caminho = (explorador.caminho(selecionado)
               if selecionado not in (None, DESFECHO) else None)
    gargalo = explorador.gargalo(caminho) if caminho else None

    opcao, vertices = GrafoECharts(modo_escuro()).arvore_g2(
        explorador.layout_radial(), DESFECHO, estado["aceitas"],
        componentes, estabilidade, caminho, gargalo)
    st.session_state[f"{chave_grafico}_vertices"] = vertices
    col_g, col_p = st.columns([1.6, 1])
    with col_g:
        st.echarts_chart(
            opcao, on_select="rerun", key=chave_grafico,
            height=px_desenho(640),
            alt=f"Árvore de dependências em {momento} com {k} arestas "
            "aceitas pelo Kruskal")
    with col_p:
        if caminho:
            painel_caminho(selecionado, caminho, gargalo, explorador,
                           resultado["im_desfecho"])
        else:
            st.markdown("**Vizinhos do Desfecho na árvore completa**")
            viz = pd.DataFrame(resultado["vizinhos_desfecho"],
                               columns=["atributo", "IM (bits)"])
            viz["atributo"] = viz.atributo.map(lambda a: ROTULOS.get(a, a))
            st.dataframe(viz, hide_index=True, width="stretch",
                         alt="Vizinhos do Desfecho na árvore",
                         column_config={"IM (bits)":
                                        st.column_config.NumberColumn(
                                            format="%.4f")})
            st.info("Clique num atributo do desenho para ver o caminho "
                    "dele até o Desfecho e o gargalo.",
                    icon=":material/touch_app:")


REGUA = st.fragment(regua_e_caminho)
REGUA_ANIMADA = st.fragment(regua_e_caminho, run_every=0.7)


def atributo_selecionado(chave: str) -> str | None:
    """Atributo clicado no desenho (estado do gráfico no session_state)."""
    evento = st.session_state.get(chave)
    vertices = st.session_state.get(f"{chave}_vertices", [])
    if not evento:
        return None
    for grupo in evento.get("selection", {}).get("selected", []):
        indices = grupo.get("data_indices") or []
        if grupo.get("data_type") == "node" and indices:
            if indices[0] < len(vertices):
                return vertices[indices[0]]
    return None


def painel_caminho(origem: str, caminho: list, gargalo: tuple,
                   explorador: ExploradorArvore,
                   im_desfecho: dict) -> None:
    nome = ROTULOS.get(origem, origem)
    st.markdown(f"#### {nome} → Desfecho")
    passos = explorador.profundidade(origem)
    c1, c2 = st.columns(2)
    c1.metric("Gargalo", f"{br(gargalo[2], 3)} bits", border=True,
              help=f"{ROTULOS.get(gargalo[0], gargalo[0])} — "
              f"{ROTULOS.get(gargalo[1], gargalo[1])}")
    c2.metric("Distância", f"{passos} aresta" + ("s" if passos > 1
                                                  else ""), border=True)
    st.dataframe(pd.DataFrame([
        {"de": ROTULOS.get(u, u), "para": "Desfecho" if v == DESFECHO
         else ROTULOS.get(v, v), "IM (bits)": p,
         "gargalo": "◀" if (u, v, p) == gargalo else ""}
        for u, v, p in caminho]), hide_index=True, width="stretch",
        alt=f"Caminho de {nome} até o Desfecho",
        column_config={"IM (bits)": st.column_config.NumberColumn(
            format="%.4f")})
    direta = im_desfecho.get(origem)
    if passos > 1 and direta is not None:
        nota(f"IM direta de {nome} com o Desfecho: {br(direta, 3)} bits, "
             f"no máximo igual ao gargalo. Na árvore geradora máxima, este "
             "caminho tem o maior gargalo possível entre todos os "
             "caminhos do grafo completo: a ligação com a evasão passa "
             "pelos atributos intermediários.")
    else:
        nota("Vizinho direto do Desfecho: a dependência com a evasão "
             "não passa por nenhum outro atributo.")


def pagina_g3() -> None:
    st.title("G3 · DAG de trajetórias")
    momento = escolher_momento("g3_momento", padrao="T2")
    conhecidas = CAMADAS_POR_MOMENTO[momento]
    modelo, resultado = modelo_g3(momento)
    oficial = ler_json("resultados.json")["g3"][momento]
    avisar_divergencia(
        resultado["trajetoria_risco"] == oficial["trajetoria_risco"]
        and resultado["trajetoria_segura"] == oficial["trajetoria_segura"])
    risco = resultado["trajetoria_risco"]
    segura = resultado["trajetoria_segura"]
    virada = resultado["ponto_de_virada"]
    vistas = list(CAMADAS_SIMULADOR.values())[:conhecidas]

    teste = oficial["teste"]
    c = st.columns(4)
    c[0].metric("Camadas observadas",
                f"{conhecidas} de {len(CAMADAS_SIMULADOR)}",
                help=f"{', '.join(vistas)}. O DAG é o mesmo nos três "
                "momentos; muda quanto do caminho do aluno já se conhece "
                "para prever.")
    c[1].metric("Acurácia balanceada", br(teste["acuracia_balanceada"], 3))
    c[2].metric("Recall da evasão", br(teste["recall_evasao"], 3))
    c[3].metric("Especificidade", br(teste["especificidade"], 3))

    c = st.columns(4)
    ver_risco = c[0].toggle("Trajetória de maior risco", value=True)
    ver_segura = c[1].toggle("Trajetória mais segura", value=True)
    ver_virada = c[2].toggle("Ponto de virada", value=True)
    ver_simulado = c[3].toggle("Caminho simulado", value=True)

    st.markdown(f"**Simule um aluno em {momento}:** escolha um estado em "
                "cada camada já observada.")
    caminho, linhas = simular_caminho(modelo, conhecidas)

    leitura = [
        "**Tamanho** = nº de alunos no estado. **Cor** = taxa de "
        "evasão: azul = formam, vermelho = evadem.",
        "**Número antes do nome** = posição na ordem topológica de "
        "Kahn.",
        "**Vermelho/verde** = trajetória de maior risco/mais segura "
        "(Bellman-Ford). **Laranja** = caminho simulado. O número na "
        "aresta é o nº de alunos.",
        "**Tracejado** = ponto de virada. Arestas **claras** = abaixo "
        "do suporte mínimo."]
    if conhecidas < len(CAMADAS_SIMULADOR):
        leitura.append(f"**Esmaecido** = camadas que ainda não se "
                       f"conhecem em {momento}; o preditor não as usa.")
    como_ler(
        "um estado da trajetória (ex.: \"S1: aprovou < 50%\"), em "
        "camadas: ingresso → financeiro → S1 → S2 → desfecho.",
        "alunos de treino que passam de um estado ao seguinte; só "
        "vai da esquerda para a direita, por isso não há ciclo.",
        leitura)
    desenhar(montador().dag_g3(
        modelo.grafo, resultado["ordem_topologica"],
        modelo.contagem_vertice_, modelo.evasoes_vertice_,
        modelo.min_suporte,
        risco=risco["caminho"] if ver_risco else None,
        segura=segura["caminho"] if ver_segura else None,
        virada=virada["arco"] if ver_virada else None,
        caminho_usuario=caminho if ver_simulado else None,
        conhecidas=conhecidas), altura=640)

    resultado_simulacao(modelo, linhas, resultado["limiar"], momento)
    c = st.columns(3)
    c[0].metric("Trajetória de maior risco",
                f"{risco['evasoes']} de {risco['n']} evadiram",
                border=True)
    c[1].metric("Trajetória mais segura",
                f"{segura['evasoes']} de {segura['n']} evadiram",
                border=True)
    c[2].metric("Ponto de virada",
                f"{pct(virada['taxa_antes'])} → "
                f"{pct(virada['taxa_depois'])}", border=True,
                help=f"{virada['arco'][0]} → {virada['arco'][1]} "
                f"({virada['n']} alunos)")
    nota("As trajetórias e o ponto de virada usam o caminho "
               "completo e são iguais nos três momentos. Fonte das "
               "métricas: `outputs/resultados.json`.")
    if not apresentacao():
        with st.expander("Trajetórias e ordem topológica de Kahn"):
            st.markdown(
                f"- **Risco:** {' → '.join(risco['caminho'])}\n"
                f"- **Segura:** {' → '.join(segura['caminho'])}\n"
                f"- **Virada:** {virada['arco'][0]} → {virada['arco'][1]}\n"
                f"- **Ordem de Kahn:** "
                f"{' → '.join(resultado['ordem_topologica'])}")


def simular_caminho(modelo: DagTrajetorias,
                    conhecidas: int) -> tuple[list[str], list]:
    """Seletores das camadas observadas; devolve o caminho e o resumo."""
    estados = estados_treino()
    caminho = [INICIO]
    filtro = np.ones(len(estados), dtype=bool)
    linhas = []
    colunas = st.columns(len(CAMADAS_SIMULADOR))
    for i, (col, camada) in enumerate(zip(colunas, CAMADAS_SIMULADOR)):
        nome = CAMADAS_SIMULADOR[camada]
        if i >= conhecidas:
            col.text_input(nome, "ainda não observado", disabled=True,
                           key=f"sim_futuro_{camada}")
            continue
        sucessores = sorted(
            (v for v in modelo.grafo.vizinhos(caminho[-1])
             if v.startswith(camada)),
            key=lambda v: -modelo.contagem_arco_[(caminho[-1], v)])
        if not sucessores:
            break
        escolha = col.selectbox(
            nome, sucessores, format_func=lambda v: v.split(":", 1)[1],
            key=f"sim_{camada}")
        filtro &= (estados[camada] == escolha).to_numpy()
        n = int(filtro.sum())
        evasoes = int(estados.evadiu[filtro].sum())
        linhas.append({
            "camada": nome,
            "estado": escolha.split(":", 1)[1],
            "alunos no arco": modelo.contagem_arco_[(caminho[-1], escolha)],
            "evasão no arco": modelo.taxa_arco(caminho[-1], escolha),
            "alunos com este caminho": n,
            "evasão no caminho": evasoes / n if n else np.nan,
        })
        caminho.append(escolha)
    return caminho, linhas


def resultado_simulacao(modelo: DagTrajetorias, linhas: list,
                        limiar: float, momento: str) -> None:
    """Risco e previsão do G3 para o trecho simulado.

    Mesma regra do preditor: taxa de evasão, no treino, do maior trecho
    conhecido com pelo menos ``min_suporte`` alunos; sem nenhum, a taxa
    do treino. Prevê evasão quando o risco passa do limiar.
    """
    if not linhas:
        return
    com_suporte = [r for r in linhas
                   if r["alunos com este caminho"] >= modelo.min_suporte]
    risco = (com_suporte[-1]["evasão no caminho"] if com_suporte
             else modelo.taxa_treino_)
    final = linhas[-1]
    c = st.columns(3)
    c[0].metric(f"Alunos do treino com este trecho ({momento})",
                final["alunos com este caminho"], border=True)
    c[1].metric("Risco do trecho", pct(risco, 0), border=True,
                help="Taxa de evasão, no treino, de quem fez este trecho.")
    c[2].metric("Previsão do G3", "evasão" if risco > limiar
                else "formatura", border=True,
                help=f"Evasão quando o risco passa de {pct(limiar, 1)} "
                "(taxa de evasão do treino).")
    if len(com_suporte) < len(linhas):
        usado = (f"até {com_suporte[-1]['camada']}" if com_suporte
                 else "a taxa geral do treino")
        st.warning(f"Menos de {modelo.min_suporte} alunos com este trecho: "
                   f"o preditor usa {usado}.")
    if not apresentacao():
        with st.expander("Caminho simulado, passo a passo"):
            barra = st.column_config.ProgressColumn(
                format="percent", min_value=0, max_value=1)
            st.dataframe(pd.DataFrame(linhas), hide_index=True,
                         width="stretch",
                         column_config={"evasão no arco": barra,
                                        "evasão no caminho": barra})


# ----------------------------------------------------------------------
# Navegação
# ----------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Evasão em grafos", page_icon="🎓",
                       layout="wide")
    paginas = [
        st.Page(pagina_resumo, title="Resumo",
                icon=":material/dashboard:", default=True),
        st.Page(pagina_g1, title="G1 · Corte mínimo",
                icon=":material/hub:", url_path="g1"),
        st.Page(pagina_g2, title="G2 · Árvore de dependências",
                icon=":material/account_tree:", url_path="g2"),
        st.Page(pagina_g3, title="G3 · DAG de trajetórias",
                icon=":material/route:", url_path="g3"),
    ]
    pagina = st.navigation(paginas)
    st.sidebar.toggle(
        ":material/slideshow: Modo apresentação", key="apresentacao",
        help="Esconde os quadros de leitura, os detalhes e as notas e "
        "aumenta os grafos. Para projetor ou vídeo.")
    pagina.run()


if __name__ == "__main__":
    main()
