"""Interface gráfica do trabalho (Streamlit).

Rodar a partir da raiz do repositório::

    source .venv/bin/activate
    streamlit run src/app.py

Este módulo só cuida da tela: os grafos vêm das classes do projeto
(``GrafoSimilaridade``, ``ArvoreDependencias``, ``DagTrajetorias``), o
desenho vem de ``visual.MontadorVisual`` e as animações de
``passo_a_passo``. Os números oficiais (métricas, IC, McNemar) são
lidos de ``outputs/``; nada é digitado à mão.

Sem vazamento: a visão oficial usa os hiperparâmetros gravados em
``outputs/hiperparametros.json``. Só a página de exploração aceita
outros k e λ, e ela roda numa dobra de validação **do treino**.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from arvore_dependencias import DESFECHO, ArvoreDependencias
from avaliacao import Avaliador
from corte_minimo import FONTE, SUMIDOURO
from dados import BaseDeDados
from dag_trajetorias import INICIO, DagTrajetorias
from grafo_similaridade import GrafoSimilaridade
from passo_a_passo import DemoEdmondsKarp, DemoKruskal
from visual import (COR_DESTAQUE, COR_EVASAO, COR_FORMATURA, COR_SEGURA,
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


@st.cache_resource(show_spinner="Montando a rede de demonstração...")
def demo_edmonds_karp(momento: str, k: int, lam: float) -> dict:
    pesos = modelo_g1(momento)[0].pesos_
    return DemoEdmondsKarp(momento, k=k, lam=lam).executar(
        carregar_base(), pesos)


@st.cache_resource
def demo_kruskal(momento: str) -> dict:
    modelo, resultado = modelo_g2(momento)
    return DemoKruskal().executar(modelo, resultado["arestas"])


@st.cache_resource
def particoes_treino() -> list[tuple[list[int], list[int]]]:
    return Avaliador().particoes(carregar_base())


@st.cache_data(show_spinner="Rodando o corte mínimo na dobra de "
                            "validação...")
def explorar_g1(momento: str, k: int, lam: float, dobra: int) -> dict:
    base = carregar_base()
    treino, validacao = particoes_treino()[dobra]
    modelo = GrafoSimilaridade(momento, k, lam=lam)
    modelo.construir(base, treino, validacao)
    prev = modelo.executar()["previsoes"]
    return Avaliador.metricas(base.y(validacao),
                              np.array([prev[i] for i in validacao]))


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
def desenhar(desenho: tuple, altura: int = 560) -> None:
    vertices, arestas, opcoes = desenho
    html = MontadorVisual(altura).html(vertices, arestas, opcoes)
    st.iframe(html, height=altura + 12)


def escolher_momento(chave: str, padrao: str = "T1") -> str:
    return st.segmented_control(
        "Momento", MOMENTOS, default=padrao, required=True,
        format_func=NOME_MOMENTO.get, key=chave)


def selo(confere: bool, fonte: str) -> None:
    """Confirma que o grafo refeito na tela bate com ``outputs/``."""
    if confere:
        st.caption(f":green[✔ confere com `outputs/{fonte}`]")
    else:
        st.warning(f"O resultado na tela difere de `outputs/{fonte}`. "
                   "Rode `python src/main.py` de novo.")


def garantia(texto: str) -> None:
    st.info(texto, icon=":material/verified:")


def legenda(*itens: tuple[str, str]) -> None:
    partes = [f"<span style='color:{cor}'>●</span> {texto}"
              for cor, texto in itens]
    st.markdown(" &nbsp; ".join(partes), unsafe_allow_html=True)


def navegar_passos(chave: str, total: int) -> int:
    """Controle de passos: botões ◀ ▶ e barra deslizante."""
    if chave not in st.session_state:
        st.session_state[chave] = 0
    st.session_state[chave] = min(st.session_state[chave], total - 1)

    def mover(delta: int) -> None:
        st.session_state[chave] = int(
            np.clip(st.session_state[chave] + delta, 0, total - 1))

    c1, c2, c3, c4 = st.columns([1, 1, 1, 8])
    c1.button("⏮", key=f"{chave}_ini", on_click=mover, args=(-total,),
              width="stretch")
    c2.button("◀", key=f"{chave}_ant", on_click=mover, args=(-1,),
              width="stretch")
    c3.button("▶", key=f"{chave}_prox", on_click=mover, args=(1,),
              width="stretch")
    with c4:
        st.slider("Passo", 0, max(total - 1, 1), key=chave,
                  label_visibility="collapsed",
                  disabled=total <= 1)
    return st.session_state[chave]


def nome_aluno(v) -> str:
    return str(v)


def formatar_caminho(caminho: list[tuple]) -> str:
    if not caminho:
        return "—"
    return " → ".join([nome_aluno(caminho[0][0]),
                       *(nome_aluno(v) for _, v in caminho)])


# ----------------------------------------------------------------------
# Páginas
# ----------------------------------------------------------------------
def pagina_visao_geral() -> None:
    st.title("Evasão estudantil como grafos")
    st.markdown(
        "Quais características, em que combinação e em que momento "
        "distinguem quem evade de quem se forma? Três grafos, aprendidos "
        "com 80% dos alunos e avaliados uma única vez nos outros 20%.")

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


def pagina_g1() -> None:
    st.title("G1 · Similaridade e corte mínimo")
    st.markdown(
        "Cada aluno é ligado aos k mais parecidos pela distância de "
        "Gower; a capacidade da aresta é 1 − d. A fonte S liga-se aos "
        "evadidos do treino e o sumidouro T aos formados. O **corte "
        "mínimo** (Edmonds–Karp) separa os dois lados: cada aluno de "
        "teste é previsto pelo lado em que cai.")
    momento = escolher_momento("g1_momento")
    modelo, resultado, destaques = modelo_g1(momento)
    oficial = ler_json("resultados.json")["g1"][momento]
    selo(abs(resultado["fluxo"] - oficial["fluxo"]) < TOLERANCIA
         and resultado["num_arestas"] == oficial["num_arestas"],
         "resultados.json")

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
    c1, c2 = st.columns([2, 1])
    categoria = c1.radio("Filtrar", list(categorias), horizontal=True)
    candidatos = [i for i in idx_teste if categorias[categoria](i)]
    aluno = c2.selectbox(f"Aluno ({len(candidatos)})", candidatos)
    if aluno is None:
        st.info("Nenhum aluno nesta categoria.")
        return

    lado_evasao = set(destaques["vertices"])
    vizinhos = modelo.grafo.vizinhos(aluno)
    linhas = []
    for v, cap in sorted(vizinhos.items(), key=lambda x: -x[1]):
        if v in modelo.rotulos_:
            papel = "treino: " + ("evadiu" if modelo.rotulos_[v]
                                  else "formou")
        else:
            papel = "teste (sem rótulo)"
        linhas.append({"aluno": v, "papel": papel,
                       "lado do corte": "S (evasão)" if v in lado_evasao
                       else "T (formatura)",
                       "distância de Gower": 1 - cap,
                       "capacidade": cap})
    tabela = pd.DataFrame(linhas)
    para_s = tabela.loc[tabela["lado do corte"].str.startswith("S"),
                        "capacidade"].sum()
    para_t = tabela.capacidade.sum() - para_s

    col_g, col_i = st.columns([3, 1.3])
    with col_g:
        desenhar(MontadorVisual().ego_g1(
            modelo.grafo, aluno, modelo.rotulos_, previsoes,
            lado_evasao, real), altura=520)
        legenda((COR_EVASAO, "treino evadiu"),
                (COR_FORMATURA, "treino formou"),
                ("#8a96a3", "◆ teste: cor = previsão, borda = real"),
                (COR_EVASAO, "- - - aresta cortada"))
    with col_i:
        st.metric("Previsão", "evasão" if previsoes[aluno]
                  else "formatura")
        st.metric("Desfecho real", "evasão" if real[aluno]
                  else "formatura",
                  delta="acertou" if previsoes[aluno] == real[aluno]
                  else "errou",
                  delta_color="normal" if previsoes[aluno] == real[aluno]
                  else "inverse")
        st.metric("Vizinhos no kNN", len(vizinhos))
        st.metric("Capacidade para o lado S", br(para_s, 2))
        st.metric("Capacidade para o lado T", br(para_t, 2))
    with st.expander("Vizinhos do aluno"):
        st.dataframe(tabela, hide_index=True, width="stretch",
                     column_config={
                         "distância de Gower":
                             st.column_config.NumberColumn(format="%.3f"),
                         "capacidade":
                             st.column_config.NumberColumn(format="%.3f")})
    st.caption(
        f"Grafo completo: {br(resultado['num_vertices'], 0)} vértices e "
        f"{br(resultado['num_arestas'], 0)} arestas (k = {modelo.k}, "
        f"λ = {br(modelo.lam, 1)}); "
        f"{br(len(destaques['arestas']), 0)} arestas cortadas. O "
        "desfecho real só aparece aqui porque o teste já foi avaliado "
        "uma vez pelo pipeline.")


def pagina_exploracao() -> None:
    st.title("G1 · Exploração de k e λ")
    st.warning(
        "Exploração **só no treino**: o grafo usa 4 dobras do treino e "
        "classifica a 5ª (validação). O teste não é tocado. Estes "
        "números não são os resultados oficiais.",
        icon=":material/science:")
    momento = escolher_momento("exp_momento")
    k_oficial, lam_oficial = hiperparametros_g1(momento)
    c1, c2, c3 = st.columns(3)
    k = c1.slider("k (vizinhos)", 5, 60, k_oficial, step=5)
    lam = c2.slider("λ (viés para a evasão)", 0.0, 2.0, lam_oficial,
                    step=0.25)
    dobra = c3.selectbox("Dobra de validação", range(5),
                         format_func=lambda d: f"{d + 1} de 5")
    m = explorar_g1(momento, k, lam, dobra)

    c = st.columns(4)
    c[0].metric("Acurácia balanceada", br(m['acuracia_balanceada'], 3))
    c[1].metric("Recall da evasão", br(m['recall_evasao'], 3))
    c[2].metric("Especificidade", br(m['especificidade'], 3))
    c[3].metric("VP / FP / FN / VN",
                f"{m['vp']} / {m['fp']} / {m['fn']} / {m['vn']}")

    historico = st.session_state.setdefault("exp_historico", {})
    historico[(momento, k, lam, dobra)] = m["acuracia_balanceada"]
    tabela = pd.DataFrame(
        [{"momento": a, "k": b, "λ": c_, "dobra": d + 1,
          "acurácia balanceada": v}
         for (a, b, c_, d), v in historico.items()])
    cv = ler_csv("avaliacao_cv.csv")
    ref = cv[(cv.momento == momento) & (cv.metodo == "corte_minimo")]
    if not ref.empty:
        st.caption(
            f"Referência oficial (CV de 5 dobras com k = {k_oficial}, "
            f"λ = {lam_oficial}): {br(ref.media.iloc[0], 3)} ± "
            f"{br(ref.dp.iloc[0], 3)}. Fonte: `outputs/avaliacao_cv.csv`.")
    st.subheader("Configurações testadas nesta sessão")
    st.dataframe(tabela.sort_values(["momento", "dobra", "λ", "k"]),
                 hide_index=True, width="stretch",
                 column_config={"acurácia balanceada":
                                st.column_config.ProgressColumn(
                                    format="%.3f", min_value=0.5,
                                    max_value=1.0)})


def pagina_edmonds_karp() -> None:
    st.title("Edmonds–Karp passo a passo")
    st.markdown(
        "Uma rede pequena com alunos **do treino** (sorteio com semente "
        "42): os rotulados ficam ligados a S (evadiu) ou a T (formou) "
        "com capacidade ∞, e os alunos “sem rótulo” recebem o viés "
        "λ·(1 − q) de S e λ·q para T. A cada passo, a **BFS** acha o "
        "caminho aumentante mais curto e empurra o gargalo.")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        momento = escolher_momento("ek_momento")
    k = c2.segmented_control("k", [2, 3, 4], default=3, required=True,
                             key="ek_k")
    lam = c3.segmented_control("λ", [0.0, 1.0], default=1.0,
                               required=True, key="ek_lam")
    demo = demo_edmonds_karp(momento, k, lam)
    passos = demo["passos"]
    n = len(passos) - 1
    i = navegar_passos(f"ek_passo_{momento}_{k}_{lam}", n + 2)

    final = i == n + 1
    passo = passos[min(i, n)]
    c = st.columns(4)
    if i == 0:
        c[0].metric("Passo", "rede inicial")
    elif final:
        c[0].metric("Passo", "corte mínimo")
    else:
        c[0].metric("Passo", f"caminho {i} de {n}")
    c[1].metric("Gargalo", "—" if i == 0 or final
                else br(passo['gargalo'], 3))
    c[2].metric("Fluxo acumulado", br(passo['fluxo'], 3))
    c[3].metric("Capacidade do corte", br(demo['capacidade_corte'], 3)
                if final else "—")
    if 0 < i <= n:
        st.markdown(f"**Caminho aumentante (BFS):** "
                    f"`{formatar_caminho(passo['caminho'])}`")
    elif i == 0:
        st.markdown("Os arcos com fluxo mostram **fluxo / "
                    "capacidade**; passe o mouse num arco para ver a "
                    "capacidade. Avance para ver o primeiro caminho.")
    else:
        st.markdown(
            "Não há mais caminho de S a T na rede residual. Uma BFS a "
            "partir de S marca quem ainda é alcançável: esse é o **lado "
            "da evasão**. As arestas tracejadas formam o corte mínimo.")

    desenhar(MontadorVisual().rede_fluxo(
        demo["rede"], FONTE, SUMIDOURO, passo["residual"],
        demo["rotulos"],
        caminho=passo["caminho"] if 0 < i <= n else (),
        lado_s=demo["lado_s"] if final else None), altura=600)
    legenda((COR_EVASAO, "evadiu (treino)"),
            (COR_FORMATURA, "formou (treino)"),
            ("#bbbbbb", "○ sem rótulo"),
            (COR_DESTAQUE, "caminho da vez"),
            ("#4a5560", "arco com fluxo"))

    if final:
        st.subheader("Resultado")
        tabela = pd.DataFrame([
            {"aluno": v,
             "previsão": "evasão" if demo["previsoes"][v]
             else "formatura",
             "real": "evasão" if demo["real"][v] else "formatura"}
            for v in demo["sem_rotulo"]])
        tabela["acertou"] = np.where(
            tabela["previsão"] == tabela["real"], "✔", "✘")
        st.dataframe(tabela, hide_index=True, width="stretch")
    garantia(
        "**Garantia (Edmonds–Karp, aula 6):** fluxo máximo em O(V·E²), "
        "porque cada caminho aumentante é o mais curto (BFS). Pelo "
        "teorema do fluxo máximo e corte mínimo, o valor final do fluxo "
        "é igual à capacidade do corte. Ao final, o fluxo é "
        f"{br(demo['fluxo'], 3)} e o corte, "
        f"{br(demo['capacidade_corte'], 3)}.")


def pagina_g2() -> None:
    st.title("G2 · Árvore de dependências")
    st.markdown(
        "Cada atributo é um vértice, mais o **Desfecho**. O peso de cada "
        "par é a informação mútua (bits, no treino). O **Kruskal** liga "
        "todos os vértices sem ciclos com a maior informação total; os "
        "vizinhos do Desfecho são os fatores com informação própria.")
    momento = escolher_momento("g2_momento")
    modelo, resultado = modelo_g2(momento)
    oficial = ler_json("resultados.json")["g2"][momento]
    selo(abs(resultado["peso_total"] - oficial["peso_total"]) < TOLERANCIA,
         "resultados.json")

    csv = ler_csv(f"g2_arvore_{momento}.csv")
    estabilidade = {tuple(sorted((r.u, r.v))): r.estabilidade
                    for r in csv.itertuples()}
    c = st.columns(4)
    c[0].metric("Vértices", oficial["num_vertices"])
    c[1].metric("Peso total (bits)", br(oficial['peso_total'], 2))
    c[2].metric("Estabilidade média", pct(oficial['estabilidade_media'], 0),
                help="Fração de 100 bootstraps do treino em que cada "
                "aresta reaparece")
    c[3].metric("Sobreposição treino × teste",
                pct(oficial['sobreposicao_treino_teste'], 0))

    col_g, col_t = st.columns([3, 1.2])
    with col_g:
        desenhar(MontadorVisual().arvore_g2(
            resultado["arestas"], modelo.grafo.vertices(), DESFECHO,
            estabilidade), altura=620)
        st.caption("Espessura = informação mútua; opacidade = "
                   "estabilidade no bootstrap. Arraste os vértices.")
    with col_t:
        st.markdown("**Vizinhos do Desfecho**")
        viz = pd.DataFrame(oficial["vizinhos_desfecho"])
        st.dataframe(viz.rename(columns={"im_bits": "IM (bits)"}),
                     hide_index=True, width="stretch",
                     column_config={"estabilidade":
                                    st.column_config.NumberColumn(
                                        format="percent")})
    with st.expander("Todas as arestas da árvore"):
        st.dataframe(csv, hide_index=True, width="stretch")
    garantia("**Garantia (Kruskal, aula 5):** árvore geradora de peso "
             "máximo em O(E log E). Liga todos os vértices, não tem "
             "ciclos e nenhuma outra árvore tem soma de IM maior. "
             "Não é caminho mínimo nem causalidade.")


def pagina_kruskal() -> None:
    st.title("Kruskal passo a passo")
    st.markdown(
        "As arestas do grafo completo de informação mútua são examinadas "
        "em ordem decrescente. O **union-find** diz se as pontas já estão "
        "no mesmo componente: se estão, a aresta fecharia um ciclo e é "
        "rejeitada. Cada cor é um componente.")
    momento = escolher_momento("kr_momento")
    demo = demo_kruskal(momento)
    so_aceitas = st.toggle("Pular as rejeitadas", value=False)
    indices = [j for j, p in enumerate(demo["passos"])
               if p["aceita"] or not so_aceitas]
    pos = navegar_passos(f"kr_passo_{momento}_{so_aceitas}", len(indices))
    j = indices[pos]
    passo = demo["passos"][j]
    total = len(demo["vertices"]) - 1

    c = st.columns(4)
    c[0].metric("Aresta examinada", f"{j + 1} de {demo['num_examinadas']}",
                help=f"o grafo completo tem "
                f"{demo['num_arestas_completo']} arestas; o Kruskal para "
                "quando a árvore fica completa")
    c[1].metric("IM (bits)", br(passo["peso"], 4)
                if passo["peso"] >= 1e-4 else "< 0,0001")
    c[2].metric("Arestas na árvore", f"{passo['num_aceitas']} de {total}")
    c[3].metric("Componentes", len(set(passo["componentes"].values())))
    if passo["aceita"]:
        st.success(f"**Aceita:** {passo['u']} — {passo['v']} une dois "
                   "componentes.")
    else:
        st.error(f"**Rejeitada:** {passo['u']} e {passo['v']} já estão "
                 "no mesmo componente; a aresta fecharia um ciclo.")

    aceitas = [(p["u"], p["v"], p["peso"])
               for p in demo["passos"][:j] if p["aceita"]]
    desenhar(MontadorVisual().passo_kruskal(
        demo["vertices"], DESFECHO, aceitas,
        (passo["u"], passo["v"], passo["peso"], passo["aceita"]),
        passo["componentes"]), altura=700)
    legenda((COR_SEGURA, "aresta aceita na vez"),
            (COR_EVASAO, "aresta rejeitada (ciclo)"),
            ("#4a5560", "árvore até aqui"))
    garantia("**Garantia (Kruskal, aula 5):** ao fim, as "
             f"{total} arestas aceitas formam uma árvore geradora de peso "
             "máximo. As decisões vêm da saída do `kruskal()` do projeto; "
             "a demonstração só reconstitui a ordem.")


def pagina_g3() -> None:
    st.title("G3 · DAG de trajetórias")
    st.markdown(
        "Camadas: **ingresso** (bolsa e idade) → **financeiro** "
        "(mensalidade e dívida) → **aprovação no S1** → **aprovação no "
        "S2** → **desfecho**. O peso do arco é o número de alunos do "
        "treino que passam por ele. **Kahn** dá a ordem topológica (o "
        "número antes de cada estado) e o **Bellman-Ford**, com peso "
        "−ln(taxa), acha a trajetória de maior risco.")
    modelo, resultado = modelo_g3("T2")
    oficial = ler_json("resultados.json")["g3"]["T2"]
    selo(resultado["trajetoria_risco"] == oficial["trajetoria_risco"]
         and resultado["trajetoria_segura"] == oficial["trajetoria_segura"],
         "resultados.json")

    c1, c2, c3 = st.columns(3)
    ver_risco = c1.toggle("Trajetória de maior risco", value=True)
    ver_segura = c2.toggle("Trajetória mais segura", value=True)
    ver_virada = c3.toggle("Ponto de virada", value=True)

    risco = resultado["trajetoria_risco"]
    segura = resultado["trajetoria_segura"]
    virada = resultado["ponto_de_virada"]
    desenhar(MontadorVisual().dag_g3(
        modelo.grafo, resultado["ordem_topologica"],
        modelo.contagem_vertice_, modelo.evasoes_vertice_,
        modelo.min_suporte,
        risco=risco["caminho"] if ver_risco else None,
        segura=segura["caminho"] if ver_segura else None,
        virada=virada["arco"] if ver_virada else None), altura=640)
    legenda((COR_EVASAO, "maior risco"), (COR_SEGURA, "mais segura"),
            ("#8a96a3", "cor do estado: azul = forma, vermelho = evade"),
            ("#8a96a3", "- - - ponto de virada"))

    c = st.columns(3)
    c[0].metric("Trajetória de maior risco",
                f"{risco['evasoes']} de {risco['n']} evadiram")
    c[1].metric("Trajetória mais segura",
                f"{segura['evasoes']} de {segura['n']} evadiram")
    c[2].metric("Ponto de virada",
                f"{pct(virada['taxa_antes'])} → "
                f"{pct(virada['taxa_depois'])}",
                help=f"{virada['arco'][0]} → {virada['arco'][1]} "
                f"({virada['n']} alunos)")
    st.markdown(
        f"- **Risco:** {' → '.join(risco['caminho'])}\n"
        f"- **Segura:** {' → '.join(segura['caminho'])}\n"
        f"- **Virada:** {virada['arco'][0]} → {virada['arco'][1]}")

    teste = ler_csv("avaliacao_teste.csv")
    dag = teste[teste.metodo == "caminho_dag"].set_index("momento")
    st.caption(
        "Como preditor (acurácia balanceada no teste): "
        + "; ".join(
            f"{NOME_MOMENTO[m]} {br(dag.loc[m].acuracia_balanceada)}"
            for m in MOMENTOS if m in dag.index)
        + ". Fonte: `outputs/avaliacao_teste.csv`.")
    with st.expander("Ordem topológica de Kahn"):
        st.write(" → ".join(resultado["ordem_topologica"]))
    garantia("**Garantias:** Kahn (aula 3) devolve uma ordem topológica "
             "e prova que o grafo não tem ciclos. Bellman-Ford (aula 4) "
             "acha o caminho mínimo da origem a cada estado; com peso "
             "−ln(taxa), o caminho mínimo é o de maior produto das taxas "
             "de evasão (só arcos com pelo menos "
             f"{modelo.min_suporte} alunos).")


def pagina_simulador() -> None:
    st.title("G3 · Simulador de trajetória")
    st.markdown(
        "Monte o caminho de um aluno camada por camada. Em cada passo "
        "aparecem os estados alcançáveis no DAG e a taxa de evasão, no "
        "treino, de quem fez exatamente esse caminho até ali.")
    modelo, resultado = modelo_g3("T2")
    estados = estados_treino()
    nomes = {"L0": "Ingresso", "L1": "Financeiro",
             "L2": "Aprovação no S1", "L3": "Aprovação no S2"}

    caminho = [INICIO]
    filtro = np.ones(len(estados), dtype=bool)
    linhas = []
    colunas = st.columns(4)
    for col, camada in zip(colunas, nomes):
        sucessores = sorted(
            (v for v in modelo.grafo.vizinhos(caminho[-1])
             if v.startswith(camada)),
            key=lambda v: -modelo.contagem_arco_[(caminho[-1], v)])
        if not sucessores:
            break
        escolha = col.selectbox(
            nomes[camada], sucessores,
            format_func=lambda v: v.split(":", 1)[1], key=f"sim_{camada}")
        filtro &= (estados[camada] == escolha).to_numpy()
        n = int(filtro.sum())
        evasoes = int(estados.evadiu[filtro].sum())
        linhas.append({
            "camada": nomes[camada],
            "estado": escolha.split(":", 1)[1],
            "alunos no arco": modelo.contagem_arco_[(caminho[-1], escolha)],
            "evasão no arco": modelo.taxa_arco(caminho[-1], escolha),
            "alunos com este caminho": n,
            "evasão no caminho": evasoes / n if n else np.nan,
        })
        caminho.append(escolha)

    if len(linhas) == len(nomes):
        final = linhas[-1]
        taxa = final["evasão no caminho"]
        c = st.columns(3)
        c[0].metric("Alunos do treino com este caminho",
                    final["alunos com este caminho"])
        c[1].metric("Taxa de evasão", "—" if np.isnan(taxa)
                    else pct(taxa, 0))
        c[2].metric("Taxa do treino", pct(modelo.taxa_treino_, 0))
        if final["alunos com este caminho"] < modelo.min_suporte:
            st.warning(f"Menos de {modelo.min_suporte} alunos: estimativa "
                       "pouco confiável (o preditor usa o trecho mais "
                       "curto com suporte).")
    st.dataframe(pd.DataFrame(linhas), hide_index=True,
                 width="stretch",
                 column_config={
                     "evasão no arco": st.column_config.ProgressColumn(
                         format="percent", min_value=0, max_value=1),
                     "evasão no caminho": st.column_config.ProgressColumn(
                         format="percent", min_value=0, max_value=1)})
    desenhar(MontadorVisual().dag_g3(
        modelo.grafo, resultado["ordem_topologica"],
        modelo.contagem_vertice_, modelo.evasoes_vertice_,
        modelo.min_suporte, caminho_usuario=caminho), altura=560)
    legenda((COR_DESTAQUE, "caminho escolhido"))


# ----------------------------------------------------------------------
# Navegação
# ----------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Evasão em grafos", page_icon="🎓",
                       layout="wide")
    paginas = {
        "Resultados": [
            st.Page(pagina_visao_geral, title="Visão geral",
                    icon=":material/dashboard:", default=True),
        ],
        "G1 · Similaridade": [
            st.Page(pagina_g1, title="Corte mínimo",
                    icon=":material/hub:", url_path="g1"),
            st.Page(pagina_edmonds_karp, title="Edmonds–Karp passo a passo",
                    icon=":material/water:", url_path="edmonds-karp"),
            st.Page(pagina_exploracao, title="Exploração (só treino)",
                    icon=":material/tune:", url_path="exploracao"),
        ],
        "G2 · Dependências": [
            st.Page(pagina_g2, title="Árvore de informação mútua",
                    icon=":material/account_tree:", url_path="g2"),
            st.Page(pagina_kruskal, title="Kruskal passo a passo",
                    icon=":material/linear_scale:", url_path="kruskal"),
        ],
        "G3 · Trajetórias": [
            st.Page(pagina_g3, title="DAG de trajetórias",
                    icon=":material/route:", url_path="g3"),
            st.Page(pagina_simulador, title="Simulador",
                    icon=":material/alt_route:", url_path="simulador"),
        ],
    }
    st.navigation(paginas).run()


if __name__ == "__main__":
    main()
