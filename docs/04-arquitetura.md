# Arquitetura

## Tecnologias

| Item | Versão / uso |
|---|---|
| Python | 3.12 |
| streamlit | interface gráfica (bônus), `src/app.py` |
| pyvis | só o vis-network que vem empacotado nele, embutido no HTML da interface (funciona sem internet) |
| pandas, numpy | dados e matrizes |
| networkx | **somente verificação** (asserts contra as implementações próprias) |
| scipy | testes estatísticos (McNemar, bootstrap, Spearman) |
| scikit-learn | split estratificado, CV, informação mútua, regressão logística de referência |
| matplotlib | figuras do relatório em PDF vetorial |
| LaTeX | template IMC/UNIFEI (`relatorio/imc.cls`, bibliografia `apalike2_ptBR`), editado no Overleaf |

Versões fixadas em [requirements.txt](../requirements.txt).

## Estrutura

```
data/data.csv                dataset UCI 697
src/
├── grafo.py                 classe Grafo: estrutura comum (base)                         (Rafael)
├── modelo.py                classe abstrata ModeloGrafo (base)                           (Rafael)
├── dados.py                 carga, filtro Enrolled, split 80/20, atributos por momento   (Rafael)
├── grafo_similaridade.py    Gower + kNN (G1)                                            (Carlos)
├── corte_minimo.py          Edmonds–Karp próprio e classificação pelo corte (G1)        (Rafael)
├── arvore_dependencias.py   Kruskal + union-find, árvore máxima de IM (G2)              (Carlos)
├── dag_trajetorias.py       DAG, Kahn, Bellman-Ford (G3)                                (Carlos)
├── avaliacao.py             CV, teste, bootstrap, McNemar + Holm, ablação               (Rafael)
├── fatores.py               lift treino × teste, replicação das direções, IM (H1, H2)   (Rafael)
├── figuras.py               Figs. 3–5 em PDF → relatorio/figs/                          (Rafael)
├── main.py                  pipeline completo → outputs/                               (Carlos)
├── app.py                   interface gráfica (Streamlit), só a tela                    (Rafael)
├── visual.py                grafos → desenho (vis-network) e opções clicáveis (ECharts)  (Rafael)
├── explicacao.py            similaridade de Gower decomposta por atributo (painel do G1) (Rafael)
└── passo_a_passo.py         passos do Edmonds–Karp e do Kruskal para a interface        (Rafael)
outputs/                     resultados (json/csv) e figuras gerados pelo pipeline
docs/                        esta documentação e os planos
relatorio/                   projeto LaTeX (Nathan organiza)
slides/                      apresentação
```

Cada pessoa edita só os próprios arquivos. Mudança em arquivo de outra pessoa: combinar antes.

## Contrato do grafo (acordado)

Duas classes base sustentam os três modelos.

**`Grafo` (`grafo.py`, concreta).** Lista de adjacência com peso, `dict[vértice, dict[vértice, float]]`, dirigida ou não. Vértices de qualquer tipo hashable.

| Método | Faz |
|---|---|
| `adicionar_vertice(v)` | inclui um vértice sem arestas |
| `adicionar_aresta(u, v, peso)` | inclui u → v; se não dirigido, também v → u |
| `vizinhos(u)` | `{v: peso}` das arestas que saem de u |
| `peso(u, v)` | peso da aresta u → v |
| `tem_aresta(u, v)` | indica se a aresta u → v existe |
| `vertices()` / `arestas()` | listas de vértices e de `(u, v, peso)`; aresta não dirigida aparece uma vez |
| `len(g)` / `num_arestas()` / `v in g` | número de vértices, número de arestas, pertinência |
| `para_networkx(atributo="weight")` | conversão usada só em asserts; use `atributo="capacity"` para comparar fluxo |

`vizinhos(u)` devolve uma visão somente leitura, sem cópia: o Edmonds–Karp deve montar a própria rede residual em vez de alterar o grafo recebido. Vértice ou aresta inexistente gera `KeyError`.

**`ModeloGrafo` (`modelo.py`, abstrata, `abc.ABC`).** Interface comum para o `Pipeline` e a futura interface gráfica.

| Método abstrato | Devolve |
|---|---|
| `construir(base)` | o `Grafo` do modelo, a partir de uma `BaseDeDados` (só treino) |
| `executar()` | `dict` com os resultados do algoritmo (previsões, métricas, caminhos) |
| `destaques()` | `{"vertices": [...], "arestas": [(u, v), ...]}` a realçar na tela |

Atributos: `nome`, `descricao` (texto para a interface) e `grafo` (preenchido por `construir`).

| Modelo | `construir` | `executar` | `destaques` |
|---|---|---|---|
| G1 `GrafoSimilaridade` | kNN de Gower | chama `ClassificadorCorteMinimo` (Edmonds–Karp) | arestas do corte, alunos de cada lado |
| G2 `ArvoreDependencias` | grafo completo de informação mútua | Kruskal | arestas da árvore, vizinhos do Desfecho |
| G3 `DagTrajetorias` | DAG de estados | Kahn + Bellman-Ford | trajetória de maior risco |

Uso pelo `Pipeline` (e, no futuro, pela interface gráfica):

```python
for modelo in modelos:
    modelo.construir(base)
    resultado = modelo.executar()
    desenhar(modelo.grafo, modelo.destaques())
```

**`BaseDeDados` (`dados.py`).** Criada com `BaseDeDados()`: lê o CSV, remove os Enrolled, numera os alunos de 0 a 3.629 e faz a divisão oficial.

| Membro | Devolve |
|---|---|
| `dados` | DataFrame com os 36 atributos, `taxa_aprov_s1`, `taxa_aprov_s2`, `Target` e `evadiu` (True = Dropout) |
| `idx_treino`, `idx_teste` | posições dos alunos (2.904 e 726), ordenadas |
| `atributos(momento, incluir_derivados=False)` | colunas de `"T0"` (24), `"T1"` (30) ou `"T2"` (36); com derivados, + taxa de aprovação dos semestres disponíveis |
| `X(idx, momento, incluir_derivados=False)` | DataFrame dos atributos do momento para os alunos `idx` |
| `y(idx)` | array booleano do desfecho |
| `rotulos(idx)` | `{aluno: evadiu}`, o formato do `ClassificadorCorteMinimo` |
| `taxa_evasao(idx=None)` | fração de evadidos |
| `dividir(semente)` | nova divisão estratificada (para os 10 splits de robustez) |

Constantes do módulo: `GRUPOS` (os 7 grupos de atributos, usados na ablação), `MOMENTOS`, `DERIVADOS`, `FAIXAS_APROVACAO` (com a função `faixa_aprovacao`, usada pelo G1 e pelo G3) e `NOMINAIS` (códigos sem ordem, comparados por igualdade na distância de Gower; os demais são numéricos ou 0/1). Taxa de aprovação com 0 inscritas fica `NaN`: é a faixa "sem inscrição", tratada por quem usa a base.

A divisão com semente 42 reproduz os números de `analise_v2` no treino (0% de aprovação no S1: n = 387, 99,5% de evasão; mensalidade atrasada: n = 383, 94,0%).

**Fronteira Carlos ↔ Rafael (G1):**

```python
GrafoSimilaridade(momento, k, pesos=None).construir(base) -> Grafo
ClassificadorCorteMinimo(lam=1.0).classificar(grafo, rotulos_treino, idx_teste) -> dict[int, bool]
#   rotulos_treino: dict[int, bool]   idx_teste: lista de vértices sem rótulo
```

- `corte_minimo.py` tem duas classes de algoritmo, que não são modelos:
  - `EdmondsKarp(registrar=False)`: fluxo máximo genérico. `fluxo_maximo(rede, s, t) -> float` e `lado_da_fonte() -> set` (corte mínimo por BFS no residual). Não altera a rede recebida. Com `registrar=True`, guarda em `historico` cada caminho aumentante e o gargalo (interface).
  - `ClassificadorCorteMinimo(lam=1.0, contrair=True, registrar=False)`: monta a rede do G1 com S e T, chama o `EdmondsKarp` e traduz o corte em previsões. Depois de `classificar`, expõe `fluxo`, `num_caminhos`, `lado_evasao` e `arestas_cortadas()` (para `destaques`); com `registrar=True`, também `rede` e `historico`.
- `GrafoSimilaridade.executar()` usa o `ClassificadorCorteMinimo`.
- **`GrafoSimilaridade(momento, k, pesos=None, lam=1.0, incluir_derivados=True, excluir=())`**:
  - `construir(base, idx_treino=None, idx_teste=None)`: por padrão usa a divisão oficial; na CV, passe as dobras. Os vértices são treino + teste; pesos e amplitudes vêm só de `idx_treino`, e os rótulos do teste nunca são lidos.
  - `pesos=None` calcula a informação mútua no treino. Para isso o atributo é discretizado: taxas de aprovação nas faixas do DAG (sem inscrição, 0%, (0,50%), [50,100%), 100%), categorias nominais com menos de 30 alunos agrupadas em "outros" e numéricas com mais de 10 valores em quartis. Essa discretização reproduz a IM do `analise_v2`. Um dict fixa os pesos e restringe os atributos (`dict.fromkeys(cols, 1.0)` dá pesos iguais).
  - Valor ausente (taxa sem inscrição) é uma categoria própria: dois ausentes têm δ = 0, um só ausente tem δ = 1. O δ numérico é limitado a 1 (aluno do teste fora da amplitude do treino).
  - O kNN liga i e j se um estiver entre os k mais próximos do outro. Empates de distância ficam com o aluno de menor número.
  - `K_PADRAO = {"T0": 25, "T1": 25, "T2": 40}`. `incluir_derivados=True` foi escolhido por CV de 5 dobras no treino (T1: 0,847 contra 0,822 sem as taxas; T2: 0,875 contra 0,861).
  - `executar()` devolve `previsoes`, `fluxo`, `num_caminhos`, `num_vertices`, `num_arestas` e `pesos`. `destaques()` devolve os alunos do lado da evasão e as arestas cortadas. `distancia(i, j)` dá o Gower de um par (exemplo dos "gêmeos").
  - **Assortatividade (H5):** os valores oficiais 0,27 / 0,35 / 0,44 saem do kNN com k = 10, só treino, **pesos iguais e sem as taxas derivadas**: `GrafoSimilaridade(m, 10, pesos=dict.fromkeys(base.atributos(m), 1.0), incluir_derivados=False).construir(base, idx_teste=[])`. Com os pesos de IM, a assortatividade é maior.
- **Viés λ:** cada aluno sem rótulo ganha u → T com capacidade λ·q e S → u com λ·(1 − q), sendo q a taxa de evasão do treino, como diz o plano v2 ("arcos u → T e S → u, com capacidade λ·q e λ·(1−q)"). Como q < 0,5, o viés puxa para S (evasão) e aumenta o recall da evasão, o mesmo padrão da tabela oficial (T0: recall 0,606 com λ = 0 e 0,820 com λ = 1). A primeira versão invertia o sentido; foi corrigida pela leitura do plano e pela CV no treino (ver `estado.md`). `lam=0` desliga o viés.
- **Contração:** com `contrair=True` (padrão), os rotulados são fundidos em S e T antes do fluxo, o que não muda o corte e reduz a rede.
- No G1, vértice = posição do aluno no DataFrame filtrado (após remover `Enrolled` e `reset_index`), de 0 a 3.629. No G2 e no G3, vértices são `str`.
- Os terminais S e T são criados dentro de `ClassificadorCorteMinimo`, não no grafo recebido.
- **Rótulos:** `True` = evadiu (`Dropout`), `False` = formou (`Graduate`).
- O que for específico de um modelo volta no dict de `executar()`. Não crie métodos abstratos novos sem combinar.
- Parâmetros novos (como o viés λ) entram como argumentos opcionais com valor padrão. Hiperparâmetros vão no `__init__`.
- `GrafoSimilaridade.discretizar(serie)` devolve os códigos das faixas usadas na IM. O G2 usa as mesmas faixas.
- `GrafoSimilaridade.assortatividade(rotulos=None)` calcula a assortatividade de Newman do desfecho, sem peso nas arestas, só entre vértices rotulados. Rótulos embaralhados servem ao teste de permutação. Confere com o networkx.

**`ArvoreDependencias(momento, incluir_derivados=True, excluir=())` (G2):**

- `construir(base, idx=None)`: grafo completo com vértices = atributos do momento + `"Desfecho"` (25 / 32 / 39 vértices) e peso = IM em bits, calculada nos alunos `idx` (padrão: treino).
- `executar()`: Kruskal decrescente com `UniaoBusca` (união por posto e compressão de caminho). Devolve `arestas` (u, v, im), `peso_total`, `vizinhos_desfecho`, `im_desfecho`.
- `ArvoreDependencias.kruskal(grafo, maximo=True)`: Kruskal genérico. Empates são desfeitos pelo nome dos vértices.
- `estabilidade(base, n=100)`: fração dos bootstraps do treino em que cada aresta volta. `sobreposicao(a, b)`: fração das arestas de `a` presentes em `b`.
- **Empates exatos:** PIB, desemprego e inflação têm um valor por coorte, e `International` equivale a `Nacionality`. Por isso, mais de uma árvore máxima é possível. O assert contra o networkx compara o peso total e a lista de pesos, não o conjunto de arestas.

**`Pipeline` (`main.py`):** `Pipeline(saida, momentos, n_bootstrap_arvore=100, n_permutacoes=200, verificar=True, avaliacao_completa=True, progresso=None).executar() -> dict`. Não imprime: as mensagens vão para `progresso` (`print` na linha de comando). Chama o `Avaliador` no fim.

| Arquivo em `outputs/` | Conteúdo | Usado em |
|---|---|---|
| `split.csv` | aluno e conjunto (treino ou teste) | reprodutibilidade |
| `hiperparametros.json` | parâmetros do protocolo e `ambiente` (Python, bibliotecas, sistema, CPU, RAM, tempo por etapa) | Tab. 2 e "Ambiente de testes" |
| `g1_previsoes_teste.csv` | aluno, desfecho real, previsão em T0/T1/T2 | `Avaliador` (bootstrap, McNemar) |
| `g1_metricas_teste.csv` | acurácia balanceada, recall, especificidade, precisão, F1, VP/FP/FN/VN; corte mínimo e classe majoritária | Tab. 3 |
| `g1_assortatividade.csv` | r de Newman, p da permutação, máximo nulo | H5 |
| `g2_arvore_T*.csv` | arestas da árvore com IM, estabilidade e presença na árvore do teste | Fig. 5, H2 |
| `g3_previsoes_teste.csv` / `g3_metricas_teste.csv` | risco e previsão do caminho no DAG por aluno; métricas no teste | Tab. 3 |
| `g2_im_desfecho.csv` | IM de cada atributo com o desfecho por momento | H1 |
| `avaliacao_teste.csv` | por momento e método (majoritária, regra de 1 variável, DAG, corte, corte só com essenciais, regressão logística): métricas, IC 95% e AUC | **Tab. 3** |
| `avaliacao_mcnemar.csv` | corte × regra e corte × DAG: acertos exclusivos, p e p de Holm | Tab. 3, H4 |
| `avaliacao_cv.csv` | CV de 5 dobras no treino (média e dp) de corte, DAG e regra | seção 4 |
| `avaliacao_ablacao.csv` | Δ por grupo, IC 95% e classe (essencial, neutro, prejudicial) | Fig. de ablação, H3 |
| `avaliacao_backward.csv` | histórico da eliminação backward (atributo, acurácia na CV, saiu ou ficou) | H3 |
| `avaliacao_robustez.csv` | 10 divisões 80/20: média, dp e acurácias | Tab. 3 |
| `avaliacao_previsoes.csv` | previsão de cada método para cada aluno do teste | conferência |
| `resultados.json` | tudo acima, mais fluxo, caminhos aumentantes e pesos do G1 | relatório e slides |
| `fatores_lift.csv` | 27 fatores da Fig. 3: n, taxa e lift no treino e no teste, p de Fisher e q (BH) no treino | **Fig. 3**, H2 |
| `fatores_faixas.csv` | lift de todas as faixas de todos os atributos (cortes do treino) | H2 |
| `fatores_im.csv` | IM com o desfecho por atributo, no treino e no teste | H1, H2 |
| `fatores_replicacao.json` | % das faixas significativas com a mesma direção no teste, % de todas, Spearman da IM, topo 10 em comum | H2 |

Os arquivos `fatores_*` e as figuras são gravados por `python src/figuras.py`, que lê o resto de `outputs/` (rode o `main.py` antes).

`python src/main.py --rapido` pula a ablação, o backward e a robustez (cerca de 3 min). A execução completa leva cerca de 2h10 (o backward é a parte lenta).

O G3 entra no pipeline por `DagTrajetorias(momento)`; o risco e a previsão por aluno vão para `g3_previsoes_teste.csv`, e o resto do `executar()` para `resultados.json`.

A Fig. 2 (`fig:protocolo`) é gerada por `relatorio/figs/gerar_fig_protocolo.py`, que lê `outputs/resultados.json`: rode depois do `main.py`.

**G3 (`dag_trajetorias.py`).** Três classes:

- `Kahn().ordenar(grafo) -> list`: ordem topológica; `ValueError` se houver ciclo.
- `BellmanFord()`: `distancias(grafo, fonte) -> dict` e `caminho(destino) -> list`; aceita pesos negativos e detecta ciclo negativo.
- `DagTrajetorias(momento="T2", min_suporte=20, limiar=None)`: `construir(base, idx_treino=None, idx_teste=None)` monta o DAG só com o treino (peso do arco = número de alunos); do teste guarda só os estados L0–L3. `executar()` devolve `ordem_topologica`, `trajetoria_risco` e `trajetoria_segura` (`caminho`, `n`, `evasoes` no treino), `ponto_de_virada`, `risco` (nota contínua, serve para a AUC) e `previsoes` por aluno, e `limiar`. `destaques()` devolve a trajetória de maior risco. Estatísticas: `contagem_vertice_`, `evasoes_vertice_`, `contagem_arco_`, `evasoes_arco_`, `probabilidade(u, v)`, `taxa_arco(u, v)`; `DagTrajetorias.estados(df)` dá os estados de qualquer aluno.

**Avaliação (`avaliacao.py`).** Não lê nem grava arquivos; devolve dicts e DataFrames.

- `Avaliador(dobras=5, n_bootstrap=1000, n_splits=10, semente=42)`:
  - Testes estatísticos: `metricas(y, prev)` (usado também pelo `Pipeline`), `auc(y, nota)`, `ic_bootstrap(y, prev)` (IC 95% por percentil da acurácia balanceada), `mcnemar(y, a, b)` (exato, binomial) e `holm(pvalores)`.
  - Fábricas de classificador `classificar(idx_treino, idx_teste, excluir=()) -> {aluno: bool}`: `classificador_g1`, `classificador_dag` e `classificador_regra`.
  - Avaliação no treino: `particoes(base)` dá as dobras estratificadas (semente 42); `validacao_cruzada(base, classificar, excluir=())` usa só o treino.
  - `ablacao_grupos(base, momento, classificar)`: Δ = (sem − com) por dobra pareada, com IC de t (4 g.l.). É essencial se o IC fica todo abaixo de 0, prejudicial se fica todo acima e neutro nos outros casos. Tirar um grupo acadêmico tira também a taxa derivada dele.
  - `eliminacao_backward(base, momento, classificar, tolerancia=0)`: versão gulosa ordenada. Tenta tirar cada atributo, do de menor para o de maior IM no treino, e mantém a remoção se a CV não piorar. É um teste por atributo, para caber no tempo do trabalho.
  - `robustez(base, classificar)`: 10 divisões 80/20, com sementes sorteadas a partir da 42.
  - Teste: `avaliar_teste(y, previsoes, notas)` e `comparar(y_por_momento, previsoes, pares)`, com McNemar e Holm sobre toda a família de comparações.
  - `executar(base, ablacao=True, backward=True, robustez=True, progresso=None)` roda tudo e devolve as tabelas `cv`, `teste`, `previsoes`, `mcnemar`, `regras`, `ablacao`, `backward` e `robustez`.
- `RegraUmaVariavel`: melhor limiar de um atributo numérico, aprendido no treino. Saem T0 idade ≥ 22 (o oficial dizia ≥ 23, que no treino fica 0,003 abaixo), T1 aprovação no S1 < 78% e T2 aprovação no S2 < 79%.
- `RegressaoLogisticaReferencia`: one-hot nas nominais, padronização nas numéricas, indicador de ausente e classes balanceadas. É só uma régua externa.
- A família do McNemar tem 6 testes: corte × regra e corte × DAG, em T0, T1 e T2. A AUC do corte mínimo continua sem nota contínua (pendente).

**Fatores (`fatores.py`).** `AnaliseFatores(min_teste=20, alfa=0.05).executar(base)` devolve `lift` (27 fatores da Fig. 3, mesmas definições e mesmos lifts do plano v2), `faixas`, `im` e `replicacao`. Faixas e limiares vêm só do treino (mesmas regras da IM: faixas do DAG nas taxas, nominais raras em "outros", quartis nas numéricas); o teste só mede os mesmos grupos. Significância no treino por Fisher exato com Benjamini-Hochberg. Resultado atual: 95,8% das 95 faixas significativas mantêm a direção no teste (89,3% das 169 avaliadas), Spearman da IM 0,946, 9 do topo 10 em comum. O oficial (`analise_v2`) dizia 96,2%, 88,1%, 0,958 e 8 de 10; a diferença vem da definição da família de faixas, que o `analise_v2` não documenta.

**Figuras (`figuras.py`).** `GeradorFiguras(entrada="outputs", destino="relatorio/figs").gerar_todas()` grava `fig-lift.pdf` (Fig. 3, largura de coluna), `fig-ablacao.pdf` (Fig. 4, coluna) e `fig-arvore.pdf` (Fig. 5, árvore de T1 para `figure*`). A árvore usa layout próprio em camadas (profundidade a partir do Desfecho), sem networkx.

## Padrões de código

**PEP 8 sempre.** 4 espaços de indentação, linhas de até 79 caracteres, `snake_case` para funções, métodos e variáveis, `CamelCase` para classes, `MAIUSCULAS` para constantes, imports no topo agrupados em padrão, terceiros e locais.

**Orientação a objetos.** Cada módulo expõe uma classe principal:

| Módulo | Classe | Papel |
|---|---|---|
| `grafo.py` | `Grafo` | Estrutura de dados comum aos três modelos |
| `modelo.py` | `ModeloGrafo` (abstrata) | Interface comum dos modelos |
| `dados.py` | `BaseDeDados` | Carrega, filtra os Enrolled, separa 80/20 e entrega os atributos de cada momento |
| `grafo_similaridade.py` | `GrafoSimilaridade(ModeloGrafo)` | G1: Gower e kNN; executa o corte mínimo |
| `corte_minimo.py` | `EdmondsKarp` e `ClassificadorCorteMinimo` | Fluxo máximo; classificação pelo corte (`classificar`) |
| `arvore_dependencias.py` | `ArvoreDependencias(ModeloGrafo)` | Kruskal com union-find sobre a informação mútua |
| `dag_trajetorias.py` | `DagTrajetorias(ModeloGrafo)` | Monta o DAG, ordena com Kahn e busca caminhos com Bellman-Ford |
| `avaliacao.py` | `Avaliador` | CV, teste, bootstrap, McNemar com Holm, ablação |
| `fatores.py` | `AnaliseFatores` | Lift treino × teste, replicação das direções e ranking de IM |
| `figuras.py` | `GeradorFiguras` | Figuras do relatório em PDF |
| `main.py` | `Pipeline` | Orquestra as classes e grava em `outputs/` |
| `visual.py` | `MontadorVisual`, `GrafoECharts` | Desenha o DAG do G3 (vis-network) e, clicáveis, a ego-rede do G1 e a árvore do G2 (ECharts) |
| `explicacao.py` | `ExplicadorGower` | Decompõe a similaridade de Gower de uma aresta do G1 por atributo, com os pesos do treino; confere a soma com `GrafoSimilaridade.distancia` |
| `passo_a_passo.py` | `DemoKruskal`, `ExploradorArvore` | Passos do Kruskal; estado da régua de IM, caminho até o Desfecho com gargalo e layout radial |
| `app.py` | (funções de página) | Interface Streamlit; só tela, sem cálculo próprio |

Os métodos de `Grafo`, `ModeloGrafo` e da fronteira do G1 estão fixados acima. Os demais métodos ficam a critério do dono do módulo.

**Lógica separada da interface.** As classes devolvem dados; não imprimem, não plotam e não gravam arquivos fora da própria responsabilidade (só `GeradorFiguras` e `Pipeline` escrevem em disco). Assim, a interface gráfica reaproveita as mesmas classes sem mudar o núcleo.

**Interface gráfica (`app.py`, bônus).** Streamlit com 4 páginas: **Resumo** (dados, os três grafos, Tab. 3, McNemar e "o que mais pesa": pesos do G1, ablação e lifts), e uma página por grafo, cada uma com o desenho, o quadro "como ler" (vértice, aresta e codificação visual) e o simulador do grafo. **G1**: ego-rede clicável de um aluno de teste com o corte; o clique mostra por que dois alunos estão ligados. **G2**: árvore em layout radial (Desfecho no centro) com uma régua de informação mútua que mostra a floresta do Kruskal em cada ponto (▶ anima a descida) e, ao clicar num atributo, o caminho único até o Desfecho com o gargalo (propriedade minimax da árvore geradora máxima, conferida contra o grafo completo). **G3**: um só DAG (vértices e arestas, em camadas) com as trajetórias de risco e segura, o ponto de virada e o caminho montado no simulador; o seletor T0/T1/T2 esmaece as camadas ainda não observadas, limita o simulador a elas e mostra o risco e a previsão do G3 pela mesma regra do preditor (maior trecho com suporte, limiar = taxa do treino). Regras:

- Os modelos são refeitos na tela com os hiperparâmetros de `outputs/hiperparametros.json`; fluxo, número de arestas, peso da árvore e trajetórias são conferidos com `resultados.json`, e um aviso só aparece se divergirem. Métricas, IC e p-valores são sempre lidos de `outputs/`.
- `DemoKruskal` **não altera o Kruskal**: lê a árvore devolvida por `ArvoreDependencias.kruskal`, reconstitui a ordem de exame (IM decrescente, empate pelo nome) e marca cada aresta como aceita se está na árvore. O `UniaoBusca` só acompanha os componentes para colorir o desenho e confere que cada rejeitada fecharia um ciclo.
- Tema claro e escuro em `.streamlit/config.toml` (o modo segue o sistema e troca em ⋮ → Settings), com a identidade do relatório: títulos em Libertinus Serif (a Libertine do LaTeX), texto em Libertinus Sans e as cores das figuras, que `visual.py` importa de `figuras.py` (fonte única). Os desenhos leem o modo ativo (`st.context.theme.type`) e ajustam fundo e texto.
- Modo apresentação (barra lateral): esconde os quadros "como ler", as notas e os detalhes e aumenta os grafos em 30%, para projetor e vídeo.
- O networkx não é usado na interface; as posições do DAG e do Kruskal são calculadas à mão.

Métodos auxiliares internos começam com `_` (ex.: `_bfs`).

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py        # gera outputs/
python src/figuras.py     # fatores_* em outputs/ e Figs. 3–5 em relatorio/figs/
streamlit run src/app.py  # interface gráfica (lê outputs/; rode o main.py antes)
```

## Fluxo de git

- Uma branch por tarefa, a partir da `main`; PR para a `main`.
- Mensagens de commit em português, no imperativo ("Implementa Edmonds–Karp").
- Não versionar `.venv/`, `__pycache__/`, artefatos de compilação LaTeX nem material da disciplina.
