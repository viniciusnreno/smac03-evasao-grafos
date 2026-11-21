# Arquitetura

## Tecnologias

| Item | Versão / uso |
|---|---|
| Python | 3.12 |
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
├── figuras.py               Figs. 3–5 em PDF                                            (Rafael)
└── main.py                  pipeline completo → outputs/                               (Carlos)
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

Constantes do módulo: `GRUPOS` (os 7 grupos de atributos, usados na ablação), `MOMENTOS`, `DERIVADOS` e `NOMINAIS` (códigos sem ordem, comparados por igualdade na distância de Gower; os demais são numéricos ou 0/1). Taxa de aprovação com 0 inscritas fica `NaN`: é a faixa "sem inscrição", tratada por quem usa a base.

A divisão com semente 42 reproduz os números de `analise_v2` no treino (0% de aprovação no S1: n = 387, 99,5% de evasão; mensalidade atrasada: n = 383, 94,0%).

**Fronteira Carlos ↔ Rafael (G1):**

```python
GrafoSimilaridade(momento, k, pesos=None).construir(base) -> Grafo
ClassificadorCorteMinimo(lam=1.0).classificar(grafo, rotulos_treino, idx_teste) -> dict[int, bool]
#   rotulos_treino: dict[int, bool]   idx_teste: lista de vértices sem rótulo
```

- `corte_minimo.py` tem duas classes de algoritmo, que não são modelos:
  - `EdmondsKarp`: fluxo máximo genérico. `fluxo_maximo(rede, s, t) -> float` e `lado_da_fonte() -> set` (corte mínimo por BFS no residual). Não altera a rede recebida.
  - `ClassificadorCorteMinimo(lam=1.0, contrair=True)`: monta a rede do G1 com S e T, chama o `EdmondsKarp` e traduz o corte em previsões. Depois de `classificar`, expõe `fluxo`, `num_caminhos`, `lado_evasao` e `arestas_cortadas()` (para `destaques`).
- `GrafoSimilaridade.executar()` usa o `ClassificadorCorteMinimo`.
- **`GrafoSimilaridade(momento, k, pesos=None, lam=1.0, incluir_derivados=True, excluir=())`**:
  - `construir(base, idx_treino=None, idx_teste=None)`: por padrão usa a divisão oficial; na CV, passe as dobras. Os vértices são treino + teste; pesos e amplitudes vêm só de `idx_treino`, e os rótulos do teste nunca são lidos.
  - `pesos=None` calcula a informação mútua no treino. Para isso o atributo é discretizado: taxas de aprovação nas faixas do DAG (sem inscrição, 0%, (0,50%), [50,100%), 100%), categorias nominais com menos de 30 alunos agrupadas em "outros" e numéricas com mais de 10 valores em quartis. Essa discretização reproduz a IM do `analise_v2`. Um dict fixa os pesos e restringe os atributos (`dict.fromkeys(cols, 1.0)` dá pesos iguais).
  - Valor ausente (taxa sem inscrição) é uma categoria própria: dois ausentes têm δ = 0, um só ausente tem δ = 1. O δ numérico é limitado a 1 (aluno do teste fora da amplitude do treino).
  - O kNN liga i e j se um estiver entre os k mais próximos do outro. Empates de distância ficam com o aluno de menor número.
  - `K_PADRAO = {"T0": 25, "T1": 25, "T2": 40}`. `incluir_derivados=True` foi escolhido por CV de 5 dobras no treino (T1: 0,847 contra 0,822 sem as taxas; T2: 0,875 contra 0,861).
  - `executar()` devolve `previsoes`, `fluxo`, `num_caminhos`, `num_vertices`, `num_arestas` e `pesos`. `destaques()` devolve os alunos do lado da evasão e as arestas cortadas. `distancia(i, j)` dá o Gower de um par (exemplo dos "gêmeos").
  - **Assortatividade (H5):** os valores oficiais 0,27 / 0,35 / 0,44 saem do kNN com k = 10, só treino, **pesos iguais e sem as taxas derivadas**: `GrafoSimilaridade(m, 10, pesos=dict.fromkeys(base.atributos(m), 1.0), incluir_derivados=False).construir(base, idx_teste=[])`. Com os pesos de IM, a assortatividade é maior.
- **Viés λ:** cada aluno sem rótulo ganha S → u com capacidade λ·p e u → T com λ·(1 − p), sendo p a taxa de evasão do treino. Como p < 0,5, o viés puxa para T. Essa é a nossa leitura do plano v2, que não define p; `lam=0` desliga o viés.
- **Contração:** com `contrair=True` (padrão), os rotulados são fundidos em S e T antes do fluxo, o que não muda o corte e reduz a rede.
- No G1, vértice = posição do aluno no DataFrame filtrado (após remover `Enrolled` e `reset_index`), de 0 a 3.629. No G2 e no G3, vértices são `str`.
- Os terminais S e T são criados dentro de `ClassificadorCorteMinimo`, não no grafo recebido.
- **Rótulos:** `True` = evadiu (`Dropout`), `False` = formou (`Graduate`).
- O que for específico de um modelo volta no dict de `executar()`. Não crie métodos abstratos novos sem combinar.
- Parâmetros novos (como o viés λ) entram como argumentos opcionais com valor padrão. Hiperparâmetros vão no `__init__`.

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
| `figuras.py` | `GeradorFiguras` | Figuras do relatório em PDF |
| `main.py` | `Pipeline` | Orquestra as classes e grava em `outputs/` |

Os métodos de `Grafo`, `ModeloGrafo` e da fronteira do G1 estão fixados acima. Os demais métodos ficam a critério do dono do módulo.

**Lógica separada da interface.** As classes devolvem dados; não imprimem, não plotam e não gravam arquivos fora da própria responsabilidade (só `GeradorFiguras` e `Pipeline` escrevem em disco). Assim, a interface gráfica prevista como bônus pode reaproveitar as mesmas classes sem mudar o núcleo. Quando ela existir, ficará em um módulo próprio e o `main.py` poderá abri-la.

Métodos auxiliares internos começam com `_` (ex.: `_bfs`).

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py        # gera outputs/
```

## Fluxo de git

- Uma branch por tarefa, a partir da `main`; PR para a `main`.
- Mensagens de commit em português, no imperativo ("Implementa Edmonds–Karp").
- Não versionar `.venv/`, `__pycache__/`, artefatos de compilação LaTeX nem material da disciplina.
