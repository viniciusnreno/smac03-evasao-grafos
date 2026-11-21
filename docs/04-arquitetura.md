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
├── grafo.py                 classe Grafo: estrutura comum (base, fazer primeiro)          (a definir)
├── modelo.py                classe abstrata ModeloGrafo (base, fazer primeiro)           (a definir)
├── dados.py                 carga, filtro Enrolled, split 80/20, atributos por momento   (Carlos)
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
| `vertices()` / `arestas()` | listas de vértices e de `(u, v, peso)` |
| `para_networkx()` | conversão usada só em asserts de verificação |

**`ModeloGrafo` (`modelo.py`, abstrata, `abc.ABC`).** Interface comum para o `Pipeline` e a futura interface gráfica.

| Método abstrato | Devolve |
|---|---|
| `construir(base)` | o `Grafo` do modelo, a partir de uma `BaseDeDados` (só treino) |
| `executar()` | `dict` com os resultados do algoritmo (previsões, métricas, caminhos) |
| `destaques()` | `{"vertices": [...], "arestas": [(u, v), ...]}` a realçar na tela |

Atributos: `nome`, `descricao` (texto para a interface) e `grafo` (preenchido por `construir`).

| Modelo | `construir` | `executar` | `destaques` |
|---|---|---|---|
| G1 `GrafoSimilaridade` | kNN de Gower | chama `CorteMinimo` (Edmonds–Karp) | arestas do corte, alunos de cada lado |
| G2 `ArvoreDependencias` | grafo completo de informação mútua | Kruskal | arestas da árvore, vizinhos do Desfecho |
| G3 `DagTrajetorias` | DAG de estados | Kahn + Bellman-Ford | trajetória de maior risco |

Uso pelo `Pipeline` (e, no futuro, pela interface gráfica):

```python
for modelo in modelos:
    modelo.construir(base)
    resultado = modelo.executar()
    desenhar(modelo.grafo, modelo.destaques())
```

**Fronteira Carlos ↔ Rafael (G1):**

```python
GrafoSimilaridade(momento, k, pesos=None).construir(base) -> Grafo
CorteMinimo(lam=1.0).classificar(grafo, rotulos_treino, idx_teste) -> dict[int, bool]
#   rotulos_treino: dict[int, bool]   idx_teste: lista de vértices sem rótulo
```

- `CorteMinimo` é classe de algoritmo, não modelo; `GrafoSimilaridade.executar()` a chama.
- No G1, vértice = posição do aluno no DataFrame filtrado (após remover `Enrolled` e `reset_index`), de 0 a 3.629. No G2 e no G3, vértices são `str`.
- Os terminais S e T são criados dentro de `CorteMinimo`, não no grafo recebido.
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
| `corte_minimo.py` | `CorteMinimo` | Edmonds–Karp e classificação pelo corte (`classificar`) |
| `arvore_dependencias.py` | `ArvoreDependencias(ModeloGrafo)` | Kruskal com union-find sobre a informação mútua |
| `dag_trajetorias.py` | `DagTrajetorias(ModeloGrafo)` | Monta o DAG, ordena com Kahn e busca caminhos com Bellman-Ford |
| `avaliacao.py` | `Avaliador` | CV, teste, bootstrap, McNemar com Holm, ablação |
| `figuras.py` | `GeradorFiguras` | Figuras do relatório em PDF |
| `main.py` | `Pipeline` | Orquestra as classes e grava em `outputs/` |

Os métodos de `Grafo`, `ModeloGrafo` e da fronteira do G1 estão fixados acima. Os demais métodos ficam a critério do dono do módulo.

**Lógica separada da interface.** As classes devolvem dados; não imprimem, não plotam e não gravam arquivos fora da própria responsabilidade (só `GeradorFiguras` e `Pipeline` escrevem em disco). Assim, a interface gráfica prevista como bônus pode reaproveitar as mesmas classes sem mudar o núcleo. Quando ela existir, ficará em um módulo próprio e o `main.py` poderá abri-la.

Métodos auxiliares internos começam com `_` (ex.: `_bfs_residual`).

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
