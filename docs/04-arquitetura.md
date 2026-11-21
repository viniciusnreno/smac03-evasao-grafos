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

Grafos são listas de adjacência com capacidade:

```python
Grafo = dict[int, dict[int, float]]   # grafo[u][v] = capacidade da aresta u → v
```

- **Vértices** são inteiros: a posição do aluno no DataFrame filtrado (após remover `Enrolled` e `reset_index`), de 0 a 3.629.
- **Arestas do kNN são simétricas**: se `grafo[i][j]` existe, `grafo[j][i]` existe com o mesmo valor.
- Os terminais S e T são criados dentro de `corte_minimo`, não no grafo recebido.
- **Rótulos:** `True` = evadiu (`Dropout`), `False` = formou (`Graduate`).

Fronteira entre os módulos:

```python
grafo_similaridade.construir_knn(X, k, pesos) -> dict[int, dict[int, float]]
corte_minimo.classificar(grafo, rotulos_treino, idx_teste) -> dict[int, bool]
#   rotulos_treino: dict[int, bool]   idx_teste: lista de vértices sem rótulo
```

Parâmetros novos (como o viés λ) entram como argumentos opcionais com valor padrão, sem quebrar essas assinaturas.

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
