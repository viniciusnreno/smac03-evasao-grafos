# CLAUDE.md

Trabalho de SMAC03 (Grafos, UNIFEI 2026/2), Cenário 3: evasão estudantil modelada como grafos, com o dataset UCI 697. Entrega em **22/11/2026, 23h59**: relatório de até 9 páginas, slides e este código.

Antes de qualquer tarefa, leia [docs/README.md](docs/README.md). O mínimo é [docs/03-modelagem.md](docs/03-modelagem.md), [docs/04-arquitetura.md](docs/04-arquitetura.md) e [docs/estado.md](docs/estado.md).

## Como trabalhar nas sessões

1. **Início de sessão.** Na primeira resposta de cada sessão, leia [docs/estado.md](docs/estado.md) e traga:
   - um resumo do projeto em 2 ou 3 linhas;
   - o estado atual em tabela (etapa, responsável, prazo, status), com as etapas pendentes, em andamento e bloqueadas;
   - o que vence a seguir.
2. **Antes de cada commit.** Atualize [docs/estado.md](docs/estado.md): mude o status da etapa para `concluída` (ou `em andamento`) e escreva um resumo de uma linha. Inclua essa alteração no mesmo commit. Se a tarefa não estiver na tabela, adicione uma linha.
3. **Estilo das respostas.** Linguagem clara, sucinta e objetiva, em português. Resposta primeiro, contexto depois. Use tabelas quando houver comparação ou lista de itens. Termine as respostas com uma seção **"Resumo executivo"** em linguagem simples, citando os nomes dos algoritmos envolvidos.

## O projeto em 5 linhas

- Remove os alunos `Enrolled` e separa treino/teste 80/20 estratificado (semente 42).
- **G1** (núcleo): grafo kNN de similaridade de Gower entre alunos, classificado por **corte mínimo com Edmonds–Karp**.
- **G2**: árvore geradora máxima de informação mútua entre fatores, por **Kruskal**.
- **G3**: DAG de trajetórias (ingresso → financeiro → S1 → S2 → desfecho), com **Kahn** e **Bellman-Ford**.
- Avaliação: acurácia balanceada no teste, IC 95% por bootstrap, McNemar com Holm, ablação por CV.

## Regras invioláveis

1. **Sem vazamento.** Faixas, pesos, k, λ, limiares e seleção de variáveis vêm só do treino (CV de 5 dobras). O teste é usado uma única vez, no fim.
2. **Algoritmos de grafo implementados à mão**: Edmonds–Karp, BFS do corte, Kruskal com union-find, Kahn, Bellman-Ford. O networkx serve **só** para conferir o resultado em asserts.
3. **Sem Louvain, modularidade ou comunidades.** O professor pediu algoritmos vistos em aula.
4. **Sem ML como solução.** Regressão logística aparece apenas como referência externa.
5. **Semente 42** em todo sorteio.
6. **Números do relatório e dos slides** vêm de `outputs/` ou de [docs/06-resultados-oficiais.md](docs/06-resultados-oficiais.md). Nunca digitar de memória.
7. Cada algoritmo é descrito pelo que ele **garante** (Kruskal gera árvore geradora, não caminho mínimo).

## Donos dos arquivos

Edite só os arquivos da pessoa que pediu a tarefa. Para mexer em arquivo de outra pessoa, avise antes.

| Arquivo | Dono |
|---|---|
| `src/grafo.py`, `modelo.py` (base comum), `dados.py` | Rafael |
| `src/grafo_similaridade.py`, `arvore_dependencias.py`, `dag_trajetorias.py`, `main.py` | Carlos |
| `src/corte_minimo.py`, `avaliacao.py`, `figuras.py` | Rafael |
| `src/app.py`, `visual.py`, `passo_a_passo.py` (interface) | Rafael |
| `relatorio/main.tex`, `configuracao.tex`, capa, `referencias.bib`, seções 1, 2 e 5 | Nathan |
| `relatorio/elemtextuais/3-Desenvolvimento.tex` | 3.1 Vinícius · 3.2 Carlos |
| `relatorio/elemtextuais/4-Resultados.tex` | Vinícius (texto) e Rafael (figuras e tabela) |

## Contrato do grafo

Duas classes base, em `src/`:

- **`Grafo`** (`grafo.py`, concreta): lista de adjacência com peso, `dict[vértice, dict[vértice, float]]`, dirigida ou não. Usada pelos três modelos. Grafo não dirigido grava os dois sentidos, o que garante o kNN simétrico. `para_networkx()` existe só para os asserts.
- **`ModeloGrafo`** (`modelo.py`, abstrata): todo modelo implementa `construir(base) -> Grafo`, `executar() -> dict` e `destaques() -> dict`. É o que o `Pipeline` e a futura interface gráfica usam.

```python
class GrafoSimilaridade(ModeloGrafo)    # G1 · Carlos
class ArvoreDependencias(ModeloGrafo)   # G2 · Carlos (Kruskal)
class DagTrajetorias(ModeloGrafo)       # G3 · Carlos (Kahn + Bellman-Ford)

# Fronteira Carlos ↔ Rafael (G1)
GrafoSimilaridade(momento, k, pesos=None).construir(base) -> Grafo
ClassificadorCorteMinimo(lam=1.0).classificar(grafo, rotulos_treino, idx_teste) -> dict[int, bool]
```

- `corte_minimo.py` tem duas classes de algoritmo (não são modelos): `EdmondsKarp`, fluxo máximo genérico, e `ClassificadorCorteMinimo`, que monta a rede com S e T e chama o `EdmondsKarp`. `GrafoSimilaridade.executar()` usa o classificador.
- No G1, vértice = posição do aluno no DataFrame após remover `Enrolled` e `reset_index`. No G2 e no G3, vértices são `str` (atributos e estados).
- S e T são criados dentro de `ClassificadorCorteMinimo`, não no grafo recebido.
- Rótulo `True` = evadiu, `False` = formou.
- Tudo o que for específico de um modelo (previsões, métricas) volta no dict de `executar()`; não crie métodos abstratos novos sem combinar.
- Parâmetros novos entram como argumentos opcionais, sem quebrar as assinaturas.

## Dados

`data/data.csv`: separador `;`, e a coluna `"Daytime/evening attendance\t"` tem um tab no nome. Normalize com `df.columns = df.columns.str.strip()`. Detalhes e armadilhas em [docs/02-cenario-e-dados.md](docs/02-cenario-e-dados.md).

## Comandos

```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py          # pipeline completo, grava em outputs/
streamlit run src/app.py    # interface gráfica (bônus), lê outputs/
```

## Padrões de código

- **PEP 8 sempre**: 4 espaços, linhas de até 79 caracteres, `snake_case` para funções, métodos e variáveis, `CamelCase` para classes, `MAIUSCULAS` para constantes, imports no topo (padrão, terceiros, locais).
- **Orientação a objetos**: cada módulo de `src/` expõe uma classe principal com métodos bem definidos (tabela em [docs/04-arquitetura.md](docs/04-arquitetura.md)). Os três modelos herdam de `ModeloGrafo` (`abc.ABC`). Hiperparâmetros entram no `__init__`; os dados entram nos métodos.
- **Lógica separada da interface**: as classes não imprimem, não plotam e não leem arquivos fora do que é a responsabilidade delas. Elas devolvem dados. Isso permite que uma interface gráfica (bônus, fase futura) use as mesmas classes que o pipeline.
- Métodos auxiliares internos começam com `_` (ex.: `_bfs`).

## Convenções

- Código, nomes, docstrings e mensagens em português, seguindo os nomes do plano (`corte_minimo`, `taxa_aprov_s1`).
- Docstring de cada algoritmo cita a aula de origem e a garantia (ex.: "Edmonds–Karp, aula 6: fluxo máximo em O(V·E²)").
- Figuras em PDF vetorial; rótulos LaTeX e chaves bibliográficas fixos em [docs/05-equipe-e-cronograma.md](docs/05-equipe-e-cronograma.md).
- Git: uma branch por tarefa a partir da `main`, PR para a `main`, commits em português no imperativo.
- Mudou uma decisão? Atualize o `docs/` correspondente no mesmo commit.
