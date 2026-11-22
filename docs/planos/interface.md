# Plano da interface gráfica

Bônus do protótipo e base da demonstração em vídeo (etapa 18). Decidido em 22/11: Streamlit, para usar as classes do projeto diretamente, com todas as telas abaixo. O Kruskal (`arvore_dependencias.py`, do Carlos) **não é alterado**.

## Como rodar

```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py          # se outputs/ ainda não existir
streamlit run src/app.py
```

A primeira abertura de cada momento do G1 leva cerca de 3 a 4 s (kNN de Gower); depois fica em cache. Funciona sem internet: o vis-network vai embutido no HTML.

## Arquivos

| Arquivo | Dono | Papel |
|---|---|---|
| `src/app.py` | Rafael | Só a tela: páginas, controles, cache. Sem cálculo próprio. |
| `src/visual.py` | Rafael | `MontadorVisual`: `Grafo` + destaques → vértices e arestas do vis-network e o HTML. Não importa o Streamlit. |
| `src/passo_a_passo.py` | Rafael | `DemoEdmondsKarp` e `DemoKruskal`: passos para as animações. Não importa o Streamlit. |
| `src/corte_minimo.py` | Rafael | Argumento opcional `registrar=False` no `EdmondsKarp` e no `ClassificadorCorteMinimo` (histórico de caminhos aumentantes). Sem mudança de comportamento quando desligado. |
| `requirements.txt` | comum | `streamlit==1.67.0` e `pyvis==0.3.2`. |

## Páginas

| Página | Conteúdo | Origem dos números |
|---|---|---|
| Visão geral | Dados, os três grafos e o que cada algoritmo garante, acurácia balanceada com IC 95%, McNemar com Holm | `outputs/avaliacao_teste.csv`, `avaliacao_mcnemar.csv` |
| G1 · Corte mínimo | Ego-rede de um aluno de teste: vizinhos no kNN, rótulos, lado do corte, arestas cortadas; filtro por VP/VN/FP/FN | modelo refeito + `resultados.json` |
| Edmonds–Karp passo a passo | Rede com 8 rotulados e 6 sem rótulo do treino; caminho da BFS, gargalo, fluxo acumulado, corte final | `DemoEdmondsKarp` |
| Exploração (só treino) | k e λ livres numa dobra de validação do treino; histórico da sessão | CV no treino; referência de `avaliacao_cv.csv` |
| G2 · Árvore | Árvore de IM; espessura = IM, opacidade = estabilidade; vizinhos do Desfecho | modelo refeito + `g2_arvore_T*.csv` |
| Kruskal passo a passo | Arestas em ordem decrescente, aceita ou rejeitada (ciclo), componentes do union-find | `DemoKruskal` |
| G3 · DAG | Camadas, ordem de Kahn, trajetórias de maior risco e mais segura, ponto de virada | modelo refeito + `resultados.json` |
| Simulador | O usuário monta um caminho e vê a evasão no treino em cada trecho | `DagTrajetorias` + estados do treino |

## Regras do projeto na interface

- **Sem vazamento:** a visão oficial usa os hiperparâmetros de `outputs/hiperparametros.json`. Só a página de exploração muda k e λ, e roda numa dobra do treino. As demonstrações usam só alunos do treino.
- **Números oficiais:** lidos de `outputs/`. Os grafos refeitos na tela mostram um selo quando batem com `resultados.json`.
- **Kruskal intocado:** `DemoKruskal` lê a árvore devolvida por `kruskal()` e só reconstitui a ordem de exame; o `UniaoBusca` serve para colorir os componentes e conferir que cada rejeitada fecharia ciclo.
- **networkx:** não é usado na interface. As posições do DAG, do Kruskal e da rede de fluxo são calculadas à mão.

## Verificação feita

| Item | Resultado |
|---|---|
| Edmonds–Karp da demonstração | soma dos gargalos = fluxo = capacidade do corte = corte do networkx, em T0/T1/T2 com λ = 0 e 1 |
| Kruskal passo a passo | arestas aceitas = saída de `kruskal()`; 1 componente no fim; toda rejeitada fecharia ciclo (T0, T1, T2) |
| Selos | G1 (fluxo e arestas), G2 (peso total) e G3 (trajetórias) conferem com `resultados.json` |
| App | as 8 páginas abrem sem exceção (`streamlit.testing.v1.AppTest`); interações testadas (passos, filtros, simulador, exploração) |
| Estilo | PEP 8 (pycodestyle) e pyflakes sem avisos nos arquivos novos |

## Roteiro sugerido para o vídeo (cerca de 3 min)

1. Visão geral: a pergunta e a Tab. 3.
2. G1: um aluno de teste acertado e um falso negativo; mostrar as arestas cortadas.
3. Edmonds–Karp passo a passo: 3 ou 4 caminhos e depois o corte final (fluxo = corte).
4. G2: vizinhos do Desfecho em T0 e T1; Kruskal com "pular as rejeitadas".
5. G3: trajetória de risco (57 de 57) e a segura; simulador com 0% de aprovação no S1.
