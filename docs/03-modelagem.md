# Modelagem

## Pergunta

Quais características, em que combinação e em que momento da trajetória, distinguem quem evade de quem se forma? Uma representação em grafo aprendida com 80% dos alunos identifica os outros 20% melhor que indicadores isolados?

## Hipóteses

| | Hipótese | Como se testa |
|---|---|---|
| H1 | Existe uma característica dominante, e ela é acadêmica | Informação mútua por variável; melhor regra de 1 variável aprendida no treino e aplicada no teste |
| H2 | A direção de cada fator aprendida no treino se mantém no teste | % de efeitos significativos com o mesmo sinal no teste; Spearman do ranking de IM |
| H3 | Parte dos parâmetros pode sair sem perda | Ablação por grupo e eliminação backward, em CV dentro do treino |
| H4 | O grafo supera indicadores isolados | Acurácia balanceada do corte mínimo × regra de 1 variável, McNemar com Holm |

Associação, não causalidade.

## Protocolo 80/20

1. Remover os `Enrolled` (3.630 alunos).
2. Split estratificado 80/20, **semente 42**: 2.904 treino / 726 teste.
3. Toda escolha (faixas, pesos, k, λ, limiares, variáveis) é feita **só no treino**, com CV estratificada de 5 dobras.
4. O teste é usado **uma única vez**, no fim.
5. Robustez: repetir com 10 splits 80/20 e reportar média ± desvio.

## Os três grafos

### G1 · Similaridade entre alunos + corte mínimo (núcleo)

- **Vértices:** alunos do treino (rotulados) e do teste (sem rótulo), mais uma super-fonte S e um super-sumidouro T.
- **Arestas:** kNN pela distância de Gower; capacidade c(i, j) = c(j, i) = 1 − d(i, j).
- **Distância de Gower:** média ponderada por atributo de |xᵢ − xⱼ| / amplitude (numéricos) ou [xᵢ ≠ xⱼ] (nominais). Peso de cada atributo = informação mútua com o desfecho, calculada no treino.
- **Terminais:** S → evadidos do treino e formados do treino → T, com capacidade ∞.
- **Algoritmo:** fluxo máximo por **Edmonds–Karp** (Ford–Fulkerson com BFS, aula 6). Uma **BFS** na rede residual final dá o lado de S. Aluno do teste no lado de S é previsto como evasão.
- **Otimização:** os rotulados de cada classe podem ser contraídos em S e T sem mudar o corte (no T0: de 3.632 vértices e 128.940 arcos para 728 e 6.306).
- **Hiperparâmetros (escolhidos na CV):** k = 25 em T0 e T1, k = 40 em T2; viés λ = 1,0.
- O teorema fluxo máximo = corte mínimo não está nos slides da aula 6: citar Cormen et al., cap. 26, e Blum e Chawla (2001).

### G2 · Árvore de dependências entre fatores

- **Vértices:** atributos discretizados e o vértice Desfecho.
- **Arestas:** grafo completo com peso = informação mútua entre os dois (só treino).
- **Algoritmo:** **Kruskal** em ordem decrescente com union-find (aula 5), gerando a árvore geradora máxima (árvore de Chow e Liu, 1968).
- **Leitura:** vizinhos diretos do Desfecho são os fatores com informação própria (em T1: aprovação no S1, mensalidade, bolsa).

### G3 · DAG de trajetórias

- **Camadas:** L0 bolsa × faixa etária → L1 mensalidade × devedor → L2 aprovação S1 → L3 aprovação S2 → L4 desfecho. 23 vértices e 76 arcos no treino.
- **Arcos:** transições observadas, com contagem e P(próximo | atual).
- **Algoritmos:** ordenação topológica de **Kahn** (aula 3) e **Bellman-Ford / Dijkstra** com peso −ln p (aula 4) para a trajetória mais provável até a evasão.
- **Como preditor:** o risco do aluno é a taxa de evasão do caminho dele no treino.

### Fora do relatório

G4 (grafo de perfis no hipercubo Q₆, BFS e Dijkstra de intervenção) fica como trabalho futuro. **Louvain foi descartado** a pedido do professor: não usa o desfecho, não prova nada no 80/20 e não está na ementa.

## Métricas

| Métrica | Uso |
|---|---|
| Acurácia balanceada | **Principal**. Média de recall da evasão e especificidade. A acurácia simples engana: "sempre forma" acerta 61% |
| Recall, precisão e F1 da evasão | Complementares |
| AUC | Quando o método devolve uma nota de risco contínua |
| IC 95% | Bootstrap no teste, 1.000 reamostras |
| McNemar exato + Holm | Comparar métodos nos mesmos alunos |
| 10 splits | Robustez (média ± dp) |
| Lift, informação mútua, V de Cramér | Análise de fatores |
| Assortatividade de Newman | Homofilia do desfecho no kNN (k = 10, treino), com teste de permutação (200) |

Comparadores na tabela de resultados: classe majoritária, regra de 1 variável, caminho no DAG, corte mínimo e regressão logística (só como referência externa, não faz parte da solução).
