# Resultados oficiais

**Esta é a única fonte de números para o relatório e os slides.** Nada é digitado de memória.

Origem: análise `analise_v2/` (split estratificado 80/20, semente 42), descrita no [plano v2](planos/plano-v2.html). Essa pasta **ainda não está no repositório**. Quando `src/main.py` regenerar os resultados em `outputs/`, compare com esta página. Se algo mudar, atualize aqui e avise o grupo.

## Dados e fatores

| Item | Valor |
|---|---|
| Alunos após remover os 794 Enrolled | 3.630 |
| Evasão (Dropout × Graduate) | 39,15% (1.421 × 2.209) |
| Treino / teste | 2.904 / 726 |
| Evasão com 0% de aprovação no S1 (inscritos) | 99,5% (n = 387) |
| Evasão de quem aprova 100% no S1 | 10,3% |
| Evasão com mensalidade atrasada | 94,0% (lift ×2,40; teste ×2,41) |
| Lift: bolsista | ×0,35 (teste ×0,38) |
| Lift: devedor · idade ≥ 28 · masculino | ×1,94 · ×1,62 · ×1,45 |
| Direções significativas que se repetem no teste | 96,2% |
| Spearman do ranking de IM, treino × teste | 0,958 |
| IM com o desfecho: aprovação S2 · S1 · mensalidade | 0,474 · 0,381 · 0,149 bits |
| Assortatividade do desfecho no kNN (T0 / T1 / T2) | 0,27 / 0,35 / 0,44 (permutação p = 0,005) |

## Teste 20% (acurácia balanceada)

| Método | T0 | T1 | T2 |
|---|---|---|---|
| Classe majoritária | 0,500 | 0,500 | 0,500 |
| Regra de 1 variável | 0,664 | 0,825 | 0,860 |
| Caminho no DAG (G3) | 0,716 | 0,816 | 0,849 |
| **Corte mínimo (G1)** | **0,763** | **0,847** | **0,899** |
| IC 95% do corte | 0,732–0,793 | 0,820–0,873 | 0,874–0,922 |
| F1 / recall da evasão (corte) | 0,720 / 0,820 | 0,813 / 0,835 | 0,877 / 0,877 |
| AUC (corte) | 0,852 | 0,937 | 0,965 |
| Regressão logística (referência) | 0,768 | 0,884 | 0,924 |
| McNemar corte × regra isolada (p Holm) | 0,012 | 0,41 (empate) | 0,0007 |
| 10 splits, corte (média ± dp) | 0,760 ± 0,015 | 0,849 ± 0,010 | 0,882 ± 0,010 |
| Corte só com variáveis essenciais | 0,740 (8 var.) | 0,858 (3 var.) | 0,893 (7 var.) |

Matriz de confusão do corte em T0: TP 233, FP 130, FN 51, TN 312.

Regras de 1 variável aprendidas no treino: T0, idade no ingresso ≥ 23; T1, aprovação no S1 < 78%; T2, aprovação no S2 < 79%.

Pendente: o plano não diz de onde vem a nota contínua usada na AUC do corte mínimo. Confirmar ao implementar.

## As três frases-resumo

1. A principal característica de quem evade é a **reprovação no 1º semestre**. Antes de existir nota, o sinal mais forte é **financeiro**.
2. O que aprendemos com 80% dos alunos **se confirmou nos 20%**: 96% das direções se repetem, e a maior parte dos 36 parâmetros pode sair sem perda.
3. No ingresso, quando nenhuma variável sozinha resolve, o **corte mínimo no grafo de similaridade** acerta 0,76 contra 0,66 (+10 p.p., significativo). O momento de intervir é o **fim do 1º semestre**.
