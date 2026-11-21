# Glossário

| Termo | Significado no trabalho |
|---|---|
| Vértice | Um aluno (G1), um atributo (G2) ou um estado da trajetória (G3) |
| Aresta / arco | Ligação entre vértices: "são parecidos" (G1), "compartilham informação" (G2), "um estado leva ao outro" (G3) |
| Capacidade | Quanto fluxo uma aresta suporta; no G1 é a similaridade entre dois alunos |
| kNN | Ligar cada aluno aos k mais parecidos com ele |
| Distância de Gower | Distância entre alunos que mistura atributos numéricos e categóricos |
| Super-fonte / super-sumidouro (S, T) | Vértices artificiais ligados a todos os evadidos (S) e a todos os formados (T) do treino |
| Fluxo máximo | Maior quantidade de "fluxo" que vai de S a T respeitando as capacidades |
| Corte mínimo | Conjunto de arestas de menor capacidade total que separa S de T; tem o mesmo valor do fluxo máximo |
| Edmonds–Karp | Ford–Fulkerson que escolhe o caminho aumentante por BFS |
| Rede residual | Capacidade que ainda sobra em cada aresta durante o fluxo |
| Kruskal | Árvore geradora pegando arestas em ordem de peso e evitando ciclos (union-find) |
| Árvore geradora máxima | Árvore que liga todos os vértices com a maior soma de pesos |
| DAG | Grafo dirigido sem ciclos |
| Ordenação topológica (Kahn) | Ordem dos vértices de um DAG em que todo arco vai para a frente |
| Bellman-Ford / Dijkstra | Caminho mínimo; com peso −ln p, o caminho mínimo é a trajetória mais provável |
| Informação mútua (IM) | Quanto, em bits, saber um atributo reduz a incerteza sobre outro |
| Lift | Taxa de evasão do grupo ÷ taxa geral (39,15%) |
| Acurácia balanceada | Média entre "evadidos que o modelo pegou" e "formados que acertou" |
| Recall da evasão | Dos que evadiram, quantos foram previstos |
| AUC | Probabilidade de o modelo dar nota de risco maior a um evadido do que a um formado |
| Bootstrap | Reamostrar o teste com reposição para obter o intervalo de confiança |
| McNemar | Teste que compara dois métodos nos mesmos alunos |
| Holm | Correção dos p-valores quando se fazem vários testes |
| Ablação | Remover variáveis e medir quanto o resultado piora |
| Assortatividade | Se vértices ligados tendem a ter o mesmo desfecho (−1 a 1) |
| Vazamento | Usar informação do teste (ou do futuro) para tomar decisões; invalida a prova |
| CRISP-NET | Roteiro de projeto de dados com atividades de rede, indicado pelo professor |
