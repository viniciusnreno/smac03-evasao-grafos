"""G1: grafo de similaridade. Classe: ``GrafoSimilaridade(ModeloGrafo)``.

Responsável: Carlos.

- ``construir``: distância de Gower ponderada pela informação mútua
  (só treino) e kNN com capacidade c(i, j) = 1 - d(i, j).
- ``executar``: chama ``CorteMinimo`` e devolve as previsões.
- ``destaques``: arestas do corte e alunos de cada lado.
"""
