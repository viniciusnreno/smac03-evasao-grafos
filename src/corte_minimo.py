"""G1: corte mínimo por Edmonds-Karp. Classe: ``CorteMinimo``.

Responsável: Rafael.

Classe de algoritmo (não é um ``ModeloGrafo``): recebe um ``Grafo`` e
os rótulos do treino. Cria a super-fonte S (ligada aos evadidos) e o
super-sumidouro T (ligado aos formados), com capacidade infinita.
Calcula o fluxo máximo por caminhos aumentantes via BFS na rede
residual; o lado de S do corte mínimo é previsto como evasão.
"""
