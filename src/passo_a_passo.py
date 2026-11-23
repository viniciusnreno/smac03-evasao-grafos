"""Demonstração passo a passo do Kruskal.

``DemoKruskal`` roda o algoritmo do projeto (não o reimplementa) e
devolve a lista de passos que a interface anima: lê a árvore devolvida
por ``ArvoreDependencias.kruskal`` e reconstitui a ordem em que as
arestas foram examinadas. A decisão (aceita ou rejeita) vem da saída
oficial; o ``UniaoBusca`` só acompanha os componentes para colorir o
desenho e confirmar que cada rejeitada fecharia um ciclo. Não depende
do Streamlit.
"""

from arvore_dependencias import ArvoreDependencias, UniaoBusca


class DemoKruskal:
    """Kruskal (aula 5) passo a passo, a partir da árvore oficial.

    A ordem de exame é a do ``ArvoreDependencias.kruskal`` com
    ``maximo=True``: IM decrescente, empates pelo nome dos vértices.
    Cada aresta é marcada como aceita se está na árvore devolvida pelo
    Kruskal; para as rejeitadas, confere-se que as pontas já estavam no
    mesmo componente (fecharia um ciclo).

    Garantia mostrada: a árvore final é geradora (liga todos os
    vértices, sem ciclos) e tem a maior soma de IM possível.
    """

    def executar(self, modelo: ArvoreDependencias,
                 arvore: list[tuple[str, str, float]]) -> dict:
        """Passos do Kruskal sobre o grafo completo de ``modelo``.

        Args:
            modelo: ``ArvoreDependencias`` já construído.
            arvore: ``executar()["arestas"]`` do mesmo modelo.

        Devolve ``vertices``, ``passos`` (``u``, ``v``, ``peso``,
        ``aceita``, ``componentes`` depois do passo e ``num_aceitas``)
        e ``num_examinadas``.
        """
        grafo = modelo.grafo
        if grafo is None:
            raise RuntimeError("construa o modelo antes")
        na_arvore = {tuple(sorted((u, v), key=str)) for u, v, _ in arvore}
        arestas = sorted(
            (tuple(sorted((u, v), key=str)) + (p,)
             for u, v, p in grafo.arestas() if u != v),
            key=lambda a: (-a[2], str(a[0]), str(a[1])))

        vertices = sorted(grafo.vertices(), key=str)
        conjuntos = UniaoBusca(vertices)
        passos = []
        aceitas = 0
        for u, v, p in arestas:
            if aceitas == len(arvore):
                break
            aceita = (u, v) in na_arvore
            if aceita:
                if not conjuntos.unir(u, v):
                    raise AssertionError(f"aresta da árvore fecha ciclo: "
                                         f"{u} — {v}")
                aceitas += 1
            elif conjuntos.buscar(u) != conjuntos.buscar(v):
                raise AssertionError(f"aresta rejeitada não fecharia "
                                     f"ciclo: {u} — {v}")
            passos.append({
                "u": u, "v": v, "peso": p, "aceita": aceita,
                "num_aceitas": aceitas,
                "componentes": {x: conjuntos.buscar(x) for x in vertices},
            })
        return {"vertices": vertices, "passos": passos,
                "num_examinadas": len(passos),
                "num_arestas_completo": len(arestas)}
