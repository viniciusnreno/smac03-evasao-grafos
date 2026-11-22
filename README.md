# Evasão Estudantil como Problema de Grafos

Projeto da disciplina SMAC03 – Grafos (UNIFEI, 2026/2), Cenário 3.
Dataset: [UCI 697 – Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

## Grafos

| Grafo | Algoritmo | Pergunta |
|---|---|---|
| G1 · similaridade entre alunos | Edmonds–Karp (corte mínimo) | Quem está em risco de evasão? |
| G2 · dependências entre fatores | Kruskal (árvore geradora máxima) | Como os fatores dependem uns dos outros? |
| G3 · trajetórias | Kahn + Bellman-Ford | Por quais caminhos se chega à evasão? |

Protocolo: alunos "Enrolled" removidos; treino/teste 80/20 estratificado (semente 42); toda escolha é feita só no treino e o teste é usado uma única vez.

## Instalação

Requisitos: Python 3.14 (testado com 3.14.4) e Git. O repositório é privado: peça ao dono para ser adicionado como colaborador.

```bash
git clone git@github.com:viniciusnreno/smac03-evasao-grafos.git
cd smac03-evasao-grafos
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

```bash
python src/main.py            # pipeline completo, grava em outputs/ (~2h10)
python src/main.py --rapido   # sem ablação, backward e robustez (~2 min)
python src/figuras.py         # fatores_* em outputs/ e Figs. 3–5 do relatório
streamlit run src/app.py      # interface gráfica (lê outputs/)
```

O pipeline confere cada algoritmo de grafo com o networkx em asserts (Edmonds–Karp, Kruskal, Kahn e Bellman-Ford). Os números do relatório vêm de `outputs/`.

## Estrutura

```
data/data.csv
src/
├── grafo.py                # classe Grafo (base comum)
├── modelo.py               # classe abstrata ModeloGrafo (base comum)
├── dados.py                # Rafael
├── grafo_similaridade.py   # Carlos
├── corte_minimo.py         # Rafael (Edmonds–Karp próprio)
├── arvore_dependencias.py  # Carlos (Kruskal)
├── dag_trajetorias.py      # Carlos (Kahn + Bellman-Ford)
├── avaliacao.py            # Rafael
├── fatores.py              # Rafael (lift e informação mútua)
├── figuras.py              # Rafael
├── main.py                 # Carlos
├── app.py                  # Rafael (interface Streamlit)
├── visual.py               # Rafael (desenho dos grafos)
└── passo_a_passo.py        # Rafael (Edmonds–Karp e Kruskal passo a passo)
outputs/                    # resultados e figuras gerados
relatorio/                  # projeto LaTeX (Overleaf)
slides/
```

**Atenção:** a versão final do relatório é editada no Overleaf. Em `relatorio/`, só a seção 3.2 e as figuras estão atualizadas; as demais seções e `slides/` ainda trazem o conteúdo do modelo v1 (Louvain e comunidades), que foi descartado.
