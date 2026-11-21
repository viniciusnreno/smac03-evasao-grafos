# Evasão Estudantil como Problema de Grafos

Projeto da disciplina SMAC03 – Grafos (UNIFEI, 2026/2), Cenário 3.
Dataset: [UCI 697 – Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

O plano completo está em [docs/plano-v3.html](docs/plano-v3.html) (plano de ação) e [docs/plano-v2.html](docs/plano-v2.html) (análise técnica).

## Grafos

| Grafo | Algoritmo | Pergunta |
|---|---|---|
| G1 · similaridade entre alunos | Edmonds–Karp (corte mínimo) | Quem está em risco de evasão? |
| G2 · dependências entre fatores | Kruskal (árvore geradora máxima) | Como os fatores dependem uns dos outros? |
| G3 · trajetórias | Kahn + Bellman-Ford | Por quais caminhos se chega à evasão? |

Protocolo: alunos "Enrolled" removidos; treino/teste 80/20 estratificado (semente 42); toda escolha é feita só no treino e o teste é usado uma única vez.

## Instalação

```bash
git clone git@github.com:viniciusnreno/smac03-evasao-grafos.git
cd smac03-evasao-grafos
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

```bash
python src/main.py
```

Em construção: os módulos de `src/` ainda são esqueletos.

## Estrutura

```
data/data.csv
src/
├── dados.py                # Carlos
├── grafo_similaridade.py   # Carlos
├── corte_minimo.py         # Rafael (Edmonds–Karp próprio)
├── arvore_dependencias.py  # Carlos (Kruskal)
├── dag_trajetorias.py      # Carlos (Kahn + Bellman-Ford)
├── avaliacao.py            # Rafael
├── figuras.py              # Rafael
└── main.py                 # Carlos
outputs/                    # resultados e figuras gerados
docs/                       # planos v2 e v3
relatorio/                  # projeto LaTeX (Overleaf)
slides/
```
