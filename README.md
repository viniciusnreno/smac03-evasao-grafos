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

Requisitos: Python 3.12 e Git. O repositório é privado: peça ao dono para ser adicionado como colaborador.

```bash
git clone git@github.com:viniciusnreno/smac03-evasao-grafos.git
cd smac03-evasao-grafos
python -m venv .venv             # cria o ambiente virtual em .venv/
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

O ambiente virtual fica em `.venv/`, que o git ignora. Em cada terminal novo, ative-o de novo (`source .venv/bin/activate`) antes de rodar os comandos abaixo.

## Execução

Todos os comandos partem da raiz do repositório, com o ambiente virtual ativado.

### Abrir a interface (Streamlit)

```bash
streamlit run src/app.py
```

O navegador abre em `http://localhost:8501` (porta padrão do Streamlit). A interface lê os resultados de `outputs/`, que já vêm versionados, então não é preciso rodar o pipeline antes. Para encerrar, use `Ctrl+C` no terminal.

### Rodar o pipeline

```bash
python src/main.py            # completo
python src/main.py --rapido   # sem ablação, backward e robustez (cerca de 2 min)
```

Roda G1, G2 e G3, avalia no teste uma única vez e grava os resultados em `outputs/`, sobrescrevendo os arquivos versionados. O modo `--rapido` não regrava os arquivos da ablação, do backward e da robustez.

### Gerar as figuras do relatório

```bash
python src/figuras.py         # PDFs vetoriais em relatorio/figs/
```

Usa os resultados de `outputs/` e regrava `outputs/fatores_*`.

## Estrutura

```
data/
└── data.csv                    # dataset UCI 697 (separador ";")
src/
├── grafo.py                    # classe Grafo: lista de adjacência com peso, dirigida ou não
├── modelo.py                   # classe abstrata ModeloGrafo (construir, executar, destaques)
├── dados.py                    # BaseDeDados: carga, filtro de Enrolled, divisão 80/20 e momentos
├── grafo_similaridade.py       # G1: grafo de similaridade (Gower + kNN)
├── corte_minimo.py             # G1: Edmonds–Karp e classificação pelo corte mínimo
├── arvore_dependencias.py      # G2: árvore geradora máxima de informação mútua (Kruskal)
├── dag_trajetorias.py          # G3: DAG de trajetórias (Kahn + Bellman-Ford)
├── avaliacao.py                # Avaliador: CV, teste, bootstrap, McNemar com Holm e ablação
├── fatores.py                  # AnaliseFatores: lift e informação mútua dos fatores
├── figuras.py                  # GeradorFiguras: figuras do relatório em PDF vetorial
├── main.py                     # Pipeline: roda G1, G2, G3 e a avaliação, grava em outputs/
├── app.py                      # interface gráfica em Streamlit
├── visual.py                   # desenho dos grafos na interface (vis-network)
├── passo_a_passo.py            # Kruskal passo a passo na interface
└── explicacao.py               # explicação das arestas do G1 na interface
outputs/                        # resultados do pipeline (CSV e JSON), lidos pela interface
relatorio/                      # projeto LaTeX do relatório
slides/                         # apresentação (apresentacao.pptx e gerar_slides.js)
.streamlit/config.toml          # tema da interface
requirements.txt                # dependências, com versões fixas
```
