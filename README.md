# Trajetórias Acadêmicas: Modelagem da Evasão e do Sucesso Estudantil com Grafos

Projeto da disciplina SMAC03 – Grafos (UNIFEI, 2026/2), Cenário 3.
Dataset: [UCI 697 – Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (4.424 estudantes, 36 atributos, desfecho Dropout/Enrolled/Graduate).

## Modelos

| Modelo | Vértices | Arestas | Pergunta que responde |
|---|---|---|---|
| **1. Rede de Similaridade Multirrelacional** | estudantes | k-NN (k=15) por similaridade `R = α·S_ACAD + β·S_SOCIO + γ·S_DEMO` (α=0,5; β=0,3; γ=0,2) | Quais perfis se agrupam e com que desfecho? Quais perfis parecidos divergem? |
| **2. Rede de Fatores** | fatores (atributo=valor) + 3 nós de desfecho | co-ocorrência com lift ≥ 1,15 | Como os fatores se entrelaçam e que constelação cerca a evasão? |

O desfecho **não** é usado para construir o Modelo 1 – serve só para avaliar a estrutura.

## Análises
- Métricas estruturais, assortatividade e homofilia pelo desfecho
- Detecção de comunidades: Louvain (Multilevel), Fast Greedy, Label Propagation – modularidade, % arestas inter-comunidades, NMI/ARI
- Perfis de comunidade (fatores sobre-representados) e taxa de evasão por comunidade
- Pares com perfil de ingresso quase idêntico e desfechos opostos (o que os separa)
- Escore de risco relacional (voto dos vizinhos, propagação de rótulos) validado por AUC em validação cruzada 5-fold, comparado a regressão logística e à melhor variável isolada; cenário de alerta precoce (só dados de ingresso)
- Sensibilidade a α, β, γ e k
- Rede de fatores: centralidade, comunidades, vizinhança do nó Evasão, sinergia entre pares de fatores

## Instalação

Requisitos: Python 3.10+ e Git. O repositório é privado: peça ao dono para ser adicionado como colaborador.

```bash
git clone git@github.com:viniciusnreno/smac03-evasao-grafos.git
cd smac03-evasao-grafos
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Como executar

1. **Pipeline completo**: gera tabelas, figuras e os artefatos usados pelo explorador.
   ```bash
   python -m src.main            # ~4 min; use --rapido para pular a análise de sensibilidade
   ```
   Saídas em `outputs/`: `tabelas/*.csv`, `figuras/*.png`, `resumo.json` e `artefatos.pkl`.

2. **Explorador interativo** (demo): depende do `outputs/artefatos.pkl` gerado no passo 1, que não é versionado.
   ```bash
   streamlit run app/explorador.py
   ```

Para testar outros pesos ou outro `k`:
```bash
python -m src.main --alpha 0.3 --beta 0.5 --gamma 0.2 --k 25 --out outputs_alt
```

### Relatório e slides
- **Relatório**: o projeto LaTeX está em `relatorio/`. Envie a pasta para o Overleaf e compile com pdfLaTeX. `relatorio/relatorio_preview.pdf` é uma prévia compilada.
- **Slides**: edite `slides/apresentacao.pptx` direto no PowerPoint ou no Google Slides. O `slides/gerar_slides.js` foi usado só para gerar a primeira versão.

## Estrutura
```
data/data.csv         dataset original
src/preprocess.py     discretização e agrupamento dos atributos (ACAD/SOCIO/DEMO)
src/graphs.py         construção dos Modelos 1 e 2
src/analysis.py       métricas, comunidades, validação, fatores
src/plots.py          figuras
src/main.py           pipeline
relatorio/            relatório LaTeX (template IMC/UNIFEI)
```
