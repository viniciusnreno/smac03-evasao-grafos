# Equipe e cronograma

O integrante designado como Modelador B (Gabriel) trancou a disciplina. As tarefas dele foram redistribuídas.

## Papéis

| Pessoa | Papel | Código | Relatório | Slides |
|---|---|---|---|---|
| Nathan | Coordenador | — | Seções 1, 2 e 5; `main.tex`, capa, `referencias.bib`; revisão final e envio no SIGAA | Abertura, limitações e conclusões |
| Vinícius | Modelador | — | Seção 3.1, Eq. 1 (Gower), Tab. 1, Fig. 1 (rede de fluxo); texto da seção 4 | Protocolo, fatores, ablação, árvore |
| Carlos | Desenvolvedor A | `grafo_similaridade`, `arvore_dependencias`, `dag_trajetorias`, `main` | Seção 3.2, Alg. 1–2, Tab. 2, Fig. 2 (protocolo) | Corte mínimo e demonstração |
| Rafael | Desenvolvedor B | `grafo`, `modelo`, `dados`, `corte_minimo` (Edmonds–Karp), `avaliacao`, `figuras` | Figs. 3–5, Tab. 3 | Resultados no teste e momento de intervir |

## Cronograma

| Quando | Ponto de junção |
|---|---|
| Sáb 21/11, 19h | `src/` roda G1, G2 e G3; seções 1 e 3.1 em rascunho |
| Sáb 21/11, 23h | Figs. 1–5, Tabs. 1–3 e Algs. 1–2 no Overleaf; seções 2 e 3.2 em rascunho |
| Dom 22/11, 13h | Relatório completo e revisão cruzada |
| Dom 22/11, 18h | Slides, vídeo da demonstração, ensaio de 15 min |
| Dom 22/11, 22h | PDF enviado no SIGAA (prazo oficial 23h59) |

## Convenções do relatório

**Rótulos LaTeX:** `fig:fluxo`, `fig:protocolo`, `fig:lift`, `fig:ablacao`, `fig:arvore`, `fig:dag`, `tab:grafos`, `tab:hiper`, `tab:resultados`, `alg:ek`, `alg:kruskal`, `eq:gower`, `eq:cap`.

**Chaves de referência:** `realinho2022`, `tinto1975`, `gower1971`, `blum2001`, `chow1968`, `cormen2009`, `newman2003`, `acunacid2025`, `porto2025`.

**Figuras:** em `relatorio/figs/`, PDF vetorial, com o nome do rótulo (`fig-lift.pdf`, `fig-ablacao.pdf`, `fig-arvore.pdf`).

**Arquivos LaTeX:** cada pessoa edita só a própria seção em `relatorio/elemtextuais/`. `main.tex`, `configuracao.tex` e a capa são do Nathan.

## Slides

Padrão assertion-evidence: o título é uma frase-afirmação e o corpo é a figura que a prova. Sem slides de tópicos. Roteiro detalhado no [plano v3](planos/plano-v3.html), seção "Apresentação".
