# Estado do desenvolvimento

Atualize esta tabela **antes de cada commit**: mude o status da etapa e escreva um resumo de uma linha do que foi feito.

Status: `pendente` · `em andamento` · `concluída` · `bloqueada`

| # | Etapa | Responsável | Prazo | Status | Resumo |
|---|---|---|---|---|---|
| 1 | Migrar o repositório para a estrutura do v3 | Rafael | Sáb 21/11 | concluída | Código v1 removido, esqueletos em `src/`, planos em `docs/planos/` |
| 2 | `CLAUDE.md` e documentação em `docs/` | Rafael | Sáb 21/11 | concluída | Regras, donos, contrato do grafo, PEP 8 + OO, interface comum (`Grafo`, `ModeloGrafo`), regras de sessão, 8 páginas de contexto e README |
| 3 | Subir `analise_v2/` (fonte dos números oficiais) | a definir | Sáb 21/11 | bloqueada | Pasta não está no repositório; descobrir quem tem |
| 3a | `grafo.py` (`Grafo`) e `modelo.py` (`ModeloGrafo`): base comum | Rafael | Sáb 21/11, 17h | concluída | `Grafo` implementado (dirigido ou não, vizinhos só leitura, conversão para networkx) e conferido contra o networkx |
| 4 | `dados.py`: carga, filtro Enrolled, split 80/20, momentos | Rafael | Sáb 21/11, 19h | concluída | Classe `BaseDeDados`; 3.630 alunos, 2.904/726; reproduz os números de treino de `analise_v2` |
| 5 | `grafo_similaridade.py`: Gower + kNN | Rafael (Carlos não iniciou) | Sáb 21/11, 19h | concluída | `GrafoSimilaridade` integrado ao corte mínimo; Gower e kNN conferidos por força bruta; reproduz a IM e a assortatividade (0,27/0,35/0,44) do `analise_v2`; CV no treino: 0,732 / 0,847 / 0,875 |
| 6 | `corte_minimo.py`: Edmonds–Karp próprio + corte por BFS | Rafael | Sáb 21/11, 19h | concluída | Classes `EdmondsKarp` e `ClassificadorCorteMinimo`; conferidas contra o networkx (300 redes + 100 classificações); 0,2 s na escala real. Integrado ao G1 pelo `GrafoSimilaridade.executar()` |
| 7 | `arvore_dependencias.py`: Kruskal + union-find | Carlos | Sáb 21/11, 21h | concluída | `ArvoreDependencias` e `UniaoBusca`; vizinhos do Desfecho e IM iguais ao `analise_v2` (T0: mensalidade, curso, bolsa); peso total confere com o networkx; estabilidade por 100 bootstraps |
| 8 | `dag_trajetorias.py`: Kahn + Bellman-Ford | Rafael (assumiu do Carlos) | Sáb 21/11, 21h | concluída | `Kahn`, `BellmanFord` e `DagTrajetorias`; algoritmos conferidos com o networkx (300 grafos cada); reproduz 23 vértices/76 arcos, trajetória de risco 57/57, segura 13/319 e ponto de virada; CV no treino: 0,731 / 0,817 / 0,823 |
| 9a | Corrigir o sentido do viés λ no corte mínimo | Rafael | Sáb 21/11, 22h | concluída | Viés invertido em relação ao plano v2 (puxava para T); agora S → u = λ(1 − q) e u → T = λq. Na CV do treino, λ = 1 passa a aumentar o recall (T0: 0,568 → 0,591), mas o efeito é bem menor que no oficial (0,606 → 0,820 no teste) |
| 9 | `main.py`: pipeline grava em `outputs/` | Carlos | Sáb 21/11, 21h | concluída | `Pipeline` roda G1 (teste uma vez), assortatividade (0,273/0,352/0,441, p = 0,005), G2, G3 e a avaliação; grava os resultados em `outputs/` e registra o ambiente e o tempo por etapa em `hiperparametros.json`. **Após a etapa 9a: G1 no teste 0,747/0,880/0,913 (oficial 0,763/0,847/0,899); G3 no teste 0,716/0,816/0,836 (oficial 0,716/0,816/0,849). Falta decidir quais números valem** |
| 10 | `avaliacao.py`: CV, teste, bootstrap, McNemar, ablação | Rafael | Sáb 21/11, 19h | concluída | `Avaliador` (CV 5 dobras, teste único com IC bootstrap, McNemar exato + Holm, ablação por grupo, backward, robustez em 10 divisões) + referências (regra de 1 variável, regressão logística); integrado ao `main.py` (flag `--rapido`), grava 7 `avaliacao_*.csv`. Teste: corte 0,747/0,880/0,913; corte × regra e corte × DAG significativos após Holm nos 3 momentos; robustez 0,734±0,017 / 0,855±0,020 / 0,878±0,012 |
| 11 | `figuras.py`: Figs. 3–5 em PDF | Rafael | Sáb 21/11, 21h | concluída | `GeradorFiguras` grava `fig-lift`, `fig-ablacao` e `fig-arvore` (T1) em `relatorio/figs/`; nova `AnaliseFatores` (`fatores.py`) reproduz os 27 lifts do plano v2 e grava `outputs/fatores_*`. Replicação H2: 95,8% (oficial 96,2%), Spearman da IM 0,946 (oficial 0,958) |
| 12 | Tab. 3 de resultados | Rafael | Sáb 21/11, 23h | pendente | |
| 13 | Seção 3.1, Eq. 1, Tab. 1, Fig. 1 | Vinícius | Sáb 21/11, 21h | pendente | |
| 14 | Seção 3.2, Algs. 1–2, Tab. 2, Fig. 2 | Carlos | Sáb 21/11, 23h | concluída | 3.2 reescrita em `3-Desenvolvimento.tex` (3.1 intocada): protocolo, Algs. 1–2 fiéis ao código, Tab. 2 e ambiente a partir de `outputs/`; Fig. 2 no estilo dos veteranos (`relatorio/figs/gerar_fig_protocolo.py`). Faltam no `.bib`: `cormen2009`, `blum2001`, `chow1968` |
| 15 | Seções 1 e 2, `referencias.bib` | Nathan | Sáb 21/11, 23h | pendente | |
| 16 | Seção 4 (texto) | Vinícius | Dom 22/11, 11h | pendente | |
| 17 | Seção 5 e revisão cruzada | Nathan | Dom 22/11, 13h | pendente | |
| 18 | Slides, vídeo da demonstração e ensaio | Nathan, Carlos, Rafael | Dom 22/11, 18h | pendente | |
| 19 | Envio do PDF no SIGAA | Nathan | Dom 22/11, 22h | pendente | |
