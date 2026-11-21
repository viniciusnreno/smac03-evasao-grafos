# Documentação do projeto

Contexto do trabalho de SMAC03 (Grafos, UNIFEI 2026/2), Cenário 3: evasão estudantil.
Os arquivos estão em ordem de leitura. Quem chega agora (pessoa ou Claude) deve ler pelo menos 03 e 04.

| Arquivo | Conteúdo |
|---|---|
| [01-disciplina.md](01-disciplina.md) | O que a disciplina exige: entregas, prazos, estrutura do relatório, papéis |
| [02-cenario-e-dados.md](02-cenario-e-dados.md) | O tema, o dataset UCI 697 e as armadilhas dos dados |
| [03-modelagem.md](03-modelagem.md) | Pergunta, hipóteses, protocolo 80/20, os três grafos, algoritmos e métricas |
| [04-arquitetura.md](04-arquitetura.md) | Estrutura do código, donos dos módulos, contrato do grafo, tecnologias, como rodar |
| [05-equipe-e-cronograma.md](05-equipe-e-cronograma.md) | Quem faz o quê, prazos e convenções do relatório e dos slides |
| [06-resultados-oficiais.md](06-resultados-oficiais.md) | Números oficiais: a única fonte para citar no relatório e nos slides |
| [estado.md](estado.md) | Estado atual de cada etapa: atualizado antes de cada commit |
| [glossario.md](glossario.md) | Termos de grafos e de avaliação usados no trabalho |
| [planos/plano-v3.html](planos/plano-v3.html) | Plano de ação final (fonte destas páginas) |
| [planos/plano-v2.html](planos/plano-v2.html) | Análise técnica completa, com gráficos interativos (anexo) |

## Histórico de decisões

- **v1 (descartado):** rede de similaridade com Louvain e rede de fatores. Usava os alunos "Enrolled" e não tinha teste separado.
- **v2:** após consultoria com o professor. Remove os "Enrolled", separa 80/20, faz ablação e troca o Louvain por algoritmos vistos em aula.
- **v3 (vigente):** enxuga o v2 para 3 grafos e 4 hipóteses e divide as tarefas.

Quando estas páginas e os HTML divergirem, vale o que está em `docs/*.md`. Atualize-as quando uma decisão mudar.
