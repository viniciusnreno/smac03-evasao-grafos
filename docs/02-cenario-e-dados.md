# Cenário e dados

## O tema (Cenário 3)

**"Trajetórias Acadêmicas: Modelagem da Evasão e do Sucesso Estudantil."**

Coordenações têm recursos limitados e precisam decidir quem acompanhar, quais sinais anunciam o abandono e que ações funcionam. Os relatórios atuais olham cada variável separadamente (nota, renda, reprovações) e não mostram como os fatores se combinam. O desafio é representar os estudantes como grafo para revelar perfis parecidos, dependências entre fatores e trajetórias que levam à evasão, e validar isso com métricas quantitativas.

## O dataset

[UCI 697 – Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (Realinho et al., 2022). Instituto Politécnico de Portalegre, Portugal, 17 cursos, coortes de 2008/09 a 2018/19.

Arquivo: `data/data.csv`, 4.424 linhas, 36 atributos + `Target`. Sem valores ausentes nem duplicatas.

| `Target` | Alunos | Uso no projeto |
|---|---|---|
| `Graduate` | 2.209 | classe "formou" |
| `Dropout` | 1.421 | classe "evadiu" (positiva) |
| `Enrolled` | 794 | **removidos**: ainda não têm desfecho |

Após o filtro: 3.630 alunos, 39,15% de evasão.

### Grupos de atributos

| Grupo | Exemplos |
|---|---|
| Demográfico | estado civil, gênero, idade no ingresso, nacionalidade, internacional, deslocado, necessidades especiais |
| Família | escolaridade e ocupação da mãe e do pai |
| Ingresso | modo de ingresso, ordem de opção, curso, turno, formação prévia e nota, nota de admissão |
| Financeiro | devedor, mensalidade em dia, bolsista |
| Acadêmico S1 / S2 | disciplinas creditadas, inscritas, avaliadas, aprovadas, sem avaliação, média |
| Macro | desemprego, inflação, PIB (um valor por coorte) |

### Momentos

| Momento | Atributos disponíveis | Papel |
|---|---|---|
| T0 · ingresso | demográfico, família, ingresso, macro, financeiro (24) | alerta precoce |
| T1 · fim do S1 | T0 + acadêmico S1 (30) | **resultado principal** |
| T2 · fim do S2 | T1 + acadêmico S2 (36) | teto |

Atributo derivado: taxa de aprovação do semestre = aprovadas / inscritas. Com 0 inscritas fica indefinida e vira a faixa "sem inscrição".

## Armadilhas

**De formato:**
- Separador `;` (use `pd.read_csv(path, sep=";")`).
- A coluna `"Daytime/evening attendance\t"` tem um **tab no fim do nome**. Normalize com `df.columns = df.columns.str.strip()`.
- Muitas colunas são códigos (ex.: curso `9119` = Engenharia Informática). O dicionário está no plano v2, seção "Dados e curadoria".

**De conteúdo (o relatório precisa declarar):**
1. **O S2 contém o desfecho.** Quem evadiu no 1º ano aparece com 0 inscritas ou 0 aprovadas no S2. Por isso T2 é teto e T1 é o resultado principal.
2. **Mensalidade e devedor podem ter sido medidos depois do ingresso.** Chamar de "situação financeira registrada".
3. **Macro tem só 10 valores**, um por coorte. Funcionam como código do ano, não como efeito individual. A ablação confirma que não fazem diferença.
4. **Outro país e outra instituição.** Generalização limitada para a UNIFEI.
5. **Só dois semestres.** Não capta evasão tardia.
