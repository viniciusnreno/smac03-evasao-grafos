// Gera slides/apresentacao.pptx
// Uso: NODE_PATH=<node_modules com pptxgenjs, react-icons, react, react-dom, sharp> node slides/gerar_slides.js
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(ROOT, "outputs/figuras", f);
const OUT = path.join(__dirname, "apresentacao.pptx");

const HEX = {
  navy: "1F2A44", ink: "1B1B1B", muted: "52514E", light: "EEF1F6", white: "FFFFFF",
  orange: "EB6834", aqua: "1BAF7A", blue: "2A78D6", violet: "6250D6", grid: "D9DCE3",
};
const THEME = {
  name: "Trajetorias em Rede",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: HEX.ink, lt1: HEX.white, dk2: HEX.navy, lt2: HEX.light,
    accent1: HEX.orange, accent2: HEX.aqua, accent3: HEX.blue, accent4: HEX.violet,
    accent5: HEX.muted, accent6: "EDA100", hlink: HEX.blue, folHlink: HEX.violet,
  },
};

async function icon(Comp, color) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
  pres.title = "Trajetórias Acadêmicas em Rede";
  pres.subject = "SMAC03 - Grafos - UNIFEI";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  const W = 13.333, M = 0.6;

  // ------------------------------------------------------------ layouts
  pres.defineSlideMaster({
    title: "CAPA",
    background: { color: C.text2 },
    objects: [],
    placeholders: [],
  });
  pres.defineSlideMaster({
    title: "CONTEUDO",
    background: { color: C.background1 },
    objects: [
      { text: { text: "SMAC03 – Grafos · UNIFEI · 2026/2", options: { x: M, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
      { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 1.0, fontSize: 32,
          bold: true, color: C.text2, fontFace: THEME.headFontFace, valign: "middle", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
  });
  pres.defineSlideMaster({
    title: "ESCURO",
    background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 1.0, fontSize: 32,
          bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "middle", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
  });

  const content = (sec, title) => {
    const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: sec });
    s.addText(title, { placeholder: "title" });
    return s;
  };
  const body = (s, text, o) => s.addText(text, { isTextBox: true, fontSize: 16, color: C.text1, valign: "top", margin: 0, ...o });
  const card = (s, x, y, w, h, fill, name) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
  const imgFit = (s, file, x, y, maxW, maxH, ratio, alt) => {
    let w = maxW, h = maxW / ratio;
    if (h > maxH) { h = maxH; w = maxH * ratio; }
    s.addImage({ path: FIG(file), x: x + (maxW - w) / 2, y, w, h, altText: alt });
  };

  const ic = {
    grad: await icon(fa.FaUserGraduate, HEX.white), net: await icon(fa.FaProjectDiagram, HEX.white),
    users: await icon(fa.FaUsers, HEX.white), chart: await icon(fa.FaChartBar, HEX.white),
    book: await icon(fa.FaBookReader, HEX.white), money: await icon(fa.FaHandHoldingUsd, HEX.white),
    search: await icon(fa.FaSearch, HEX.white), link: await icon(fa.FaLink, HEX.white),
  };
  const iconCircle = (s, x, y, d, fill, data) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
    s.addImage({ data, x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56 });
  };

  // ------------------------------------------------------------ 1. capa
  pres.addSection({ title: "Abertura" });
  let s = pres.addSlide({ masterName: "CAPA", sectionTitle: "Abertura" });
  iconCircle(s, M, 1.0, 1.0, C.accent1, ic.net);
  s.addText("Trajetórias Acadêmicas em Rede", { isTextBox: true, x: M, y: 2.2, w: 11.5, h: 1.1, fontSize: 44, bold: true,
    color: C.background1, fontFace: THEME.headFontFace, margin: 0 });
  s.addText("Modelagem da evasão e do sucesso estudantil com grafos de similaridade e de fatores",
    { isTextBox: true, x: M, y: 3.3, w: 11.5, h: 0.8, fontSize: 22, color: C.background2, margin: 0 });
  s.addText([
    { text: "Integrante A (Coordenador) · Integrantes B e C (Modeladores) · Integrantes D e E (Desenvolvedores)", options: { breakLine: true } },
    { text: "SMAC03 – Grafos · Prof. Rafael Frinhani · UNIFEI · 2026/2" },
  ], { isTextBox: true, x: M, y: 5.6, w: 11.5, h: 0.9, fontSize: 14, color: C.background2, margin: 0, paraSpaceAfter: 6 });
  s.addNotes("Apresentar o grupo e os papéis. Tema: Cenário 3, evasão e sucesso estudantil, tratados como problema relacional.");

  // ------------------------------------------------------------ 2. problema
  pres.addSection({ title: "Problema" });
  s = content("Problema", "A evasão não tem causa única: ela é relacional");
  const stats = [["4.424", "estudantes, 17 cursos"], ["32,1%", "evadiram"], ["36", "atributos por estudante"]];
  stats.forEach(([n, l], i) => {
    const y = 1.6 + i * 1.7;
    s.addText(n, { isTextBox: true, x: M, y, w: 3.6, h: 0.9, fontSize: 54, bold: true, color: i === 1 ? C.accent1 : C.text2,
      fontFace: THEME.headFontFace, margin: 0 });
    s.addText(l, { isTextBox: true, x: M, y: y + 0.9, w: 3.6, h: 0.4, fontSize: 14, color: C.accent5, margin: 0 });
  });
  card(s, 4.8, 1.6, 7.93, 4.9, C.background2, "card-problema");
  body(s, [
    { text: "Indicadores isolados respondem “quanto” cada fator pesa, mas não “como” os fatores se combinam.", options: { breakLine: true } },
    { text: "Dois estudantes com a mesma média ou renda seguem caminhos diferentes.", options: { breakLine: true } },
    { text: "Questão de pesquisa", options: { bold: true, color: C.text2, breakLine: true } },
    { text: "Como representar estudantes e fatores como grafos revela perfis e combinações associados à evasão e ao sucesso, além das análises isoladas de variáveis?", options: { italic: true } },
  ], { x: 5.2, y: 1.95, w: 7.1, h: 4.3, paraSpaceAfter: 14 });
  s.addText("Dataset: UCI 697 – Predict Students' Dropout and Academic Success (Realinho et al., 2022)",
    { isTextBox: true, x: M, y: 6.6, w: 12, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 });
  s.addNotes("Motivar: instituições têm os dados, mas em relatórios que não mostram relações. Ler a questão de pesquisa.");

  // ------------------------------------------------------------ 3. hipóteses
  s = content("Problema", "Quatro hipóteses guiam a investigação");
  const hs = [
    ["H1", "Homofilia", "Em um grafo construído sem o desfecho, estudantes semelhantes compartilham o mesmo desfecho", ic.users, C.accent3],
    ["H2", "Perfis de risco", "Comunidades isolam perfis com taxas de evasão muito diferentes da média (32%)", ic.net, C.accent1],
    ["H3", "Escore relacional", "O risco pela vizinhança supera o melhor indicador isolado e compete com o modelo tabular", ic.chart, C.accent4],
    ["H4", "Constelações", "Na rede de fatores, os fatores da evasão formam um núcleo distinto dos do sucesso", ic.link, C.accent2],
  ];
  hs.forEach(([h, t, d, im, col], i) => {
    const x = M + (i % 2) * 6.17, y = 1.6 + Math.floor(i / 2) * 2.55;
    card(s, x, y, 5.87, 2.25, C.background2, "card-" + h);
    iconCircle(s, x + 0.3, y + 0.35, 0.8, col, im);
    s.addText(`${h} · ${t}`, { isTextBox: true, x: x + 1.35, y: y + 0.3, w: 4.3, h: 0.5, fontSize: 20, bold: true, color: C.text2, margin: 0 });
    body(s, d, { x: x + 1.35, y: y + 0.85, w: 4.3, h: 1.25, fontSize: 15 });
  });
  s.addNotes("Hipóteses verificáveis com os dados e métodos previstos. Cada uma será retomada nos resultados.");

  // ------------------------------------------------------------ 4. método
  pres.addSection({ title: "Modelagem" });
  s = content("Modelagem", "Do dado tabular a duas redes complementares");
  const steps = [
    ["1 · Dados", "UCI 697\n4.424 × 36"], ["2 · Curadoria", "20 atributos\n77 fatores"], ["3 · Modelagem", "Modelo 1: estudantes\nModelo 2: fatores"],
    ["4 · Análise", "comunidades, assorta-\ntividade, centralidade"], ["5 · Validação", "AUC em validação\ncruzada (5 partições)"],
  ];
  steps.forEach(([t, d], i) => {
    const x = M + i * 2.45;
    s.addShape(pres.shapes.CHEVRON, { x, y: 1.7, w: 2.35, h: 1.0, fill: { color: i === 2 ? C.accent1 : C.text2 }, line: { color: C.background1 } });
    s.addText(t, { isTextBox: true, x: x + 0.35, y: 1.7, w: 1.75, h: 1.0, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0 });
    s.addText(d, { isTextBox: true, x, y: 2.85, w: 2.35, h: 0.8, fontSize: 14, color: C.text1, align: "center", margin: 0 });
  });
  const dims = [
    ["ACAD · α = 0,5", "Notas de admissão e de formação anterior, aprovação e média no 1º/2º semestre, disciplina sem avaliação", C.accent3],
    ["SOCIO · β = 0,3", "Devedor, mensalidade em dia, bolsista, deslocado, escolaridade da mãe e do pai", C.accent1],
    ["DEMO · γ = 0,2", "Idade, gênero, estado civil, turno, internacional, forma de ingresso, curso", C.accent2],
  ];
  dims.forEach(([t, d, col], i) => {
    const x = M + i * 4.1;
    card(s, x, 4.15, 3.85, 2.45, C.background2, "dim-" + i);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 4.45, w: 0.3, h: 0.3, fill: { color: col }, line: { color: col } });
    s.addText(t, { isTextBox: true, x: x + 0.75, y: 4.4, w: 2.9, h: 0.4, fontSize: 18, bold: true, color: C.text2, margin: 0 });
    body(s, d, { x: x + 0.3, y: 4.95, w: 3.3, h: 1.5, fontSize: 14 });
  });
  s.addNotes("Processo CRISP-NET. Discretização em faixas ordinais, como em Porto et al. (2025). Atributos macroeconômicos foram descartados por descreverem o ano, e não o estudante.");

  // ------------------------------------------------------------ 5. modelo 1 + código
  s = content("Modelagem", "Modelo 1: estudantes ligados por similaridade");
  body(s, [
    { text: "G₁ = (V, E, w)", options: { bold: true, fontSize: 20, color: C.text2, breakLine: true } },
    { text: "Vértices: os 4.424 estudantes", options: { bullet: true, breakLine: true } },
    { text: "Aresta: j está entre os k = 15 mais similares a i", options: { bullet: true, breakLine: true } },
    { text: "Peso: Rᵢⱼ = α·S_ACAD + β·S_SOCIO + γ·S_DEMO", options: { bullet: true, breakLine: true } },
    { text: "Ordinal: 1 − |rᵢ − rⱼ| / (A − 1); nominal: igual ou não", options: { bullet: true, breakLine: true } },
    { text: "O desfecho NÃO entra na construção: ele só é usado para avaliar a estrutura", options: { bullet: true, bold: true, color: C.accent1 } },
  ], { x: M, y: 1.6, w: 5.6, h: 4.9, paraSpaceAfter: 10 });
  card(s, 6.6, 1.6, 6.13, 4.95, C.text2, "codigo");
  s.addText("src/graphs.py · núcleo vetorizado", { isTextBox: true, x: 6.9, y: 1.75, w: 5.6, h: 0.35, fontSize: 12, color: C.background2, margin: 0 });
  s.addText(
    "for start in range(0, n, chunk):\n" +
    "    rows = np.arange(start, min(start+chunk, n))\n" +
    "    R = np.zeros((len(rows), n))\n" +
    "    for g in GROUPS:          # ACAD, SOCIO, DEMO\n" +
    "        R += w[g] * group_similarity(f, g, rows)\n" +
    "    R[np.arange(len(rows)), rows] = -1   # sem laços\n" +
    "    part = np.argpartition(-R, k)[:, :k]\n" +
    "    idx[rows] = part  # k vizinhos de cada i\n\n" +
    "G.add_edge(i, j, weight=R[i, j])",
    { isTextBox: true, x: 6.9, y: 2.2, w: 5.6, h: 4.1, fontFace: "Courier New", fontSize: 13, color: C.background1, valign: "top", margin: 0 });
  s.addNotes("Modelo multirrelacional adaptado de Porto et al. (2025). O k-NN garante grafo conexo e esparso, com 49.532 arestas. O processamento em blocos de 512 linhas evita materializar a matriz 4.424².");

  // ------------------------------------------------------------ 6. modelo 2
  s = content("Modelagem", "Modelo 2: fatores ligados por co-ocorrência");
  body(s, [
    { text: "G₂ = (F ∪ D, E₂, lift)", options: { bold: true, fontSize: 20, color: C.text2, breakLine: true } },
    { text: "Vértices: 77 fatores (atributo = valor) e 3 nós de desfecho", options: { bullet: true, breakLine: true } },
    { text: "Projeção do grafo bipartido estudante–fator", options: { bullet: true, breakLine: true } },
    { text: "Aresta se ≥ 40 estudantes compartilham o par e lift ≥ 1,15", options: { bullet: true, breakLine: true } },
    { text: "A vizinhança do nó Evasão mostra a constelação de fatores do abandono", options: { bullet: true } },
  ], { x: M, y: 1.6, w: 5.8, h: 4.9, paraSpaceAfter: 10 });
  card(s, 6.8, 1.6, 5.93, 1.5, C.background2, "formula-lift");
  s.addText("lift(a, b) = P(a ∧ b) / ( P(a) · P(b) )", { isTextBox: true, x: 7.1, y: 1.6, w: 5.4, h: 1.5, fontSize: 22, bold: true,
    color: C.text2, valign: "middle", align: "center", margin: 0 });
  const edges = [["Mensalidade em dia = Não", "Evasão", "2,69"], ["Devedor = Sim", "Evasão", "1,93"], ["Bolsista = Sim", "Graduado", "1,52"], ["Aprovação 2º sem = 100%", "Graduado", "1,73"]];
  s.addTable([
    [{ text: "Fator", options: { bold: true, color: C.background1, fill: { color: C.text2 } } },
     { text: "Desfecho", options: { bold: true, color: C.background1, fill: { color: C.text2 } } },
     { text: "lift", options: { bold: true, color: C.background1, fill: { color: C.text2 }, align: "right" } }],
    ...edges.map(([a, b, l]) => [{ text: a }, { text: b, options: { color: b === "Evasão" ? C.accent1 : C.accent2, bold: true } }, { text: l, options: { align: "right" } }]),
  ], { x: 6.8, y: 3.4, w: 5.93, colW: [3.3, 1.63, 1.0], fontSize: 14, color: C.text1, border: { type: "solid", pt: 0.5, color: HEX.grid }, rowH: 0.5 });
  s.addNotes("Lift > 1 indica co-ocorrência acima do acaso. Fatores do mesmo atributo não se ligam, porque são mutuamente exclusivos.");

  // ------------------------------------------------------------ 7. H1
  pres.addSection({ title: "Resultados" });
  s = content("Resultados", "H1: mesmo sem o desfecho, a rede é homofílica");
  imgFit(s, "fig1_rede_similaridade.png", M, 1.5, 8.9, 5.2, 2380 / 1125, "Rede de similaridade colorida por desfecho e por comunidade");
  s.addText("r = 0,39", { isTextBox: true, x: 9.8, y: 1.7, w: 2.9, h: 0.9, fontSize: 44, bold: true, color: C.text2, fontFace: THEME.headFontFace, margin: 0 });
  s.addText("assortatividade pelo desfecho", { isTextBox: true, x: 9.8, y: 2.6, w: 2.9, h: 0.4, fontSize: 14, color: C.accent5, margin: 0 });
  s.addText("63%", { isTextBox: true, x: 9.8, y: 3.4, w: 2.9, h: 0.9, fontSize: 44, bold: true, color: C.accent1, fontFace: THEME.headFontFace, margin: 0 });
  s.addText("das arestas ligam estudantes com o mesmo desfecho (38% ao acaso)", { isTextBox: true, x: 9.8, y: 4.3, w: 2.9, h: 0.8, fontSize: 14, color: C.accent5, margin: 0 });
  s.addText("Confirmada", { isTextBox: true, x: 9.8, y: 5.6, w: 2.9, h: 0.5, fontSize: 18, bold: true, color: C.accent2, margin: 0 });
  s.addNotes("49.532 arestas, grau médio 22, uma componente. Só com dados de ingresso, r cai para 0,18. Os matriculados são os menos segregados.");

  // ------------------------------------------------------------ 8. H2
  s = content("Resultados", "H2: comunidades separam riscos de 8% a 94%");
  imgFit(s, "fig2_desfecho_por_comunidade.png", M, 1.5, 7.0, 5.3, 1582 / 895, "Desfecho por comunidade Louvain");
  const perfis = [
    ["C2 · 94% de evasão", "Abandono acadêmico: nenhuma aprovação no 2º semestre", C.accent1],
    ["C9 · 83% de evasão", "Inadimplência: 100% devedores, mesmo com aprovação parcial", C.accent1],
    ["C5 · 27% de evasão", "Adultos, noturno, maiores de 23: o risco “da idade” não se confirma quando há bom desempenho", C.accent2],
  ];
  perfis.forEach(([t, d, col], i) => {
    const y = 1.6 + i * 1.65;
    card(s, 7.9, y, 4.83, 1.45, C.background2, "perfil-" + i);
    s.addText(t, { isTextBox: true, x: 8.15, y: y + 0.15, w: 4.4, h: 0.4, fontSize: 17, bold: true, color: col, margin: 0 });
    body(s, d, { x: 8.15, y: y + 0.55, w: 4.4, h: 0.85, fontSize: 14 });
  });
  s.addText("Louvain: Q = 0,715 · 12 comunidades", { isTextBox: true, x: 7.9, y: 6.55, w: 4.83, h: 0.3, fontSize: 12, color: C.accent5, margin: 0 });
  s.addNotes("Louvain teve a maior modularidade (Fast Greedy: 0,652; Label Propagation: 0,603). NMI baixo é esperado: as comunidades são perfis, e não desfechos. Média institucional: 32%.");

  // ------------------------------------------------------------ 9. pares divergentes
  s = content("Resultados", "Perfis de ingresso iguais divergem no 1º ano");
  imgFit(s, "fig4_pares_divergentes.png", M, 1.5, 7.4, 5.3, 1490 / 780, "Atributos que diferem em pares divergentes");
  body(s, [
    { text: "2.382 pares Evasão × Graduado com similaridade de ingresso ≥ 0,97", options: { bullet: true, breakLine: true } },
    { text: "A aprovação no 2º semestre difere em 79% desses pares, contra 45% no controle (+34 p.p.)", options: { bullet: true, breakLine: true } },
    { text: "Os atributos socioeconômicos praticamente não diferem", options: { bullet: true, breakLine: true } },
    { text: "16,7% dos evadidos têm vizinhança majoritariamente não evadida: são casos “inesperados”", options: { bullet: true } },
  ], { x: 8.3, y: 1.7, w: 4.43, h: 4.8, paraSpaceAfter: 12 });
  s.addNotes("O desfecho é decidido na trajetória dos dois primeiros semestres: é ali que a intervenção precoce tem mais efeito.");

  // ------------------------------------------------------------ 10. H3 (gráfico nativo)
  s = content("Resultados", "H3: o grafo supera o indicador isolado");
  const metodos = ["Melhor variável isolada", "Grafo: voto dos vizinhos", "Grafo: propagação de rótulos", "Regressão logística", "Híbrido (logística + grafo)"];
  s.addChart(pres.charts.BAR, [
    { name: "Ingresso + 1º/2º semestres", labels: metodos, values: [0.856, 0.894, 0.899, 0.914, 0.915] },
    { name: "Apenas ingresso", labels: metodos, values: [0.655, 0.766, 0.774, 0.819, 0.819] },
  ], {
    x: M, y: 1.5, w: 8.0, h: 5.3, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [HEX.blue, HEX.orange], valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1,
    valAxisLabelFormatCode: "0.0", dataLabelFormatCode: "0.000", showValue: true, dataLabelPosition: "outEnd",
    dataLabelFontSize: 11, dataLabelColor: HEX.muted, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: HEX.muted,
    showTitle: true, title: "AUC-ROC (Evasão × demais), validação cruzada 5-fold", titleFontSize: 14, titleColor: HEX.navy, titleFontFace: "+mn-lt",
  });
  s.addText("0,899", { isTextBox: true, x: 9.0, y: 1.7, w: 3.7, h: 0.9, fontSize: 44, bold: true, color: C.text2, fontFace: THEME.headFontFace, margin: 0 });
  s.addText("AUC da propagação de rótulos, contra 0,856 da melhor variável isolada", { isTextBox: true, x: 9.0, y: 2.6, w: 3.7, h: 0.8, fontSize: 14, color: C.accent5, margin: 0 });
  s.addText("91%", { isTextBox: true, x: 9.0, y: 3.6, w: 3.7, h: 0.9, fontSize: 44, bold: true, color: C.accent1, fontFace: THEME.headFontFace, margin: 0 });
  s.addText("dos 20% de maior risco de fato evadiram", { isTextBox: true, x: 9.0, y: 4.5, w: 3.7, h: 0.5, fontSize: 14, color: C.accent5, margin: 0 });
  body(s, "Parcialmente confirmada: o híbrido soma só +0,001 à logística. O valor do grafo está na explicação: quem são os vizinhos e o que aconteceu com eles.",
    { x: 9.0, y: 5.3, w: 3.7, h: 1.4, fontSize: 14 });
  s.addNotes("Os rótulos do teste são ocultados em cada partição, sem vazamento. Sem dados de desempenho, a vantagem do grafo sobre a variável isolada aumenta: 0,774 contra 0,655.");

  // ------------------------------------------------------------ 11. H4
  s = content("Resultados", "H4: cada desfecho tem sua constelação de fatores");
  imgFit(s, "fig7_rede_fatores.png", M, 1.45, 7.3, 5.45, 3180 / 2380, "Rede de fatores com três comunidades, uma por desfecho");
  body(s, [
    { text: "O Louvain encontra 3 comunidades, cada uma com exatamente um desfecho", options: { bold: true, color: C.text2, breakLine: true } },
    { text: "Evasão: aprovação nula ou baixa, inadimplência, idade 23+, noturno", options: { bullet: true, breakLine: true } },
    { text: "Graduado: aprovação plena, bolsa, 1ª fase, idade 17–19", options: { bullet: true, breakLine: true } },
    { text: "Sinergia: devedor (62%) com ingresso “maiores de 23” (55%) chega a 77% de evasão", options: { bullet: true, breakLine: true } },
    { text: "Aprovação parcial (66%) com idade 30+ (54%) chega a 86%", options: { bullet: true } },
  ], { x: 8.2, y: 1.6, w: 4.53, h: 5.2, paraSpaceAfter: 10, fontSize: 15 });
  s.addNotes("Os fatores se entrelaçam: a combinação de dois fatores pesa mais do que cada um isolado. É isso que os relatórios isolados não mostram.");

  // ------------------------------------------------------------ 12. implicações
  pres.addSection({ title: "Aplicação" });
  s = content("Aplicação", "Três perfis de risco pedem três intervenções");
  const acoes = [
    ["Acompanhamento pedagógico", "C2: nenhuma aprovação. Agir logo após o 1º semestre, com tutoria e monitoria.", ic.book, C.accent3],
    ["Apoio financeiro", "C9: devedores com aprovação parcial. O problema é financeiro, e não acadêmico.", ic.money, C.accent1],
    ["Busca ativa", "C4: disciplinas sem avaliação. Contatar quem parou de comparecer antes da evasão formal.", ic.search, C.accent4],
  ];
  acoes.forEach(([t, d, im, col], i) => {
    const x = M + i * 4.1;
    card(s, x, 1.7, 3.85, 4.1, C.background2, "acao-" + i);
    iconCircle(s, x + 0.35, 2.0, 0.95, col, im);
    s.addText(t, { isTextBox: true, x: x + 0.35, y: 3.15, w: 3.2, h: 0.8, fontSize: 19, bold: true, color: C.text2, margin: 0, valign: "top" });
    body(s, d, { x: x + 0.35, y: 3.95, w: 3.2, h: 1.7, fontSize: 15 });
  });
  body(s, "Adultos e estudantes do noturno: o foco deve ser o desempenho, e não o perfil.",
    { x: M, y: 6.1, w: 12.1, h: 0.5, fontSize: 16, italic: true, color: C.text2 });
  s.addNotes("A rede transforma listas de variáveis em grupos acionáveis, com recursos limitados e alvos claros.");

  // ------------------------------------------------------------ 13. demo
  s = content("Aplicação", "Demonstração: o protótipo de ponta a ponta");
  card(s, M, 1.6, 5.9, 4.95, C.text2, "comandos");
  s.addText([
    { text: "# pipeline completo (~4 min)", options: { color: C.background2, breakLine: true } },
    { text: "python -m src.main", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "# explorador interativo", options: { color: C.background2, breakLine: true } },
    { text: "streamlit run app/explorador.py", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "# saídas", options: { color: C.background2, breakLine: true } },
    { text: "outputs/tabelas/*.csv", options: { breakLine: true } },
    { text: "outputs/figuras/*.png" },
  ], { isTextBox: true, x: M + 0.35, y: 1.9, w: 5.3, h: 4.4, fontFace: "Courier New", fontSize: 15, color: C.background1, margin: 0, valign: "top" });
  const tabs = [
    ["Estudante", "Rede ego, escore de risco, comunidade e os 5 vizinhos mais similares"],
    ["Comunidades", "Perfis, desfechos e membros de cada comunidade"],
    ["Rede de fatores", "Vizinhança de cada desfecho e sinergias entre fatores"],
  ];
  tabs.forEach(([t, d], i) => {
    const y = 1.6 + i * 1.7;
    iconCircle(s, 6.9, y + 0.15, 0.75, [C.accent3, C.accent1, C.accent2][i], [ic.grad, ic.users, ic.net][i]);
    s.addText(t, { isTextBox: true, x: 7.9, y: y + 0.1, w: 4.8, h: 0.45, fontSize: 19, bold: true, color: C.text2, margin: 0 });
    body(s, d, { x: 7.9, y: y + 0.6, w: 4.8, h: 0.9, fontSize: 15 });
  });
  s.addNotes("DEMO: abrir a aba Estudante, escolher 'evasão inesperada', mostrar a vizinhança. Depois a aba Rede de fatores com Evasão.");

  // ------------------------------------------------------------ 14. conclusões
  pres.addSection({ title: "Conclusão" });
  s = pres.addSlide({ masterName: "ESCURO", sectionTitle: "Conclusão" });
  s.addText("A rede mostra como os fatores se combinam", { placeholder: "title" });
  const res = [["H1", "Homofilia: r = 0,39", "✓"], ["H2", "Comunidades: evasão de 8% a 94%", "✓"], ["H3", "AUC 0,899 > 0,856 (isolada)", "◐"], ["H4", "Uma constelação por desfecho", "✓"]];
  res.forEach(([h, t, m], i) => {
    const y = 1.7 + i * 1.0;
    s.addText(m, { isTextBox: true, x: M, y, w: 0.6, h: 0.7, fontSize: 28, bold: true, color: m === "✓" ? C.accent2 : C.accent6, margin: 0 });
    s.addText([{ text: h + "  ", options: { bold: true } }, { text: t }], { isTextBox: true, x: M + 0.7, y, w: 5.3, h: 0.7, fontSize: 18, color: C.background1, valign: "middle", margin: 0 });
  });
  s.addText([
    { text: "Limitações", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "Uma instituição portuguesa; só 2 semestres", options: { bullet: true, breakLine: true } },
    { text: "“Matriculado” tem desfecho incerto; lift não é causal", options: { bullet: true, breakLine: true } },
    { text: "Trabalhos futuros", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "Dados da UNIFEI com histórico por disciplina", options: { bullet: true, breakLine: true } },
    { text: "Trajetórias como grafos temporais e dirigidos", options: { bullet: true, breakLine: true } },
    { text: "Pesos α, β, γ aprendidos", options: { bullet: true } },
  ], { isTextBox: true, x: 7.0, y: 1.7, w: 5.73, h: 4.6, fontSize: 16, color: C.background1, margin: 0, paraSpaceAfter: 8, valign: "top" });
  s.addText("Código, relatório e slides: github.com/viniciusnreno/smac03-evasao-grafos", { isTextBox: true, x: M, y: 6.5, w: 12, h: 0.4, fontSize: 14, color: C.background2, margin: 0 });
  s.addNotes("Fechar respondendo à questão de pesquisa. Abrir para perguntas.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("ok:", OUT);
})();
