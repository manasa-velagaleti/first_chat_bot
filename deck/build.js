const pptxgen = require("pptxgenjs");

const THEME = {
  name: "RAG Assistant",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "121A2B",   // text
    lt1: "FFFFFF",
    dk2: "21295C",   // midnight — dark slide ground
    lt2: "F1F5F9",   // light ground
    accent1: "065A82",  // deep blue, dominant
    accent2: "1C7293",  // teal
    accent3: "02C39A",  // mint, sharp accent
    accent4: "E8833A",  // amber, for caveats
    accent5: "64748B",  // muted
    accent6: "CBD5E1",  // rule
    hlink: "1C7293",
    folHlink: "64748B",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.3 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Manasa";
pres.title = "RAG Chat Assistant — Architecture";
const C = pres.SchemeColor;

const W = 13.33, H = 7.5, M = 0.7;

/* ---------- layouts ---------- */

pres.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: THEME.colors.dk2 },
  objects: [
    { text: { text: "", options: {
      placeholder: "title", x: M, y: 2.5, w: W - 2 * M, h: 1.5,
      fontSize: 44, bold: true, color: C.background1, align: "left" } } },
    { text: { text: "", options: {
      placeholder: "body", x: M, y: 4.05, w: W - 2 * M, h: 1.2,
      fontSize: 18, color: THEME.colors.accent6, align: "left" } } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { text: { text: "", options: {
      placeholder: "title", x: M, y: 0.45, w: W - 2 * M, h: 0.85,
      fontSize: 34, bold: true, color: C.text1, align: "left" } } },
    { text: { text: "", options: {
      placeholder: "body", x: M, y: 1.35, w: W - 2 * M, h: 0.5,
      fontSize: 15, color: THEME.colors.accent5, align: "left" } } },
  ],
  slideNumber: { x: W - 1.0, y: H - 0.5, fontSize: 10,
                 color: THEME.colors.accent5 },
});

pres.defineSlideMaster({
  title: "DARK_CONTENT",
  background: { color: THEME.colors.dk2 },
  objects: [
    { text: { text: "", options: {
      placeholder: "title", x: M, y: 0.45, w: W - 2 * M, h: 0.85,
      fontSize: 34, bold: true, color: C.background1, align: "left" } } },
    { text: { text: "", options: {
      placeholder: "body", x: M, y: 1.35, w: W - 2 * M, h: 0.5,
      fontSize: 15, color: THEME.colors.accent6, align: "left" } } },
  ],
  slideNumber: { x: W - 1.0, y: H - 0.5, fontSize: 10,
                 color: THEME.colors.accent5 },
});

/* ---------- helpers ---------- */

let n = 0;
const card = (s, x, y, w, h, fill, opts = {}) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08, fill: { color: fill },
    line: opts.line || { color: THEME.colors.accent6, width: 0.75 },
    objectName: `card${++n}`,
    ...(opts.shadow ? { shadow: { type: "outer", angle: 90, offset: 2,
                                  blur: 6, opacity: 0.10, color: "000000" } } : {}),
  });

const txt = (s, t, o) => s.addText(t, { isTextBox: true, margin: 0, ...o });

const chip = (s, label, x, y, w, color) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h: 0.34, rectRadius: 0.17, fill: { color },
    line: { color, width: 0 }, objectName: `chip${++n}`,
  });
  txt(s, label, { x, y, w, h: 0.34, fontSize: 11, bold: true,
                  color: "FFFFFF", align: "center", valign: "middle" });
};

/* ================= 1. title ================= */
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Opening" });
txt(s, "A multi-model assistant that answers from your documents", {
  x: M, y: 1.75, w: W - 2 * M, h: 0.4, fontSize: 15,
  color: THEME.colors.accent3, bold: true, charSpacing: 1.5 });
s.addText("RAG Chat Assistant", { placeholder: "title" });
s.addText("Architecture and design decisions  ·  Manasa Velagaleti",
          { placeholder: "body" });
[["18 models", 0], ["3 providers", 1.85], ["587 chunks", 3.7],
 ["5 file types", 5.55]].forEach(([t, dx]) =>
  chip(s, t, M + dx, 5.5, 1.65, THEME.colors.accent1));
s.addNotes("Opening. This is a chat assistant that answers from my own documents, "
  + "across 18 models and three providers, and shows where every answer came from.");

/* ================= 2. what it does ================= */
pres.addSection({ title: "Overview" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Overview" });
s.addText("What it does", { placeholder: "title" });
s.addText("Three capabilities, one conversation", { placeholder: "body" });

const caps = [
  ["01", "Chats across 18 models",
   "Three providers, eight sampling controls, and memory that survives switching model mid-conversation.",
   THEME.colors.accent1],
  ["02", "Answers from your documents",
   "PDF, Word, CSV, text and Markdown — including scanned PDFs, read with OCR. Every answer cites its passages.",
   THEME.colors.accent2],
  ["03", "Checks its own work",
   "A second model from a different provider marks each answer correct or incorrect against the passages it used.",
   THEME.colors.accent3],
];
caps.forEach(([num, head, body, col], i) => {
  const x = M + i * 4.08;
  card(s, x, 2.1, 3.78, 3.3, C.background1, { shadow: true });
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 2.42, w: 0.62, h: 0.62,
    fill: { color: col }, line: { color: col, width: 0 },
    objectName: `num${i}` });
  txt(s, num, { x: x + 0.3, y: 2.42, w: 0.62, h: 0.62, fontSize: 15,
                bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  txt(s, head, { x: x + 0.3, y: 3.22, w: 3.18, h: 0.6, fontSize: 17,
                 bold: true, color: C.text1 });
  txt(s, body, { x: x + 0.3, y: 3.85, w: 3.18, h: 1.3, fontSize: 13,
                 color: THEME.colors.accent5, lineSpacingMultiple: 1.15 });
});
s.addNotes("Three things: it is a normal chatbot, it can answer from my documents "
  + "with citations, and a second model can grade those answers.");

/* ================= 3. the stack ================= */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Overview" });
s.addText("What it is built on", { placeholder: "title" });
s.addText("Chosen for what each layer is actually good at", { placeholder: "body" });

const stack = [
  ["Orchestration", "LangChain + LangGraph", "Agent loop, tool calling, and conversation memory via a checkpointer", THEME.colors.accent1],
  ["Generation", "18 models, 3 providers", "OpenAI · Google Gemini · Groq — switchable mid-conversation", THEME.colors.accent2],
  ["Embeddings", "text-embedding-3-small", "1536 dimensions. Chosen by measurement, not by default", THEME.colors.accent3],
  ["Vector store", "Pinecone (serverless)", "Cosine similarity, one namespace per chunking strategy", THEME.colors.accent1],
  ["Documents", "pypdf · docx2txt · RapidOCR", "Five file types, with OCR fallback for scanned pages", THEME.colors.accent2],
  ["Interface", "Streamlit · React + FastAPI", "Three front ends over one retrieval core", THEME.colors.accent3],
];
stack.forEach(([label, pick, why, col], i) => {
  const y = 2.05 + i * 0.78;
  card(s, M, y, W - 2 * M, 0.66, i % 2 ? C.background2 : C.background1,
       { line: { color: THEME.colors.accent6, width: 0.5 } });
  s.addShape(pres.shapes.OVAL, { x: M + 0.22, y: y + 0.21, w: 0.24, h: 0.24,
    fill: { color: col }, line: { color: col, width: 0 }, objectName: `dot${i}` });
  txt(s, label, { x: M + 0.62, y, w: 1.9, h: 0.66, fontSize: 13,
                  bold: true, color: THEME.colors.accent5, valign: "middle" });
  txt(s, pick, { x: M + 2.55, y, w: 3.3, h: 0.66, fontSize: 14,
                 bold: true, color: C.text1, valign: "middle" });
  txt(s, why, { x: M + 5.95, y, w: 5.6, h: 0.66, fontSize: 12.5,
                color: THEME.colors.accent5, valign: "middle" });
});
s.addNotes("The stack, layer by layer. LangGraph gives the agent loop and memory; "
  + "Pinecone holds the vectors; three front ends share one retrieval core.");

/* ================= 4. architecture ================= */
pres.addSection({ title: "Architecture" });
s = pres.addSlide({ masterName: "DARK_CONTENT", sectionTitle: "Architecture" });
s.addText("One core, three front ends", { placeholder: "title" });
s.addText("The retrieval logic is written once and shared", { placeholder: "body" });

card(s, 3.4, 2.15, 6.5, 1.5, THEME.colors.accent1, { line: { color: THEME.colors.accent2, width: 1 } });
txt(s, "core.py  +  rag_pinecone.py", { x: 3.4, y: 2.35, w: 6.5, h: 0.4,
  fontSize: 20, bold: true, color: "FFFFFF", align: "center" });
txt(s, "loading · chunking · embeddings · retrieval · generation", {
  x: 3.4, y: 2.82, w: 6.5, h: 0.5, fontSize: 13,
  color: THEME.colors.accent6, align: "center" });

const fronts = [
  ["Terminal", "rag_pinecone.py --ask", "Scripting and testing"],
  ["Streamlit app", "app.py", "The full assistant"],
  ["React + FastAPI", "rag_web/", "Document Q&A page"],
];
fronts.forEach(([name, file, use], i) => {
  const x = M + i * 4.08;
  card(s, x, 4.5, 3.78, 1.5, THEME.colors.dk1, { line: { color: THEME.colors.accent5, width: 0.75 } });
  txt(s, name, { x: x + 0.25, y: 4.7, w: 3.28, h: 0.35, fontSize: 16,
                 bold: true, color: "FFFFFF" });
  txt(s, file, { x: x + 0.25, y: 5.08, w: 3.28, h: 0.3, fontSize: 12,
                 color: THEME.colors.accent3, fontFace: "Courier New" });
  txt(s, use, { x: x + 0.25, y: 5.42, w: 3.28, h: 0.35, fontSize: 12,
                color: THEME.colors.accent6 });
  s.addShape(pres.shapes.LINE, { x: x + 1.89, y: 3.65, w: 6.65 - x - 1.89 + 0.0001, h: 0.85,
    line: { color: THEME.colors.accent5, width: 1, endArrowType: "triangle" },
    objectName: `arrow${i}` });
});
txt(s, "Adding a front end means writing an interface, not another RAG pipeline.", {
  x: M, y: 6.35, w: W - 2 * M, h: 0.4, fontSize: 13, italic: true,
  color: THEME.colors.accent3 });
s.addNotes("The retrieval logic lives in two modules. The CLI, the web page and the "
  + "chat app are all thin front ends over it — change the prompt once and all three change.");

/* ================= 5. document pipeline ================= */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architecture" });
s.addText("How a document becomes searchable", { placeholder: "title" });
s.addText("Four steps, paid once per document", { placeholder: "body" });

const steps = [
  ["Load", "PDF · DOCX · CSV\nTXT · MD", "Scanned PDFs fall back to OCR"],
  ["Chunk", "700 characters\nrecursive", "Split on paragraph, then sentence"],
  ["Embed", "text-embedding\n-3-small", "1536 numbers per chunk"],
  ["Store", "Pinecone\nnamespace", "Cosine similarity search"],
];
steps.forEach(([head, mid, foot], i) => {
  const x = M + i * 3.1;
  card(s, x, 2.3, 2.75, 2.5, i % 2 ? C.background2 : C.background1, { shadow: true });
  s.addShape(pres.shapes.OVAL, { x: x + 1.14, y: 2.5, w: 0.47, h: 0.47,
    fill: { color: THEME.colors.accent1 },
    line: { color: THEME.colors.accent1, width: 0 }, objectName: `st${i}` });
  txt(s, String(i + 1), { x: x + 1.14, y: 2.5, w: 0.47, h: 0.47, fontSize: 13,
    bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  txt(s, head, { x: x + 0.2, y: 3.08, w: 2.35, h: 0.35, fontSize: 18,
                 bold: true, color: C.text1, align: "center" });
  txt(s, mid, { x: x + 0.2, y: 3.5, w: 2.35, h: 0.7, fontSize: 12.5,
                color: THEME.colors.accent2, align: "center", bold: true });
  txt(s, foot, { x: x + 0.2, y: 4.2, w: 2.35, h: 0.5, fontSize: 11.5,
                 color: THEME.colors.accent5, align: "center" });
  if (i < 3) txt(s, "→", { x: x + 2.76, y: 3.3, w: 0.34, h: 0.4, fontSize: 20,
                           color: THEME.colors.accent6, align: "center" });
});

card(s, M, 5.15, W - 2 * M, 1.05, "FDF3E7", { line: { color: THEME.colors.accent4, width: 1 } });
txt(s, [{ text: "Scanned PDFs: ", options: { bold: true, color: THEME.colors.accent4 } },
        { text: "a scan has no text layer — pypdf finds nothing. Each page is rendered to a bitmap and read with OCR, but only after plain extraction comes back empty, so ordinary PDFs pay nothing for it.",
          options: { color: C.text1 } }],
     { x: M + 0.3, y: 5.35, w: W - 2 * M - 0.6, h: 0.7, fontSize: 13 });
s.addNotes("Load, chunk, embed, store. The OCR fallback only runs when normal "
  + "extraction finds nothing, so a text PDF is not slowed down by it.");

/* ================= 6. retrieval as a tool ================= */
pres.addSection({ title: "Design decisions" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Design decisions" });
s.addText("Retrieval is a tool, not a pipeline step", { placeholder: "title" });
s.addText("The model decides when to search", { placeholder: "body" });

card(s, M, 2.15, 5.9, 3.5, C.background2);
txt(s, "Search before every message", { x: M + 0.35, y: 2.4, w: 5.2, h: 0.4,
  fontSize: 17, bold: true, color: THEME.colors.accent5 });
txt(s, [
  { text: "Every \u201chello\u201d triggers a search", options: { bullet: true, breakLine: true } },
  { text: "Irrelevant passages enter the context", options: { bullet: true, breakLine: true } },
  { text: "Costs a retrieval on every turn", options: { bullet: true, breakLine: true } },
  { text: "Cannot combine documents with what the user says", options: { bullet: true } },
], { x: M + 0.35, y: 2.95, w: 5.2, h: 2.4, fontSize: 13.5,
     color: THEME.colors.accent5, paraSpaceAfter: 8 });

card(s, 6.95, 2.15, 5.68, 3.5, C.background1,
     { line: { color: THEME.colors.accent3, width: 1.5 }, shadow: true });
txt(s, "A tool the model can call", { x: 7.3, y: 2.4, w: 5, h: 0.4,
  fontSize: 17, bold: true, color: THEME.colors.accent1 });
txt(s, [
  { text: "Chitchat costs nothing", options: { bullet: true, breakLine: true } },
  { text: "Searches again when the question changes", options: { bullet: true, breakLine: true } },
  { text: "Blends document facts with what you tell it", options: { bullet: true, breakLine: true } },
  { text: "Needs a model that supports tool calling", options: { bullet: true } },
], { x: 7.3, y: 2.95, w: 5, h: 2.4, fontSize: 13.5, color: C.text1,
     paraSpaceAfter: 8 });

txt(s, "\u201cI know Python but have 1 year of experience — am I a fit?\u201d  needs the document and the user together. A fixed pipeline cannot answer it.",
  { x: M, y: 6.0, w: W - 2 * M, h: 0.5, fontSize: 13.5, italic: true,
    color: THEME.colors.accent2, align: "center" });
s.addNotes("Most turns in a conversation are not document questions. Letting the "
  + "model decide keeps ordinary chat ordinary, and lets it combine sources.");

/* ================= 7. capability matrix ================= */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Design decisions" });
s.addText("Not every model supports every control", { placeholder: "title" });
s.addText("Verified by testing each one, not by reading documentation", { placeholder: "body" });

const rows = [
  ["", "OpenAI", "Gemini", "Groq"],
  ["Temperature, Top P", "Yes", "Yes", "Yes"],
  ["Top K", "No", "Yes", "No"],
  ["Frequency / presence penalty", "Yes", "No", "Yes"],
  ["Seed, stop sequences", "Yes", "Yes", "Yes"],
  ["Tool calling (document search)", "Yes", "Yes", "4 of 7 models"],
];
const colX = [M, 5.9, 8.0, 10.1], colW = [5.1, 2.0, 2.0, 2.5];
rows.forEach((r, ri) => {
  const y = 2.15 + ri * 0.62;
  if (ri === 0) {
    r.forEach((c, ci) => txt(s, c, { x: colX[ci], y, w: colW[ci], h: 0.5,
      fontSize: 13, bold: true, color: THEME.colors.accent5, valign: "middle",
      align: ci ? "center" : "left" }));
    return;
  }
  card(s, M, y, W - 2 * M, 0.52, ri % 2 ? C.background2 : C.background1,
       { line: { color: THEME.colors.accent6, width: 0.5 } });
  r.forEach((c, ci) => {
    const yes = c !== "No";
    txt(s, c, { x: colX[ci] + (ci ? 0 : 0.2), y, w: colW[ci], h: 0.52,
      fontSize: ci ? 12.5 : 13.5, bold: ci === 0,
      color: ci === 0 ? C.text1 : (yes ? THEME.colors.accent1 : THEME.colors.accent4),
      valign: "middle", align: ci ? "center" : "left" });
  });
});
card(s, M, 5.95, W - 2 * M, 0.95, "FDF3E7", { line: { color: THEME.colors.accent4, width: 1 } });
txt(s, [{ text: "Declaring is not supporting.  ", options: { bold: true, color: THEME.colors.accent4 } },
        { text: "The Gemini client declares both penalty fields, but the API rejects them with a 400. The app hides what will not work rather than letting the request fail.",
          options: { color: C.text1 } }],
    { x: M + 0.3, y: 6.15, w: W - 2 * M - 0.6, h: 0.6, fontSize: 13 });
s.addNotes("Every cell here was tested by calling the model. The penalty row is the "
  + "one that caught me out — the library declares the fields, the API refuses them.");

/* ================= 8. evaluation ================= */
s = pres.addSlide({ masterName: "DARK_CONTENT", sectionTitle: "Design decisions" });
s.addText("A second model marks the first", { placeholder: "title" });
s.addText("LLM-as-judge, across providers on purpose", { placeholder: "body" });

const flow = [
  ["Question", "What the user asked", THEME.colors.accent2],
  ["Passages", "What retrieval returned", THEME.colors.accent2],
  ["Answer", "What the first model said", THEME.colors.accent2],
];
flow.forEach(([h, b, col], i) => {
  const y = 2.2 + i * 0.95;
  card(s, M, y, 4.2, 0.78, THEME.colors.dk1, { line: { color: col, width: 1 } });
  txt(s, h, { x: M + 0.25, y, w: 1.6, h: 0.78, fontSize: 15, bold: true,
              color: "FFFFFF", valign: "middle" });
  txt(s, b, { x: M + 1.85, y, w: 2.25, h: 0.78, fontSize: 12,
              color: THEME.colors.accent6, valign: "middle" });
});
s.addShape(pres.shapes.LINE, { x: 5.0, y: 3.45, w: 1.5, h: 0,
  line: { color: THEME.colors.accent3, width: 2, endArrowType: "triangle" },
  objectName: "jarrow" });
card(s, 6.6, 2.5, 3.1, 1.9, THEME.colors.accent1, { line: { color: THEME.colors.accent3, width: 1.5 } });
txt(s, "Judge model", { x: 6.6, y: 2.75, w: 3.1, h: 0.4, fontSize: 17,
  bold: true, color: "FFFFFF", align: "center" });
txt(s, "different provider", { x: 6.6, y: 3.15, w: 3.1, h: 0.3, fontSize: 12,
  color: THEME.colors.accent6, align: "center" });
txt(s, "\u201ccorrect\u201d / \u201cincorrect\u201d", { x: 6.6, y: 3.6, w: 3.1, h: 0.4,
  fontSize: 14, bold: true, color: THEME.colors.accent3, align: "center" });

txt(s, "Why a different provider", { x: 10.1, y: 2.5, w: 2.6, h: 0.35,
  fontSize: 14, bold: true, color: THEME.colors.accent3 });
txt(s, "A model marking its own work mostly agrees with itself. Two models from one family share training — and tend to share mistakes.",
  { x: 10.1, y: 2.9, w: 2.6, h: 1.5, fontSize: 12,
    color: THEME.colors.accent6, lineSpacingMultiple: 1.15 });

card(s, M, 5.0, W - 2 * M, 1.5, THEME.colors.dk1, { line: { color: THEME.colors.accent4, width: 1 } });
txt(s, "What it measures — and what it does not", { x: M + 0.3, y: 5.2, w: 11.3, h: 0.35,
  fontSize: 14, bold: true, color: THEME.colors.accent4 });
txt(s, "Five questions scored 10/10 by two judges. One was wrong anyway: the answer said visa sponsorship was not mentioned, the judges agreed — and the document does mention it, on a page OCR had mangled. The judge sees the retrieved passages, so it measures faithfulness to retrieval, not retrieval itself.",
  { x: M + 0.3, y: 5.58, w: 11.3, h: 0.8, fontSize: 12.5,
    color: THEME.colors.accent6, lineSpacingMultiple: 1.1 });
s.addNotes("The judge is shown the question, the passages and the answer, and "
  + "returns one word. Important caveat: a retrieval failure passes this test.");

/* ================= 9. numbers ================= */
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("By the numbers", { placeholder: "title" });
s.addText("Five documents indexed, measured on the running system", { placeholder: "body" });

const stats = [
  ["587", "chunks indexed", THEME.colors.accent1],
  ["1,536", "dimensions per chunk", THEME.colors.accent2],
  ["18", "models available", THEME.colors.accent3],
  ["$0.0002", "to index everything", THEME.colors.accent1],
  ["$0.0005", "per question asked", THEME.colors.accent2],
  ["~2s", "to answer with sources", THEME.colors.accent3],
];
stats.forEach(([big, label, col], i) => {
  const x = M + (i % 3) * 4.08, y = 2.2 + Math.floor(i / 3) * 2.15;
  card(s, x, y, 3.78, 1.8, i % 2 ? C.background2 : C.background1, { shadow: true });
  txt(s, big, { x: x + 0.3, y: y + 0.28, w: 3.18, h: 0.85, fontSize: 40,
                bold: true, color: col });
  txt(s, label, { x: x + 0.3, y: y + 1.15, w: 3.18, h: 0.4, fontSize: 13,
                  color: THEME.colors.accent5 });
});
s.addNotes("Indexing all five documents cost a fifth of a cent. Each question is "
  + "about half a cent. Cost was never the constraint — correctness was.");

/* ================= 10. hard problems ================= */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("The hard part was not getting answers", { placeholder: "title" });
s.addText("Every one of these produced a confident, fluent, wrong result", { placeholder: "body" });

const probs = [
  ["Answered from the wrong corpus",
   "A question about a job title was answered from a 10-K filing — the index being searched had been built before that document was uploaded.",
   "The app now refuses a stale index instead of answering from it."],
  ["Reused passages from the previous question",
   "Asked about experience right after salary, it reported experience was not specified. It is — on a page it never went back for.",
   "The prompt now requires a fresh search when the subject changes."],
  ["OCR mangled one line",
   "A sentence about visa sponsorship was garbled, so retrieval never found it. Both judges still marked the answer correct.",
   "Documented rather than hidden: evaluation cannot catch this."],
];
probs.forEach(([head, body, fix], i) => {
  const y = 2.05 + i * 1.6;
  card(s, M, y, W - 2 * M, 1.42, i % 2 ? C.background2 : C.background1,
       { line: { color: THEME.colors.accent6, width: 0.75 } });
  s.addShape(pres.shapes.OVAL, { x: M + 0.28, y: y + 0.5, w: 0.42, h: 0.42,
    fill: { color: THEME.colors.accent4 },
    line: { color: THEME.colors.accent4, width: 0 }, objectName: `p${i}` });
  txt(s, String(i + 1), { x: M + 0.28, y: y + 0.5, w: 0.42, h: 0.42, fontSize: 12,
    bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  txt(s, head, { x: M + 0.9, y: y + 0.16, w: 11.0, h: 0.32, fontSize: 15.5,
                 bold: true, color: C.text1 });
  txt(s, body, { x: M + 0.9, y: y + 0.5, w: 11.0, h: 0.42, fontSize: 12.5,
                 color: THEME.colors.accent5 });
  txt(s, fix, { x: M + 0.9, y: y + 0.95, w: 11.0, h: 0.32, fontSize: 12.5,
                bold: true, color: THEME.colors.accent1 });
});
s.addNotes("None of these crashed. All three produced fluent, well-cited, wrong "
  + "answers — which is the failure mode that actually matters in RAG.");

/* ================= 11. close ================= */
pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Close" });
txt(s, "WHAT I TOOK FROM IT", { x: M, y: 1.75, w: W - 2 * M, h: 0.4,
  fontSize: 15, color: THEME.colors.accent3, bold: true, charSpacing: 1.5 });
s.addText("Making wrong answers visible", { placeholder: "title" });
s.addText("A RAG system that fails does not look broken — it looks confident. "
  + "So every answer cites its passages, stale indexes are refused, and a second model can mark the work.",
  { placeholder: "body" });
[["Sources on every answer", 0], ["Stale indexes refused", 4.1],
 ["A second model grades it", 8.2]].forEach(([t, dx]) =>
  chip(s, t, M + dx, 5.6, 3.7, THEME.colors.accent1));
txt(s, "github.com/manasa-velagaleti/first_chat_bot", {
  x: M, y: 6.4, w: W - 2 * M, h: 0.4, fontSize: 13,
  color: THEME.colors.accent6, fontFace: "Courier New" });
s.addNotes("Closing. The hardest problem was not getting answers — it was making "
  + "wrong ones visible.");

(async () => {
  await pres.writeFile({ fileName: "rag-architecture.pptx" });
  const { applyTheme } = require("C:/Users/prati/.claude/skills/synced/fe447333-fc1c-425c-97a2-89b0832ba426_a82ebd59-7723-4079-bd79-97e4452bd271/pptx/scripts/apply_theme.js");
  await applyTheme("rag-architecture.pptx", THEME);
  console.log("built rag-architecture.pptx");
})();
