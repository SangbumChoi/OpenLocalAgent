// LocalAgent in the browser: ONNX Runtime Web (WebGPU/WASM) + JS byte tokenizer + char-ngram
// retriever + grounded tool calling. Mirrors the Python agent (model/tokenizer.py,
// agent/retriever.py, agent/constrained.py) closely enough for a client-side demo.

const DIM = 8192, NGRAMS = [3, 4, 5];
let session = null, catalog = null, toolVecs = null, backend = "wasm";

// ---- byte tokenizer (UTF-8) ----
const enc = new TextEncoder(), dec = new TextDecoder();
const encode = (s) => Array.from(enc.encode(s));
const decode = (ids) => dec.decode(Uint8Array.from(ids.filter((b) => b !== 0)));  // 0 = EOS

// ---- char n-gram hashing embedding (crc32, matches retriever.py) ----
const CRC = (() => { const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(str) { let c = 0xFFFFFFFF; const b = enc.encode(str);
  for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 0xFF] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
function embed(text) {
  const v = new Float32Array(DIM);
  const t = " " + text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim() + " ";
  for (const n of NGRAMS) for (let i = 0; i + n <= t.length; i++) v[crc32(t.slice(i, i + n)) % DIM] += 1;
  let nrm = 0; for (const x of v) nrm += x * x; nrm = Math.sqrt(nrm);
  if (nrm > 0) for (let i = 0; i < DIM; i++) v[i] /= nrm;
  return v;
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < DIM; i++) s += a[i] * b[i]; return s; };

// ---- grounding (schema format -> extract the argument from the prompt) ----
function ground(prompt, fmt) {
  if (fmt === "quoted") { const m = prompt.match(/'([^']+)'|"([^"]+)"/); return m ? (m[1] || m[2]) : null; }
  if (fmt === "path") { const m = prompt.match(/[\w./-]+\/[\w./-]*|[\w./-]+\.[a-z0-9]{1,5}\b/i); return m ? m[0].replace(/\.$/, "") : null; }
  if (fmt === "url") { const m = prompt.match(/(https?:\/\/)?[\w-]+(\.[\w-]+)+(\/[\w./-]*)?/); return m ? m[0].replace(/\.$/, "") : null; }
  // string: capitalized proper-noun, else tail after a preposition
  const caps = prompt.split(/\s+/).slice(1).filter((w) => /^[A-Z][a-z]/.test(w)).map((w) => w.replace(/[^A-Za-z]/g, ""));
  for (const p of ["for", "about", "to", "in", "on", "of", "up"]) {
    const i = prompt.toLowerCase().indexOf(" " + p + " ");
    if (i >= 0) return prompt.slice(i + p.length + 2).replace(/\s*(online|please)?[.?!]*$/i, "").trim();
  }
  return caps[0] || prompt.split(/\s+/).slice(1).join(" ").replace(/[.?!]+$/, "");
}

function retrieve(query, k = 5) {
  const q = embed(query);
  return toolVecs.map((v, i) => [catalog[i].name, dot(v, q)])
    .sort((a, b) => b[1] - a[1]).slice(0, k).map((x) => x[0]);
}

function runAgent() {
  const msg = document.getElementById("q").value.trim(); if (!msg) return;
  const top = retrieve(msg, 5);
  const tool = catalog.find((t) => t.name === top[0]);
  const val = ground(msg, tool.format);
  const args = {}; args[tool.arg] = val;
  document.getElementById("agentOut").textContent =
    `🔧 ${tool.name}(${JSON.stringify(args)})\n→ executed (stub)\n\ncandidates: ${top.join(", ")}`;
}

// ---- run the ONNX byte model (full-sequence, greedy; no KV cache in this export) ----
async function runModel() {
  if (!session) return;
  const out = document.getElementById("modelOut");
  let ids = encode(document.getElementById("p").value || "<|user|>hello");
  out.textContent = "generating…";
  for (let step = 0; step < 48; step++) {
    const t = new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    const r = await session.run({ input_ids: t });
    const logits = r.logits.data; const V = 256, last = (ids.length - 1) * V;
    let best = 0, bestv = -Infinity;
    for (let j = 0; j < V; j++) { const x = logits[last + j]; if (x > bestv) { bestv = x; best = j; } }
    if (best === 0) break;            // EOS
    ids.push(best);
    out.textContent = decode(ids);
  }
}

function ex(el) { document.getElementById("q").value = el.textContent; runAgent(); }

(async () => {
  const st = document.getElementById("status");
  catalog = await (await fetch("catalog.json")).json();
  toolVecs = catalog.map((t) => embed(`${t.name.replace(/_/g, " ")} ${t.description} ${t.examples.join(" ")}`));
  try {
    try { session = await ort.InferenceSession.create("localagent.onnx", { executionProviders: ["webgpu"] }); backend = "webgpu"; }
    catch { session = await ort.InferenceSession.create("localagent.onnx", { executionProviders: ["wasm"] }); backend = "wasm"; }
    st.innerHTML = `✅ ready — ${catalog.length} tools · model on <b>${backend}</b>`;
    document.getElementById("backend").textContent = `(ONNX Runtime · ${backend})`;
  } catch (e) {
    st.innerHTML = `⚠️ catalog loaded (${catalog.length} tools); model not found — put <code>localagent.onnx</code> here. Agent still works.`;
  }
})();
