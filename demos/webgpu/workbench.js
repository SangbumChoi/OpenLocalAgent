// OpenLocalAgent Workbench: a node canvas for one request's journey through the agent.
//
// Runs on top of app.js (loaded with window.__localAgentSkipInit so its own UI wiring is
// skipped). The model reads its tool catalog from META.tools, rendered into the prompt by the
// openai_full_catalog_v1 contract, so "registering a tool" is exactly: put it in META.tools.
// Every stage of a run is a node on the canvas that renders its own output inline; clicking a
// node opens the full output in the sheet. Each step of a multi-step run is a lane, and the
// observation of one lane feeds the prompt of the next.

(() => {
  "use strict";

  const STORAGE_KEY = "ola.workbench.registry.v2";
  const encoder = new TextEncoder();
  const $ = (id) => document.getElementById(id);

  // ================================================================ registry ====
  //  {name, description, schema, args, enabled, external, side_effect, executor}
  //  executor: {kind: "sim"|"mock"|"js"|"http", mock?, js?, method?, url?}
  let BUNDLE_TOOLS = [];
  let REGISTRY = [];
  let LAST_SELECTED = null;

  const PRESET_EXTERNAL_TOOLS = [
    { name: "stock_price", description: "Get the latest price of a stock by its ticker symbol.",
      schema: { type: "object", properties: { ticker: { type: "string" } }, required: ["ticker"] },
      side_effect: "read",
      executor: { kind: "mock", mock: '{"ticker":"{ticker}","price":118.42,"currency":"USD","asof":"now"}' } },
    { name: "translate_text", description: "Translate a piece of text into a target language.",
      schema: { type: "object", properties: { text: { type: "string" }, language: { type: "string" } }, required: ["text", "language"] },
      side_effect: "read",
      executor: { kind: "js", js: 'const table = { korean: "좋은 아침", japanese: "おはよう", french: "bonjour" };\nconst key = String(args.language || "").toLowerCase();\nreturn { translated: table[key] || `[${args.language}] ${args.text}`, source: args.text };' } },
    { name: "book_table", description: "Book a restaurant table for a party at a given time.",
      schema: { type: "object", properties: { restaurant: { type: "string" }, party_size: { type: "integer" }, time: { type: "string" } }, required: ["restaurant", "time"] },
      side_effect: "write",
      executor: { kind: "mock", mock: '{"confirmation":"RSV-2048","restaurant":"{restaurant}","time":"{time}","party_size":"{party_size}"}' } },
    { name: "ip_lookup", description: "Look up the public IP address of this device.",
      schema: { type: "object", properties: {}, required: [] },
      side_effect: "read",
      executor: { kind: "http", method: "GET", url: "https://api.ipify.org?format=json" } },
  ];

  // The model was trained on catalogs of a few tools, and 50 tools render to ~4,600 tokens
  // against a 2,048-token context. A scenario enables the handful it needs; the budget is live.
  const DEFAULT_ENABLED = new Set([
    "get_weather", "calculator", "web_search", "define", "get_news", "read_file", "write_file",
    "list_dir", "run_tests", "git_commit", "open_url", "set_timer", "send_email", "notion_write",
  ]);

  const argsOf = (schema) => Object.keys((schema && schema.properties) || {});
  const catalogEntry = (tool) => ({ name: tool.name, description: tool.description, args: argsOf(tool.schema), schema: tool.schema });

  function loadRegistry() {
    let stored = null;
    try { stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); } catch (_) { stored = null; }
    const disabled = stored ? new Set(stored.disabled_bundle_tools || []) : null;
    const bundle = BUNDLE_TOOLS.map((tool) => ({
      ...tool, args: argsOf(tool.schema), external: false,
      enabled: disabled ? !disabled.has(tool.name) : DEFAULT_ENABLED.has(tool.name),
      side_effect: sideEffectOfBundleTool(tool.name), executor: { kind: "sim" },
    }));
    const external = ((stored && stored.external) || (stored ? [] : PRESET_EXTERNAL_TOOLS)).map((tool) => ({
      ...tool, args: argsOf(tool.schema), enabled: tool.enabled !== false, external: true,
    }));
    REGISTRY = [...bundle, ...external];
  }

  function saveRegistry() {
    const payload = {
      disabled_bundle_tools: REGISTRY.filter((t) => !t.external && !t.enabled).map((t) => t.name),
      external: REGISTRY.filter((t) => t.external).map(({ name, description, schema, enabled, side_effect, executor }) =>
        ({ name, description, schema, enabled, side_effect, executor })),
    };
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(payload)); } catch (_) { /* private mode */ }
  }

  function sideEffectOfBundleTool(name) {
    const sets = [typeof SIDE_EFFECT_TOOLS !== "undefined" ? SIDE_EFFECT_TOOLS : null,
                  typeof DESTRUCTIVE_TOOLS !== "undefined" ? DESTRUCTIVE_TOOLS : null];
    return sets.some((s) => s && s.has(name)) ? "write" : "read";
  }

  // The one line that makes registration real: the model's catalog is the enabled registry.
  function syncCatalog() {
    META.tools = REGISTRY.filter((t) => t.enabled).map(catalogEntry);
    const tokens = META.tools.length ? TOKENIZER.encode(renderFullFunctionCatalog(META)).length : 0;
    const budget = META.max_seq_len - 96;
    const count = $("registry-count");
    count.textContent = `${META.tools.length}/${REGISTRY.length} · ${tokens}/${budget} tok`;
    count.style.color = tokens > budget ? "var(--coral)" : tokens > budget * 0.8 ? "var(--amber)" : "";
    renderRegistry();
    markExampleAvailability();
  }

  function renderRegistry() {
    const list = $("tool-list");
    const filter = ($("registry-search").value || "").toLowerCase();
    const onlyExternal = $("only-external").checked;
    list.innerHTML = "";
    for (const tool of REGISTRY) {
      if (onlyExternal && !tool.external) continue;
      if (filter && !(tool.name + " " + tool.description).toLowerCase().includes(filter)) continue;
      const li = document.createElement("li");
      li.className = "tool" + (tool.enabled ? "" : " disabled");
      const toggle = document.createElement("input");
      toggle.type = "checkbox"; toggle.checked = tool.enabled; toggle.title = "in the model's catalog";
      toggle.addEventListener("change", () => { tool.enabled = toggle.checked; saveRegistry(); syncCatalog(); });
      const body = document.createElement("div");
      const tags = [
        tool.external ? `<span class="tag external">external · ${esc(tool.executor.kind)}</span>` : '<span class="tag">bundle · simulated</span>',
        tool.side_effect === "write" ? '<span class="tag write">state-changing</span>' : "",
        LAST_SELECTED === tool.name ? '<span class="tag selected">selected</span>' : "",
      ].join("");
      body.innerHTML = `<div class="tool-name">${esc(tool.name)} <span class="args">(${tool.args.join(", ") || "no args"})</span></div>
        <div class="tool-desc">${esc(tool.description)}</div><div class="tags">${tags}</div>`;
      const edit = document.createElement("div");
      edit.className = "tool-edit";
      if (tool.external) {
        const e = document.createElement("button"); e.className = "btn ghost"; e.textContent = "edit";
        e.addEventListener("click", () => openToolDialog(tool));
        const d = document.createElement("button"); d.className = "btn ghost"; d.textContent = "delete";
        d.addEventListener("click", () => { REGISTRY = REGISTRY.filter((t) => t !== tool); saveRegistry(); syncCatalog(); });
        edit.append(e, d);
      } else {
        const v = document.createElement("button"); v.className = "btn ghost"; v.textContent = "schema";
        v.addEventListener("click", () => openSheet("tool", tool.name, `<pre>${esc(JSON.stringify(tool.schema, null, 2))}</pre>`));
        edit.append(v);
      }
      li.append(toggle, body, edit);
      list.appendChild(li);
    }
  }

  // An example is a scenario: the request and the tool combination the model should choose from.
  // Clicking it sets the registry to exactly that set, so the catalog on the canvas matches.
  function enableExactly(names) {
    const wanted = new Set(names);
    const missing = [...wanted].filter((n) => !REGISTRY.some((t) => t.name === n));
    REGISTRY.forEach((t) => { t.enabled = wanted.has(t.name); });
    saveRegistry(); syncCatalog();
    return missing;
  }

  function applyExample(chip) {
    if (chip.dataset.tools) {
      const missing = enableExactly(chip.dataset.tools.split(",").map((x) => x.trim()).filter(Boolean));
      if (missing.length) openSheet("registry", "Example needs tools that are not registered",
        `<p class="lead">Register or import these, or press Defaults to restore the presets:</p><pre>${esc(missing.join("\n"))}</pre>`);
    }
    $("wb-prompt").value = chip.textContent;
    $("wb-multistep").checked = chip.dataset.plan === "1";
    if (!$("wb-run").disabled) runRequest();
  }

  function markExampleAvailability() {
    const enabled = new Set(META.tools.map((t) => t.name));
    document.querySelectorAll(".chip[data-requires]").forEach((chip) => {
      chip.dataset.missing = enabled.has(chip.dataset.requires) ? "0" : "1";
      chip.title = enabled.has(chip.dataset.requires) ? "" : `needs tool "${chip.dataset.requires}" enabled`;
    });
  }

  // ============================================================= tool editor ====
  let EDITING = null;

  function openToolDialog(tool = null) {
    EDITING = tool;
    const form = $("tool-form");
    $("tool-dialog-title").textContent = tool ? `Edit ${tool.name}` : "Register a tool";
    form.reset();
    form.querySelector(".form-error")?.remove();
    if (tool) {
      form.name.value = tool.name;
      form.description.value = tool.description;
      form.schema.value = JSON.stringify(tool.schema, null, 2);
      form.executor.value = tool.executor.kind === "sim" ? "mock" : tool.executor.kind;
      form.side_effect.value = tool.side_effect || "read";
      if (tool.executor.mock) form.mock.value = tool.executor.mock;
      if (tool.executor.js) form.js.value = tool.executor.js;
      if (tool.executor.url) { form.url.value = tool.executor.url; form.method.value = tool.executor.method || "GET"; }
    }
    showExecutorFields(form.executor.value);
    $("tool-dialog").showModal();
  }

  function showExecutorFields(kind) {
    document.querySelectorAll("#tool-form [data-executor]").forEach((el) => { el.hidden = el.dataset.executor !== kind; });
  }

  function readToolForm() {
    const form = $("tool-form");
    const name = form.name.value.trim();
    const description = form.description.value.trim();
    let schema;
    try { schema = JSON.parse(form.schema.value); } catch (e) { throw new Error("parameters must be valid JSON: " + e.message); }
    if (!schema || schema.type !== "object" || typeof schema.properties !== "object") {
      throw new Error('parameters must be a JSON schema with "type": "object" and "properties"');
    }
    if (REGISTRY.some((t) => t.name === name && t !== EDITING)) throw new Error(`a tool named ${name} already exists`);
    assertNoPromptMarker(name, "tool name");
    assertNoPromptMarker(description, "tool description");
    const kind = form.executor.value;
    const executor = { kind };
    if (kind === "mock") executor.mock = form.mock.value;
    if (kind === "js") { executor.js = form.js.value; compileJs(executor.js); }
    if (kind === "http") {
      executor.method = form.method.value; executor.url = form.url.value.trim();
      if (!/^https?:\/\//.test(executor.url)) throw new Error("http executor needs an http(s) URL");
    }
    return { name, description, schema, args: argsOf(schema), enabled: true, external: true,
             side_effect: form.side_effect.value, executor };
  }

  // ================================================================ executors ====
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  function compileJs(body) {
    try { return new AsyncFunction("args", "ctx", body); } catch (e) { throw new Error("javascript executor does not compile: " + e.message); }
  }
  const fillTemplate = (template, args) => String(template).replace(/\{([a-zA-Z0-9_]+)\}/g, (_, key) =>
    args && key in args ? String(args[key]) : "");

  async function execute(tool, args, ctx) {
    const ex = tool.executor || { kind: "sim" };
    const started = performance.now();
    const request = { executor: ex.kind, tool: tool.name, args };
    let result;
    try {
      if (ex.kind === "sim") result = simResponse(tool.name, args || {});
      else if (ex.kind === "mock") { const text = fillTemplate(ex.mock, args); try { result = JSON.parse(text); } catch (_) { result = text; } }
      else if (ex.kind === "js") result = await compileJs(ex.js)(args || {}, ctx);
      else if (ex.kind === "http") {
        const url = fillTemplate(ex.url, args);
        request.url = url; request.method = ex.method || "GET";
        const init = { method: request.method, headers: {} };
        if (request.method !== "GET") { init.headers["content-type"] = "application/json"; init.body = JSON.stringify(args || {}); }
        const response = await fetch(url, init);
        const text = await response.text();
        request.status = response.status;
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${text.slice(0, 200)}`);
        try { result = JSON.parse(text); } catch (_) { result = text; }
      } else throw new Error("unknown executor " + ex.kind);
      return { ok: true, request, result, ms: performance.now() - started };
    } catch (error) {
      return { ok: false, request, error: String(error && error.message || error), ms: performance.now() - started };
    }
  }

  const observationText = (result) => { const t = typeof result === "string" ? result : JSON.stringify(result); return t.length > 600 ? t.slice(0, 600) + "…" : t; };

  // ==================================================== prompt with history ====
  // app.js builds the cached-decode prompt from the query alone. The workbench feeds prior tool
  // steps back through the same contract the model was trained on.
  const WB = { steps: [] };
  const OPENAI_FULL_CATALOG = typeof OPENAI_FULL_CATALOG_V1 !== "undefined" ? OPENAI_FULL_CATALOG_V1 : "openai_full_catalog_v1";
  // eslint-disable-next-line no-global-assign
  cachedActionPrompt = function workbenchActionPrompt(query, targetInputTokens = null) {
    const naturalText = renderFullCatalogContextText(query, WB.steps);
    const naturalIds = TOKENIZER.encode(naturalText);
    const assistantIds = META.markers?.assistant?.ids || TOKENIZER.encode(META.markers.assistant.text);
    const whitespaceIds = TOKENIZER.encode(" ");
    const ids = padPromptIds(naturalIds, assistantIds, whitespaceIds[0], targetInputTokens);
    return {
      ids, text: naturalText,
      inputBytes: encoder.encode(TOKENIZER.decode(ids, false)).length,
      naturalInputTokens: naturalIds.length, paddingTokens: ids.length - naturalIds.length,
      contextPaddingPlacement: targetInputTokens == null ? "none" : "before_assistant_marker",
      decisionInputTokens: ids.length, decisionFeatureIndex: ids.length - 1,
      promptContract: OPENAI_FULL_CATALOG, toolCatalogSize: META.tools.length,
    };
  };

  // =================================================================== canvas ====
  const STAGES = [
    { key: "request", label: "Request", family: "input" },
    { key: "catalog", label: "Catalog", family: "input" },
    { key: "prompt", label: "Prompt", family: "input" },
    { key: "prefill", label: "Prefill", family: "model" },
    { key: "decode", label: "Decode", family: "model" },
    { key: "parse", label: "Parse", family: "check" },
    { key: "validate", label: "Validate", family: "check" },
    { key: "safety", label: "Safety", family: "check" },
    { key: "dispatch", label: "Dispatch", family: "action" },
    { key: "observe", label: "Observation", family: "action" },
  ];
  // Each step is a 5 x 2 grid of nodes, so a run fits the canvas at a readable scale.
  const NODE_W = 236, NODE_H = 210, COL_GAP = 40, ROW_GAP = 34, COLS = 5, PAD = 40;
  const LANE_H = 2 * (NODE_H + ROW_GAP) + 26;
  let LANES = [];            // LANES[lane][key] = {status, ms, preview, inspect}
  let ACTIVE = null;         // {lane, key}
  const VIEW = { x: 0, y: 0, k: 1 };

  function newLane() {
    const lane = {};
    for (const s of STAGES) lane[s.key] = { status: "pending", ms: null, preview: "", inspect: null };
    LANES.push(lane);
    renderCanvas();
    return LANES.length - 1;
  }

  function setNode(lane, key, patch) {
    Object.assign(LANES[lane][key], patch);
    renderCanvas();
    if (ACTIVE && ACTIVE.lane === lane && ACTIVE.key === key) openNode(lane, key);
  }

  const nodePos = (lane, i) => ({
    x: PAD + (i % COLS) * (NODE_W + COL_GAP),
    y: PAD + 24 + lane * LANE_H + Math.floor(i / COLS) * (NODE_H + ROW_GAP),
  });

  function renderCanvas() {
    const nodes = $("nodes");
    const edges = $("edges");
    nodes.innerHTML = "";
    let paths = "";
    LANES.forEach((lane, li) => {
      const label = document.createElement("div");
      label.className = "lane-label";
      label.style.left = `${PAD}px`; label.style.top = `${PAD + li * LANE_H}px`;
      label.textContent = `step ${li + 1}`;
      nodes.appendChild(label);
      STAGES.forEach((stage, col) => {
        const s = lane[stage.key];
        const { x, y } = nodePos(li, col);
        const el = document.createElement("div");
        el.className = `node family-${stage.family} ${s.status}` + (ACTIVE && ACTIVE.lane === li && ACTIVE.key === stage.key ? " active" : "");
        el.style.left = `${x}px`; el.style.top = `${y}px`;
        el.innerHTML = `<span class="node-port in"></span><span class="node-port out"></span>
          <div class="node-head"><span class="node-index">${String(col + 1).padStart(2, "0")}</span>
            <span class="node-title">${stage.label}</span>
            <span class="node-ms">${s.ms != null ? fmt(s.ms) + " ms" : ""}</span><span class="node-status"></span></div>
          <div class="node-body">${s.preview || '<span class="sub">waiting</span>'}</div>`;
        el.addEventListener("click", (e) => { e.stopPropagation(); openNode(li, stage.key); });
        nodes.appendChild(el);
        if (col > 0) {
          const prev = lane[STAGES[col - 1].key];
          const from = nodePos(li, col - 1), to = { x, y };
          const cls = s.status === "running" ? "live" : (s.status === "pending" && prev.status !== "running") ? "dead" : "";
          if (col % COLS === 0) {
            // Row wrap: leave the last node of the row from its bottom, enter the next row's first node from its top.
            paths += `<path class="${cls}" d="M ${from.x + NODE_W / 2} ${from.y + NODE_H} C ${from.x + NODE_W / 2} ${from.y + NODE_H + 40}, ${to.x + NODE_W / 2} ${to.y - 40}, ${to.x + NODE_W / 2} ${to.y}" />`;
          } else {
            paths += edge(from.x + NODE_W, from.y + 60, to.x, to.y + 60, cls);
          }
        }
      });
      if (li > 0) {
        // Observation of the previous lane feeds this lane's prompt.
        const from = nodePos(li - 1, STAGES.length - 1), to = nodePos(li, 2);
        paths += `<path class="feedback" d="M ${from.x + NODE_W / 2} ${from.y + NODE_H} C ${from.x + NODE_W / 2} ${from.y + NODE_H + 60}, ${to.x + NODE_W / 2} ${to.y - 60}, ${to.x + NODE_W / 2} ${to.y}" />`;
      }
    });
    edges.innerHTML = paths;
    applyView();
  }

  const edge = (x1, y1, x2, y2, cls) => `<path class="${cls}" d="M ${x1} ${y1} C ${x1 + 22} ${y1}, ${x2 - 22} ${y2}, ${x2} ${y2}" />`;

  function applyView() {
    const t = `translate(${VIEW.x}px, ${VIEW.y}px) scale(${VIEW.k})`;
    $("nodes").style.transform = t;
    $("edges").style.transform = t;
  }

  function fitView() {
    const canvas = $("canvas");
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const graphW = PAD * 2 + COLS * (NODE_W + COL_GAP) - COL_GAP;
    const graphH = PAD * 2 + Math.max(1, LANES.length) * LANE_H;
    VIEW.k = Math.min(1, (w - 24) / graphW, (h - 24) / graphH);
    VIEW.x = Math.max(12, (w - graphW * VIEW.k) / 2);
    VIEW.y = 12;
    applyView();
  }

  function wireCanvas() {
    const canvas = $("canvas");
    let drag = null;
    canvas.addEventListener("mousedown", (e) => { if (e.target.closest(".node")) return; drag = { x: e.clientX - VIEW.x, y: e.clientY - VIEW.y }; canvas.classList.add("dragging"); });
    window.addEventListener("mousemove", (e) => { if (!drag) return; VIEW.x = e.clientX - drag.x; VIEW.y = e.clientY - drag.y; applyView(); });
    window.addEventListener("mouseup", () => { drag = null; canvas.classList.remove("dragging"); });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left, py = e.clientY - rect.top;
      const k = Math.min(2, Math.max(0.25, VIEW.k * (e.deltaY < 0 ? 1.1 : 0.9)));
      VIEW.x = px - (px - VIEW.x) * (k / VIEW.k); VIEW.y = py - (py - VIEW.y) * (k / VIEW.k); VIEW.k = k;
      applyView();
    }, { passive: false });
    window.addEventListener("resize", fitView);
  }

  // ================================================================== sheet ====
  function openNode(lane, key) {
    ACTIVE = { lane, key };
    renderCanvas();
    const s = LANES[lane][key];
    const stage = STAGES.find((x) => x.key === key);
    openSheet(`step ${lane + 1} · ${stage.family} · ${s.status}${s.ms != null ? ` · ${fmt(s.ms)} ms` : ""}`, stage.label,
      s.inspect ? s.inspect() : '<p class="lead">Nothing recorded for this node yet.</p>');
  }
  function openSheet(kicker, title, html) {
    $("sheet-kicker").textContent = kicker; $("sheet-title").textContent = title; $("sheet-body").innerHTML = html;
    const wasHidden = $("sheet").hidden;
    $("sheet").hidden = false;
    if (wasHidden) fitView();   // the canvas just lost the sheet's width; refit once, not on every click
  }

  // ================================================================ one step ====
  async function runStep(query, index, temperature) {
    const L = newLane();
    const step = { index, query };
    const set = (key, patch) => setNode(L, key, patch);

    // 01 request
    set("request", { status: "done",
      preview: `<div class="big" style="font-size:.9rem;font-family:var(--sans)">${esc(query)}</div><div class="sub">${encoder.encode(query).length} bytes · ${WB.steps.length} prior step(s)</div>`,
      inspect: () => `<dl><dt>text</dt><dd>${esc(query)}</dd><dt>bytes</dt><dd>${encoder.encode(query).length}</dd><dt>history</dt><dd>${WB.steps.length} prior tool step(s) fed back</dd></dl>` });

    // 02 catalog
    const catalogText = renderFullFunctionCatalog(META);
    const catalogTokens = TOKENIZER.encode(catalogText).length;
    const budget = META.max_seq_len;
    set("catalog", { status: "done",
      preview: `<div class="big">${META.tools.length} tools</div><div class="sub">${catalogTokens} of ${budget} context tokens</div>
        <div class="bar-mini"><i style="width:${Math.min(100, 100 * catalogTokens / budget).toFixed(1)}%"></i></div>
        <div class="sub" style="margin-top:.35rem">${META.tools.map((t) => esc(t.name)).join(" · ")}</div>`,
      inspect: () => `<dl><dt>contract</dt><dd>${OPENAI_FULL_CATALOG}</dd><dt>tools</dt><dd>${META.tools.length}</dd><dt>tokens</dt><dd>${catalogTokens} of ${budget}</dd></dl>
        <p class="lead">Exactly what the model reads, canonical JSON, one entry per enabled tool:</p><pre>${esc(pretty(catalogText))}</pre>` });

    // 03 prompt
    const prompt = cachedActionPrompt(query);
    const promptTokens = prompt.ids.length;
    const left = budget - promptTokens;
    set("prompt", { status: left < 32 ? "failed" : "done",
      preview: `<div class="big">${promptTokens} tok</div><div class="sub">${left} left for the answer</div>
        <div class="bar-mini"><i style="width:${Math.min(100, 100 * promptTokens / budget).toFixed(1)}%"></i></div>
        ${tokenStrip(prompt.ids.slice(-14))}`,
      inspect: () => `<dl><dt>tokens</dt><dd>${promptTokens} (${prompt.naturalInputTokens} natural + ${prompt.paddingTokens} padding)</dd><dt>bytes</dt><dd>${prompt.inputBytes}</dd><dt>decision index</dt><dd>${prompt.decisionFeatureIndex}</dd></dl>
        <p class="lead">Rendered prompt (catalog blue, user amber, assistant mint, tool responses pink):</p><pre>${highlightPrompt(prompt.text)}</pre>
        <p class="lead">First and last token pieces (hover for ids):</p>${tokenStrip(prompt.ids.slice(0, 24))} … ${tokenStrip(prompt.ids.slice(-24))}` });
    if (left < 32) throw new Error(`prompt uses ${promptTokens} of ${budget} tokens; disable some tools`);

    // 04–05 model
    set("prefill", { status: "running", preview: '<div class="sub">forward over the prompt…</div>' });
    set("decode", { status: "pending" });
    const out = await rawAutoregressiveAction(query, { temperature, seed: `wb:${index}:${query}` });
    const t = out.timing || {};
    const generatedIds = TOKENIZER.encode(out.generated_text || "");
    step.out = out;
    set("prefill", { status: "done", ms: t.ttft_ms,
      preview: `<div class="big">${fmt(t.ttft_ms)} ms</div><div class="sub">time to first token · ${promptTokens} tokens in · ${esc(BACKEND)}</div>`,
      inspect: () => `<dl><dt>time to first token</dt><dd>${fmt(t.ttft_ms)} ms</dd><dt>backend</dt><dd>${esc(BACKEND)}</dd><dt>strategy</dt><dd>${esc(out.decode_strategy || "")}</dd><dt>policy</dt><dd>${esc(out.policy || "")}</dd></dl>` });
    // Decode wall time from the per-token figure: the timing block's inference total excludes prefill.
    const decodeMs = t.tpot_ms == null ? null : t.tpot_ms * Math.max(0, (out.decode_steps || 1) - 1);
    set("decode", { status: "done", ms: decodeMs,
      preview: `<div class="big">${out.output_tokens} tok</div><div class="sub">${t.tpot_ms == null ? "–" : t.tpot_ms.toFixed(1)} ms/tok · ${t.tpot_ms ? (1000 / t.tpot_ms).toFixed(0) : "–"} tok/s · ${esc(out.stop_reason)}</div>${tokenStrip(generatedIds.slice(0, 40))}`,
      inspect: () => `<dl><dt>tokens</dt><dd>${out.output_tokens} (${out.decode_steps} decode steps)</dd><dt>per token</dt><dd>${t.tpot_ms == null ? "–" : t.tpot_ms.toFixed(2)} ms</dd><dt>stop</dt><dd>${esc(out.stop_reason)}</dd><dt>selection</dt><dd>${esc(out.token_selection || "")}</dd></dl>
        <p class="lead">Generated text, re-tokenized for display:</p>${tokenStrip(generatedIds)}<pre>${esc(out.generated_text || "")}</pre>` });

    // 06 parse
    const parsedOk = !out.parse_failure;
    const action = out.abstain ? { abstain: true } : (out.tool ? { name: out.tool, arguments: out.args } : null);
    set("parse", { status: parsedOk ? "done" : "failed", ms: t.parse_validate_ms,
      preview: `<span class="verdict ${parsedOk ? "" : "bad"}">${esc(out.parse_kind || "?")}</span><pre>${esc(action ? JSON.stringify(action, null, 1) : (out.parse_error || out.generated_text || ""))}</pre>`,
      inspect: () => `<dl><dt>kind</dt><dd>${esc(out.parse_kind || "")}</dd><dt>error</dt><dd>${esc(out.parse_error || "none")}</dd></dl><p class="lead">Parsed action:</p><pre>${esc(JSON.stringify(action ?? { text: out.generated_text }, null, 2))}</pre>` });
    if (!parsedOk) { step.stopped = "parse"; return finish(step, "parse failed"); }
    if (out.parse_kind === "direct_text" || !out.tool) {
      step.text = out.generated_text;
      for (const k of ["validate", "safety", "dispatch", "observe"]) set(k, { status: "done", preview: '<div class="sub">not needed — text answer</div>' });
      return finish(step, "answered in text");
    }

    // 07 validate
    const spec = REGISTRY.find((r) => r.name === out.tool && r.enabled);
    const required = (spec && spec.schema && spec.schema.required) || [];
    const missing = required.filter((k) => !(out.args && k in out.args));
    const unknownArgs = Object.keys(out.args || {}).filter((k) => spec && !(k in (spec.schema.properties || {})));
    const valid = Boolean(spec) && out.schema_valid !== false && missing.length === 0;
    LAST_SELECTED = out.tool;
    renderRegistry();
    set("validate", { status: valid ? "done" : "failed",
      preview: `<span class="verdict ${valid ? "" : "bad"}">${!spec ? "unknown tool" : valid ? "schema ok" : "invalid"}</span>
        <div class="kv"><span class="k">tool</span><span class="v">${esc(out.tool)}</span><span class="k">required</span><span class="v">${esc(required.join(", ") || "none")}</span>
        <span class="k">missing</span><span class="v">${esc(missing.join(", ") || "none")}</span><span class="k">unknown</span><span class="v">${esc(unknownArgs.join(", ") || "none")}</span></div>`,
      inspect: () => `<dl><dt>tool known</dt><dd>${spec ? "yes" : "no — not in the enabled catalog"}</dd><dt>schema valid</dt><dd>${out.schema_valid !== false ? "yes" : "no: " + esc(out.validation_error || "")}</dd><dt>required</dt><dd>${required.join(", ") || "none"}</dd><dt>missing</dt><dd>${missing.join(", ") || "none"}</dd><dt>unknown args</dt><dd>${unknownArgs.join(", ") || "none"}</dd></dl>
        <p class="lead">Arguments as generated:</p><pre>${esc(JSON.stringify(out.args, null, 2))}</pre><p class="lead">Schema the model was shown:</p><pre>${esc(JSON.stringify(spec ? spec.schema : null, null, 2))}</pre>` });
    step.tool = out.tool; step.args = out.args;
    if (!valid) { step.stopped = "validate"; return finish(step, "invalid call"); }

    // 08 safety
    const safety = actionSafetyPolicy(out, query, { args: out.args, toolSpec: catalogEntry(spec), untrustedText: WB.steps.at(-1)?.response || "" });
    let status = safety.status;
    let note = safety.reason || safety.note || "";
    if (status === "allowed" && spec.side_effect === "write") {
      const ok = window.confirm(`"${out.tool}" is registered as state-changing.\n\n${JSON.stringify(out.args)}\n\nDispatch it?`);
      status = ok ? "allowed" : "confirmation_declined";
      note = ok ? "state-changing tool, confirmed by the user" : "state-changing tool, declined by the user";
    }
    step.safety = { ...safety, status, note };
    set("safety", { status: status === "allowed" ? "done" : "failed",
      preview: `<span class="verdict ${status === "allowed" ? "" : "bad"}">${esc(status.replace(/_/g, " "))}</span><div class="sub">${esc(note || "no signal")} · ${spec.side_effect}</div>`,
      inspect: () => `<span class="verdict ${status === "allowed" ? "" : "bad"}">${esc(status)}</span><dl><dt>reason</dt><dd>${esc(note || "no signal")}</dd><dt>side effect</dt><dd>${spec.side_effect}</dd></dl><pre>${esc(JSON.stringify(safety, null, 2))}</pre>` });
    if (status !== "allowed") { step.stopped = "safety"; return finish(step, status); }

    // 09 dispatch
    set("dispatch", { status: "running", preview: `<div class="sub">${esc(spec.executor.kind)} executor…</div>` });
    const exec = await execute(spec, out.args, { query, steps: WB.steps });
    step.exec = exec;
    set("dispatch", { status: exec.ok ? "done" : "failed", ms: exec.ms,
      preview: `<div class="kv"><span class="k">executor</span><span class="v">${esc(spec.executor.kind)}</span>${spec.executor.url ? `<span class="k">url</span><span class="v">${esc(fillTemplate(spec.executor.url, out.args))}</span>` : ""}
        <span class="k">outcome</span><span class="v">${exec.ok ? "ok" : esc(exec.error)}</span></div>${exec.ok ? `<pre>${esc(observationText(exec.result).slice(0, 220))}</pre>` : ""}`,
      inspect: () => `<dl><dt>executor</dt><dd>${esc(spec.executor.kind)}</dd><dt>outcome</dt><dd>${exec.ok ? "ok" : "error: " + esc(exec.error)}</dd><dt>time</dt><dd>${fmt(exec.ms)} ms</dd></dl>
        <p class="lead">Request:</p><pre>${esc(JSON.stringify(exec.request, null, 2))}</pre>${exec.ok ? `<p class="lead">Result:</p><pre>${esc(JSON.stringify(exec.result, null, 2))}</pre>` : ""}` });
    if (!exec.ok) { step.stopped = "dispatch"; return finish(step, "dispatch failed"); }

    // 10 observation
    step.observation = exec.result;
    step.observation_text = observationText(exec.result);
    const feedback = META.markers.assistant.text + canonicalToolCompletion({ tool: step.tool, args: step.args }, META) + BPE_EOS_MARKER + META.markers.tool.text + META.markers.tool_response_open.text + step.observation_text + META.markers.tool_response_close.text;
    set("observe", { status: "done",
      preview: `<pre>${esc(step.observation_text.slice(0, 260))}</pre>`,
      inspect: () => `<p class="lead">Fed back to the model as the tool response of this step:</p><pre>${esc(step.observation_text)}</pre><p class="lead">In the next prompt it appears as:</p><pre>${esc(feedback)}</pre>` });
    return finish(step, "dispatched");
  }

  const finish = (step, outcome) => { step.outcome = outcome; return step; };

  // ================================================================ run loop ====
  let TRACE = [];

  async function runRequest() {
    const query = $("wb-prompt").value.trim();
    if (!query) return;
    const multistep = $("wb-multistep").checked;
    const maxSteps = multistep ? Math.max(1, Math.min(8, Number($("wb-max-steps").value) || 4)) : 1;
    const temperature = Math.max(0, Number($("wb-temp").value) || 0);
    $("wb-run").disabled = true;
    WB.steps = []; TRACE = []; LANES = []; ACTIVE = null;
    $("sheet").hidden = true;
    renderCanvas(); fitView();
    const started = performance.now();
    try {
      for (let i = 0; i < maxSteps; i++) {
        const step = await runStep(query, i, temperature);
        TRACE.push(step);
        if (!multistep || step.stopped || step.text || !step.tool) break;
        WB.steps.push({ tool: step.tool, args: step.args, response: step.observation_text });
        fitView();
      }
      renderMetrics(performance.now() - started);
      const last = TRACE.at(-1);
      openNode(TRACE.length - 1, last && last.stopped ? last.stopped : (last && last.text ? "parse" : "observe"));
      console.log("WB_RESULT " + JSON.stringify({ query, steps: TRACE.map((st) => ({
        tool: st.tool || null, args: st.args || null, outcome: st.outcome, text: st.text || null,
        observation: st.observation_text || null, parse_kind: st.out?.parse_kind, generated: st.out?.generated_text,
        input_tokens: st.out?.input_tokens, output_tokens: st.out?.output_tokens, ttft_ms: st.out?.timing?.ttft_ms, tpot_ms: st.out?.timing?.tpot_ms })),
        total_ms: performance.now() - started }));
    } catch (error) {
      const lane = LANES.length - 1;
      const running = lane >= 0 ? STAGES.find((s) => LANES[lane][s.key].status === "running") : null;
      if (running) setNode(lane, running.key, { status: "failed", preview: `<span class="verdict bad">error</span><pre>${esc(String(error.message || error))}</pre>` });
      openSheet("error", "Run failed", `<pre>${esc(String(error.stack || error))}</pre>`);
      console.log("WB_RESULT " + JSON.stringify({ query, error: String(error && error.stack || error) }));
    } finally {
      $("wb-run").disabled = false;
    }
  }

  function renderMetrics(totalMs) {
    const outs = TRACE.map((s) => s.out).filter(Boolean);
    const sum = (f) => outs.reduce((a, o) => a + (f(o) || 0), 0);
    const tokens = sum((o) => o.output_tokens);
    const decode = sum((o) => (o.timing.tpot_ms || 0) * Math.max(0, (o.decode_steps || 1) - 1));
    const last = TRACE.at(-1);
    $("wb-metrics").innerHTML = [
      `steps <b>${TRACE.length}</b>`, `total <b>${fmt(totalMs)} ms</b>`,
      `first TTFT <b>${outs.length ? fmt(outs[0].timing.ttft_ms) + " ms" : "–"}</b>`,
      `decode <b>${decode && tokens ? (tokens / (decode / 1000)).toFixed(0) + " tok/s" : "–"}</b>`,
      `catalog <b>${META.tools.length} tools</b>`,
      `outcome <b>${esc(last ? last.outcome : "–")}</b>${last && last.tool ? ` · <b>${esc(last.tool)}</b>` : ""}`,
    ].map((x) => `<span>${x}</span>`).join("");
  }

  // ================================================================= helpers ====
  function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  function fmt(ms) { return ms == null || Number.isNaN(ms) ? "–" : (ms < 10 ? ms.toFixed(1) : ms.toFixed(0)); }
  function pretty(catalogText) {
    const inner = catalogText.slice(TOOL_CATALOG_OPEN.length, -TOOL_CATALOG_CLOSE.length);
    try { return TOOL_CATALOG_OPEN + "\n" + JSON.stringify(JSON.parse(inner), null, 1) + "\n" + TOOL_CATALOG_CLOSE; } catch (_) { return catalogText; }
  }
  function highlightPrompt(text) {
    const m = META.markers;
    const cat = text.indexOf(TOOL_CATALOG_CLOSE);
    let html = "";
    if (cat >= 0) {
      html += `<span class="hl-cat">${esc(text.slice(0, TOOL_CATALOG_OPEN.length))}…${esc(text.slice(cat - 80, cat + TOOL_CATALOG_CLOSE.length))}</span>`;
      text = text.slice(cat + TOOL_CATALOG_CLOSE.length);
    }
    const markers = [m.user.text, m.assistant.text, m.tool.text, m.tool_response_open.text, m.tool_response_close.text, BPE_EOS_MARKER];
    const re = new RegExp("(" + markers.map((x) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")");
    let cls = "";
    for (const part of text.split(re)) {
      if (!part) continue;
      if (markers.includes(part)) {
        html += `<span class="hl-marker">${esc(part)}</span>`;
        cls = part === m.user.text ? "hl-user" : part === m.assistant.text ? "hl-asst" : part === m.tool.text ? "hl-tool" : cls;
      } else html += `<span class="${cls}">${esc(part)}</span>`;
    }
    return html;
  }
  function tokenStrip(ids) {
    const markerIds = new Set(Object.values(META.markers).flatMap((mk) => mk.ids || []));
    return `<div class="tokens">${ids.map((id) => {
      const piece = TOKENIZER.decode([id], false);
      const c = id === META.eos_id ? "eos" : markerIds.has(id) ? "marker" : "";
      return `<span class="tok ${c}" title="id ${id}">${esc(piece.replace(/\n/g, "⏎") || "·")}</span>`;
    }).join("")}</div>`;
  }

  // ==================================================================== boot ====
  function wire() {
    $("wb-run").addEventListener("click", runRequest);
    $("wb-prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runRequest(); } });
    document.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => applyExample(chip)));
    $("add-tool").addEventListener("click", () => openToolDialog(null));
    $("tool-form").executor.addEventListener("change", (e) => showExecutorFields(e.target.value));
    $("tool-form").addEventListener("submit", (e) => {
      if (e.submitter && e.submitter.value === "cancel") return;
      e.preventDefault();
      try {
        const tool = readToolForm();
        if (EDITING) Object.assign(EDITING, tool); else REGISTRY.push(tool);
        saveRegistry(); syncCatalog(); $("tool-dialog").close();
      } catch (error) {
        $("tool-form").querySelector(".form-error")?.remove();
        const p = document.createElement("p"); p.className = "form-error"; p.textContent = error.message;
        $("tool-form").querySelector(".dialog-actions").before(p);
      }
    });
    $("export-tools").addEventListener("click", () => {
      const ext = REGISTRY.filter((t) => t.external).map(({ name, description, schema, side_effect, executor, enabled }) => ({ name, description, schema, side_effect, executor, enabled }));
      openSheet("registry", "Export", `<p class="lead">Copy this JSON; Import accepts the same shape.</p><pre>${esc(JSON.stringify(ext, null, 2))}</pre>`);
    });
    $("import-tools").addEventListener("click", () => $("import-file").click());
    $("import-file").addEventListener("change", async (e) => {
      const file = e.target.files[0]; if (!file) return;
      try {
        const tools = JSON.parse(await file.text());
        if (!Array.isArray(tools)) throw new Error("expected a JSON array of tools");
        for (const tool of tools) {
          if (!tool.name || !tool.description || !tool.schema) throw new Error(`tool entry is incomplete: ${JSON.stringify(tool).slice(0, 80)}`);
          REGISTRY = REGISTRY.filter((t) => t.name !== tool.name);
          REGISTRY.push({ ...tool, args: argsOf(tool.schema), enabled: tool.enabled !== false, external: true,
            side_effect: tool.side_effect || "read", executor: tool.executor || { kind: "mock", mock: "ok" } });
        }
        saveRegistry(); syncCatalog();
      } catch (error) { openSheet("registry", "Import failed", `<pre>${esc(error.message)}</pre>`); }
      e.target.value = "";
    });
    $("reset-tools").addEventListener("click", () => { localStorage.removeItem(STORAGE_KEY); loadRegistry(); saveRegistry(); syncCatalog(); });
    $("enable-none").addEventListener("click", () => { REGISTRY.forEach((t) => { t.enabled = false; }); saveRegistry(); syncCatalog(); });
    $("registry-search").addEventListener("input", renderRegistry);
    $("only-external").addEventListener("change", renderRegistry);
    $("toggle-registry").addEventListener("click", () => { $("registry").hidden = !$("registry").hidden; fitView(); });
    $("fit-view").addEventListener("click", fitView);
    $("sheet-close").addEventListener("click", () => { $("sheet").hidden = true; ACTIVE = null; renderCanvas(); fitView(); });
    wireCanvas();
  }

  async function boot() {
    const hold = new URLSearchParams(window.location.search).get("hold");
    if (hold) { const img = new Image(); img.src = `http://127.0.0.1:8768/hold?ms=${encodeURIComponent(hold)}`; document.body.appendChild(img); img.hidden = true; }
    wire();
    newLane(); fitView();
    setStatus("loading", "Loading model… (first load downloads & caches the weights)");
    try {
      await loadBundle();
      BUNDLE_TOOLS = META.tools.map((t) => ({ ...t }));
      loadRegistry();
      syncCatalog();
      $("policy-badge").textContent = `raw autoregressive · ${META.model_parameters?.toLocaleString?.() || "?"} params`;
      setStatus("ready", "Model ready · runs in this tab", BACKEND);
      $("wb-run").disabled = false;
      // Deep link: ?q=<request>&tools=a,b,c&steps=<n>&run=1 preloads one exact scenario.
      const params = new URLSearchParams(window.location.search);
      if (params.get("tools")) {
        const wanted = new Set(params.get("tools").split(",").map((x) => x.trim()).filter(Boolean));
        REGISTRY.forEach((t) => { t.enabled = wanted.has(t.name); });
        syncCatalog();
      }
      if (params.get("example") != null) {
        const chip = document.querySelectorAll(".chip")[Number(params.get("example"))];
        if (chip) applyExample(chip);
      }
      if (params.get("q")) {
        $("wb-prompt").value = params.get("q");
        if (params.get("steps")) { $("wb-multistep").checked = true; $("wb-max-steps").value = params.get("steps"); }
        if (params.get("run") === "1") runRequest();
      }
    } catch (error) {
      console.error(error);
      setStatus("error", "Failed to load the model bundle: " + error.message);
    }
  }

  boot();
})();
