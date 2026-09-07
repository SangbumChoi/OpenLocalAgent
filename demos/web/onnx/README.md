# LocalAgent in the browser (ONNX Runtime Web + WebGPU)

A fully client-side demo: the byte-level model runs in your browser via **ONNX Runtime Web**
(WebGPU, with a WASM fallback); tool **selection** is an in-page char-n-gram retriever and
arguments are grounded from your text. Nothing is sent to a server.

## Files
- `index.html` / `app.js` — the UI + agent (byte tokenizer, retriever, grounding, ONNX generation)
- `catalog.json` — the tool catalog the retriever indexes (regenerate from `agent/demo_tools.py`)
- `openlocalagent.onnx` — the model weights (you export this; not committed, it's ~4.5 MB)

## Run it
```bash
# 1) export the model to ONNX into this folder
openlocalagent export onnx results/runs/flywheel/ultra-tiny.pt demos/web/onnx/openlocalagent.onnx

# 2) serve the folder (ONNX Runtime Web needs http, not file://)
cd demos/web/onnx && python -m http.server 8000
# open http://localhost:8000  — a WebGPU-capable browser (Chrome/Edge) uses the GPU; else WASM
```

The **Agent** panel works even without the `.onnx` (selection is pure JS). The **Model** panel
loads `openlocalagent.onnx` and generates bytes on WebGPU to prove the network runs client-side.

## Notes
- The ONNX export is the full-sequence forward (no KV cache), so generation recomputes the prefix
  each step — fine for short demo prompts. For fast long generation, export a cached-decode graph.
- New tools need **no retraining**: add them to `catalog.json` and the retriever picks them up.
- Parity: the exported graph matches PyTorch (max|Δ|≈9e-6, 100% argmax agreement).
