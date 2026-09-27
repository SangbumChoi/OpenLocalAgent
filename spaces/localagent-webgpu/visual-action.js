/* Experimental screenshot-conditioned action probe for the isolated visual sidecar ABI. */
(function installVisualActionProbe(root) {
  "use strict";

  const query = new URLSearchParams(root.location?.search || "");
  const MODEL_FILE = query.get("graph") || "visual_action_model.fp16.onnx";
  const MANIFEST_FILE = query.get("manifest") || "visual-action-manifest.json";
  const ACTION_NAMES = Object.freeze([
    "click", "input_text", "long_press", "navigate_back", "open_app", "scroll", "wait",
  ]);
  const IMAGE_SIZE = 96;
  const textEncoder = new TextEncoder();
  let manifest = null;
  let session = null;
  let backend = null;

  const byId = (id) => document.getElementById(id);
  const status = (text, kind = "loading") => {
    byId("visual-status").className = `status ${kind}`;
    byId("visual-status-text").textContent = text;
  };

  async function sha256(bytes) {
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
  }

  async function fetchPinnedGraph() {
    const response = await fetch(MODEL_FILE);
    if (!response.ok) throw new Error(`Failed to fetch ${MODEL_FILE}: HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const expected = manifest.artifact || {};
    const actualHash = await sha256(bytes);
    if (bytes.byteLength !== expected.bytes || actualHash !== expected.sha256) {
      throw new Error(
        `Visual graph identity mismatch: got ${bytes.byteLength}/${actualHash}, ` +
        `expected ${expected.bytes}/${expected.sha256}.`
      );
    }
    return bytes;
  }

  function imageTensor(file) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        try {
          const canvas = document.createElement("canvas");
          canvas.width = IMAGE_SIZE;
          canvas.height = IMAGE_SIZE;
          const context = canvas.getContext("2d", { willReadFrequently: true });
          context.drawImage(image, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
          const rgba = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
          const chw = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE);
          for (let y = 0; y < IMAGE_SIZE; y += 1) {
            for (let x = 0; x < IMAGE_SIZE; x += 1) {
              const source = 4 * (y * IMAGE_SIZE + x);
              const target = y * IMAGE_SIZE + x;
              chw[target] = rgba[source] / 255;
              chw[IMAGE_SIZE * IMAGE_SIZE + target] = rgba[source + 1] / 255;
              chw[2 * IMAGE_SIZE * IMAGE_SIZE + target] = rgba[source + 2] / 255;
            }
          }
          URL.revokeObjectURL(image.src);
          resolve(new ort.Tensor("float32", chw, [1, 3, IMAGE_SIZE, IMAGE_SIZE]));
        } catch (error) {
          reject(error);
        }
      };
      image.onerror = () => reject(new Error("The selected image could not be decoded."));
      image.src = URL.createObjectURL(file);
    });
  }

  function contextTensor(task) {
    // Mirrors scripts/run_m714_androidcontrol_structured_visual_pilot.py: raw UTF-8 bytes,
    // not the BPE tokenizer used by the text release.
    const ids = textEncoder.encode(`Task: ${task}\nAction: `);
    const values = new BigInt64Array(ids.length);
    ids.forEach((value, index) => { values[index] = BigInt(value); });
    return {
      input_ids: new ort.Tensor("int64", values, [1, ids.length]),
      context_lengths: new ort.Tensor("int64", new BigInt64Array([BigInt(ids.length)]), [1]),
      length: ids.length,
    };
  }

  async function predict() {
    const file = byId("visual-image").files?.[0];
    if (!file) throw new Error("Choose a screenshot image first.");
    const task = byId("visual-task").value.trim();
    if (!task) throw new Error("Task text must not be empty.");
    const context = contextTensor(task);
    const started = performance.now();
    const outputs = await session.run({
      input_ids: context.input_ids,
      images: await imageTensor(file),
      context_lengths: context.context_lengths,
    });
    const logits = Array.from(outputs.action_logits.data, Number);
    let actionIndex = 0;
    for (let index = 1; index < logits.length; index += 1) {
      if (logits[index] > logits[actionIndex]) actionIndex = index;
    }
    const pointer = Array.from(outputs.pointer_xy.data, Number);
    return {
      schema_version: 1,
      backend,
      requested_backend: byId("visual-backend").value,
      task,
      input_contract: { raw_utf8_bytes: context.length, image: [1, 3, IMAGE_SIZE, IMAGE_SIZE], image_range: "float32 RGB [0,1]" },
      output_contract: { action_logits: [1, ACTION_NAMES.length], pointer_xy: [1, 2] },
      action: ACTION_NAMES[actionIndex],
      action_index: actionIndex,
      action_logits: logits,
      pointer_xy: pointer,
      inference_ms: performance.now() - started,
      graph: manifest.artifact,
      claim_boundary: "Runtime ABI probe only: no Android emulator, external side effect, official AndroidControl score, or per-node GPU placement claim.",
    };
  }

  async function chooseSession() {
    const requested = byId("visual-backend").value;
    if (!root.ort?.InferenceSession) throw new Error("onnxruntime-web failed to load.");
    const graph = await fetchPinnedGraph();
    const options = { executionProviders: [requested], graphOptimizationLevel: "all" };
    if (requested === "wasm") {
      root.ort.env.wasm.numThreads = 1;
      root.ort.env.wasm.proxy = false;
    }
    session = await root.ort.InferenceSession.create(graph, options);
    backend = requested;
  }

  async function init() {
    try {
      const response = await fetch(MANIFEST_FILE);
      if (!response.ok) throw new Error(`Failed to fetch ${MANIFEST_FILE}: HTTP ${response.status}`);
      manifest = await response.json();
      await chooseSession();
      byId("visual-backend-badge").hidden = false;
      byId("visual-backend-badge").textContent = backend.toUpperCase();
      byId("visual-run").disabled = false;
      status("Visual sidecar ready — predictions stay local.", "ready");
    } catch (error) {
      console.error(error);
      status(`Visual sidecar unavailable: ${error.message}`, "error");
    }
  }

  byId("visual-backend").addEventListener("change", async () => {
    byId("visual-run").disabled = true;
    status("Switching execution provider…", "loading");
    try {
      await chooseSession();
      byId("visual-backend-badge").textContent = backend.toUpperCase();
      byId("visual-run").disabled = false;
      status("Visual sidecar ready — predictions stay local.", "ready");
    } catch (error) {
      status(`Provider unavailable: ${error.message}`, "error");
    }
  });
  byId("visual-run").addEventListener("click", async () => {
    byId("visual-run").disabled = true;
    try {
      const result = await predict();
      byId("visual-result").hidden = false;
      byId("visual-output").textContent = JSON.stringify(result, null, 2);
    } catch (error) {
      status(`Prediction failed: ${error.message}`, "error");
    } finally {
      byId("visual-run").disabled = false;
    }
  });

  if (typeof window !== "undefined") window.__localAgentVisualActionPredict = predict;
  init();
})(typeof window !== "undefined" ? window : globalThis);
