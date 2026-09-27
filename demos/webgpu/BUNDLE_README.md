---
license: mit
tags: [onnx, webgpu, tool-calling, agent, openlocalagent]
base_model: danelcsb/openlocalagent-10m-matrix
---

# openlocalagent-10m-matrix · WebGPU bundle

The browser bundle of [`danelcsb/openlocalagent-10m-matrix`](https://huggingface.co/danelcsb/openlocalagent-10m-matrix):
a 10.5M-parameter tool-calling agent trained from scratch on open data, exported to ONNX (fp16)
for `onnxruntime-web` with the WebGPU execution provider. It is the model behind the OpenLocalAgent
workbench demo, which runs entirely in the browser.

| file | purpose |
|---|---|
| `model.fp16.onnx` | full-sequence logits graph |
| `cached/prefill.fp16.onnx`, `cached/decode.fp16.onnx` | prefill-then-KV-cached decode graphs |
| `cached/meta.json`, `cached/provenance.json`, `cached/single-decode.json` | graph contract and pinned hashes |
| `tokenizer.json` | the 16,384-token ByteLevel BPE the model was trained with (sha256 2f49a644…) |
| `meta.json` | markers, tool catalog the demo starts from, model parameters |
| `bundle-manifest.json` | sha256 of every artifact, checkpoint identity (`posttrain-la-10m-matrix`, step 1199) |

## Prompt contract

`openai_full_catalog_v1`: the tool catalog is rendered into the prompt as canonical JSON inside
`<|tool_catalog|>…</|tool_catalog|>`, followed by `<|user|>` request `<|assistant|>`, and the model
answers with `<tool_call>{"arguments":{…},"name":"…"}</tool_call>` or plain text. Tool responses are
fed back as `<|tool|><tool_response>…</tool_response>`. The context is 2,048 tokens; keep the
catalog small (a few tools), which is what the model was trained on.

## Use

Point the demo at this revision:

```
workbench.html?bundle=https://huggingface.co/danelcsb/openlocalagent-10m-matrix-webgpu/resolve/main/
```

Source, training recipe, evaluation and the workbench: the OpenLocalAgent repository.

## Provenance

Pretrain on a five-source open matrix (FineWeb-Edu, C4, Wikipedia, Cosmopedia, FineMath;
4.4B tokens), midtrain on non-benchmark agent data, posttrain on the train splits of the scored
benchmarks. The ten-benchmark receipt for this checkpoint: sound-tool-calling mean 19.1,
legacy ten-suite mean 17.6 (three seeds: 17.6 / 16.5 / 16.8).
