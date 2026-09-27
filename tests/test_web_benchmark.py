from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WEB_BENCHMARK = ROOT / "demos" / "webgpu" / "benchmark.js"
WEB_APP = ROOT / "demos" / "webgpu" / "app.js"
WEB_BENCHMARK_HTML = ROOT / "demos" / "webgpu" / "benchmark.html"
PRETRAIN_PAPER_CONFIG = ROOT / "configs" / "data" / "pretrain-paper.yaml"
WEB_BENCHMARK_CASES = (
    ROOT / "demos" / "webgpu" / "benchmark-cases.json"
)


def test_action_suite_declares_exact_prompt_only_holdout_scope():
    suite = json.loads(WEB_BENCHMARK_CASES.read_text())
    assert suite["holdout_contract"] == {
        "case_sensitive": False,
        "normalization": "Unicode NFKC, Unicode whitespace collapse, Unicode casefold",
        "primitive_value_disjointness_claimed": False,
        "template_disjointness_claimed": False,
        "training_exclusion": "canonical normalized query equality",
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_deadline_success_requires_exact_and_schema_valid():
    records = [
        {
            "harness_ttfa_ms": 100,
            "runtime_ttfa_ms": 99,
            "ttfa_ms": 100,
            "tokenize_ms": 1,
            "inference_ms": 95,
            "dispatch_ms": 4,
            "success": True,
            "schema_valid": False,
        },
        {
            "harness_ttfa_ms": 200,
            "runtime_ttfa_ms": 198,
            "ttfa_ms": 200,
            "tokenize_ms": 2,
            "inference_ms": 190,
            "dispatch_ms": 8,
            "success": True,
            "schema_valid": True,
        },
        {
            "harness_ttfa_ms": 50,
            "runtime_ttfa_ms": 50,
            "ttfa_ms": 50,
            "tokenize_ms": None,
            "inference_ms": None,
            "dispatch_ms": None,
            "success": False,
            "schema_valid": False,
        },
    ]
    script = """
global.window = { __localAgentSkipInit: true };
const { summarizeBenchmark } = require(process.argv[1]);
const records = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(summarizeBenchmark(records)));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_BENCHMARK), json.dumps(records)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["exact_action_accuracy"] == pytest.approx(2 / 3)
    assert summary["schema_valid_rate"] == pytest.approx(1 / 3)
    assert summary["latency_ms"]["harness_ttfa_ms"]["count"] == 3
    assert summary["deadline_attainment_ms"]["250"]["on_time_rate"] == 1
    assert summary["deadline_attainment_ms"]["250"]["opportunities"] == 3
    assert summary["deadline_attainment_ms"]["250"]["useful"] == 1
    assert summary["deadline_attainment_ms"]["250"]["success_at_deadline"] == pytest.approx(
        1 / 3
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_case_shuffle_is_seeded_and_reproducible():
    script = """
global.window = { __localAgentSkipInit: true };
const { shuffledCases } = require(process.argv[1]);
const items = Array.from({ length: 20 }, (_, id) => ({ id }));
const ids = (seed) => shuffledCases(items, seed).map((item) => item.id);
process.stdout.write(JSON.stringify({ a: ids("a"), again: ids("a"), b: ids("b") }));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_BENCHMARK)],
        check=True,
        capture_output=True,
        text=True,
    )
    orders = json.loads(result.stdout)
    assert orders["a"] == orders["again"]
    assert orders["a"] != orders["b"]
    assert sorted(orders["a"]) == list(range(20))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_context_padding_preserves_assistant_suffix_and_hits_exact_token_target():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { padPromptIds } = require(process.argv[1]);
const padded = padPromptIds([10, 11, 99], [99], 32, 8);
let invalid = null;
try { padPromptIds([10, 11, 99], [98], 32, 8); } catch (error) { invalid = error.message; }
process.stdout.write(JSON.stringify({ padded, invalid }));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(result.stdout)
    assert contract["padded"] == [10, 11, 32, 32, 32, 32, 32, 99]
    assert "does not end" in contract["invalid"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_structured_fixed_compute_padding_keeps_natural_decision_boundary():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { padPromptIdsTrailing } = require(process.argv[1]);
const padded = padPromptIdsTrailing([10, 11, 99], 32, 8);
process.stdout.write(JSON.stringify({ padded }));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["padded"] == [10, 11, 99, 32, 32, 32, 32, 32]
    app = WEB_APP.read_text()
    assert '"trailing_compute"' in app
    assert "dispatchSelect(out.hidden, prompt.decisionInputTokens, query)" in app
    assert "prompt.decisionInputTokens" in app
    assert "decision_feature_index: prompt.decisionFeatureIndex" in app


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_retrieval_selector_url_branch_uses_sidecar_and_preserves_mobile_guard_precedence():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "?selector=retrieval" } };
const { dispatchSelect, retrievalEmbedding } = require(process.argv[1]);
const query = "Use the browser to email Dana the quarterly report. instruction: send an email";
const sidecar = {
  route_head: {
    weight: [[0], [0], [0], [0], [0]], bias: [0, 0, 0, 0, 0],
    routes: ["text", "browser", "computer_use", "python", "filesystem"], stop_index: 0,
  },
  dense_selector: {
    q_proj_weight: [[1]], q_proj_bias: [0], proj: 1,
    tool_matrix: [[1], [0], [0]],
    tool_names: ["browser_navigate", "email_send", "mobile_click"],
    normalize_query: true,
  },
  retrieval_selector: {
    algorithm: "char_ngram_crc32_v1", dim: 256,
    tool_matrix: [Array.from(retrievalEmbedding("browser navigate", 256)),
      Array.from(retrievalEmbedding("send an email", 256)),
      Array.from(retrievalEmbedding("tap the phone screen", 256))],
    tool_names: ["browser_navigate", "email_send", "mobile_click"],
    tool_routes: ["browser", "email", "computer_use"], normalize_query: true,
    compact_instruction_marker: " instruction:",
  },
};
const hidden = { data: new Float32Array([1]), dims: [1, 1, 1] };
const selected = dispatchSelect(hidden, 1, query, sidecar);
const guarded = dispatchSelect(
  hidden, 1, "On the Android phone screen, tap Compose at x=120 y=220.", sidecar
);
process.stdout.write(JSON.stringify({ selected, guarded }));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["selected"] == {
        "name": "email_send",
        "route": "email",
        "conf": pytest.approx(1.0),
        "isStop": False,
        "selection_policy": "retrieval_selector",
    }
    assert payload["guarded"] == {
        "name": "mobile_click",
        "route": "computer_use",
        "conf": 1,
        "isStop": False,
        "selection_policy": "mobile_lexical_guard",
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_retrieval_then_dense_scores_only_retrieved_sidecar_candidates():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "?selector=retrieval_then_dense" } };
global.META = { d_model: 1 };
const { dispatchSelect, retrievalCandidatesFromSidecar } = require(process.argv[1]);
const names = Array.from({ length: 12 }, (_, index) => `tool_${index}`);
const sidecar = {
  route_head: {
    weight: [[0], [1], [0], [0], [0]], bias: [0, 0, 0, 0, 0],
    routes: ["text", "browser", "computer_use", "python", "filesystem"], stop_index: 0,
  },
  dense_selector: {
    q_proj_weight: [[1]], q_proj_bias: [0], proj: 1,
    tool_matrix: names.map((_, index) => [index === 0 ? 2 : index === 10 ? 1 : 0]),
    tool_names: names,
    normalize_query: true,
  },
  retrieval_selector: {
    algorithm: "char_ngram_crc32_v1", dim: 1,
    tool_matrix: names.map((_, index) => [index === 0 ? 0 : 1 - index / 20]),
    tool_names: names,
    tool_routes: names.map(() => "browser"),
    compact_instruction_marker: " instruction:",
  },
};
const hidden = { data: new Float32Array([1]), dims: [1, 1, 1] };
const candidates = retrievalCandidatesFromSidecar("send an email", sidecar.retrieval_selector, 10);
const selected = dispatchSelect(hidden, 1, "send an email", sidecar);
process.stdout.write(JSON.stringify({ candidates, selected }));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert len(payload["candidates"]) == 10
    assert "tool_0" not in [row["name"] for row in payload["candidates"]]
    assert "tool_11" not in [row["name"] for row in payload["candidates"]]
    assert payload["selected"] == {
        "name": "tool_10",
        "route": "browser",
        "conf": pytest.approx(1.0),
        "isStop": False,
        "selection_policy": "retrieval_then_dense",
        "candidate_count": 10,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_ar_policy_contract_keeps_raw_and_trie_decoding_distinct():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const {
  ACTION_POLICIES, argmaxAllowed, buildCandidateTrie, greedyToken, groundedActionCandidates
} = require(process.argv[1]);
const trie = buildCandidateTrie([
  { token_ids: [2, 4], action: { tool: "a", args: {} } },
  { token_ids: [2, 5], action: { tool: "b", args: {} } },
]);
const logits = new Float32Array([0, 0, 3, 9, 4, 7]);
const second = trie.children.get(2);
const noTool = groundedActionCandidates(
  "Explain the sky.",
  { tools: [], markers: {} },
  { eosId: 0, encode: () => { throw new Error("no tool should be encoded"); } }
)[0];
process.stdout.write(JSON.stringify({
  policies: ACTION_POLICIES,
  raw: greedyToken(logits),
  rootConstrained: argmaxAllowed(logits, trie.children.keys()),
  nextConstrained: argmaxAllowed(logits, second.children.keys()),
  terminals: [
    second.children.get(4).terminal.action.tool,
    second.children.get(5).terminal.action.tool,
  ],
  noTool,
}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(result.stdout)
    assert contract["policies"] == {
        "STRUCTURED": "structured_one_forward",
        "RAW_AR": "raw_autoregressive_json",
        "CONSTRAINED_AR": "grounded_candidate_trie_autoregressive",
    }
    assert contract["raw"] == 3  # unrestricted greedy maximum
    assert contract["rootConstrained"] == 2  # token 3 is masked out by the trie
    assert contract["nextConstrained"] == 5
    assert contract["terminals"] == ["a", "b"]
    assert contract["noTool"] == {
        "action": {"abstain": True},
        "completion": "",
        "token_ids": [0],
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_full_catalog_prompt_matches_training_contract_exactly():
    from openlocalagent.agent.toolset import STANDARD_TOOLS
    from openlocalagent.data.prompt_contract import render_agent_decode_prompt
    from openlocalagent.data.schema import Message, Role
    from openlocalagent.model import tokenizer as tk

    query = "Open the settings page."
    meta = {
        "markers": {
            "user": {"text": tk.USER},
            "assistant": {"text": tk.ASSISTANT},
            "tool": {"text": tk.TOOL},
            "tool_call_open": {"text": tk.TOOL_CALL_OPEN},
            "tool_call_close": {"text": tk.TOOL_CALL_CLOSE},
            "tool_response_open": {"text": tk.TOOL_RESPONSE_OPEN},
            "tool_response_close": {"text": tk.TOOL_RESPONSE_CLOSE},
        },
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.parameters,
            }
            for tool in STANDARD_TOOLS
        ],
    }
    expected = render_agent_decode_prompt(
        [Message(role=Role.user, content=query)],
        STANDARD_TOOLS,
    )
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { renderFullCatalogContextText } = require(process.argv[1]);
const meta = JSON.parse(process.argv[2]);
const query = process.argv[3];
let reserved = null;
try {
  renderFullCatalogContextText("escape <|assistant|>", [], meta);
} catch (error) {
  reserved = error.message;
}
process.stdout.write(JSON.stringify({
  actual: renderFullCatalogContextText(query, [], meta),
  reserved,
}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP), json.dumps(meta), query],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["actual"] == expected
    assert "reserved prompt marker" in payload["reserved"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_cached_logits_support_deterministic_masked_and_seeded_sampling():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { materializeCachedLogits, selectTokenFromLogits } = require(process.argv[1]);
const values = materializeCachedLogits({
  type: "float16",
  dims: [1, 4],
  data: Uint16Array.from([0x0000, 0x3c00, 0x4000, 0xbc00]),
  location: "cpu",
}, {
  cacheDtype: "float16",
  metadata: { vocab_size: 4 },
}, "fixture.logits");
const sampleOptions = { temperature: 0.8, topK: 3, seed: "repeatable" };
process.stdout.write(JSON.stringify({
  values: Array.from(values),
  greedy: selectTokenFromLogits(values),
  masked: selectTokenFromLogits(values, {}, [0, 1, 3]),
  sampleA: selectTokenFromLogits(values, sampleOptions),
  sampleB: selectTokenFromLogits(values, sampleOptions),
}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["values"] == [0, 1, 2, -1]
    assert payload["greedy"] == 2
    assert payload["masked"] == 1
    assert payload["sampleA"] == payload["sampleB"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_cached_runner_prefills_once_then_rebinds_present_tensors_for_decode():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
global.ort = {
  Tensor: class {
    constructor(type, data, dims) {
      this.type = type; this.data = data; this.dims = dims; this.location = "cpu";
      this.disposed = false;
    }
    dispose() { this.disposed = true; }
  },
};
const {
  createCachedAutoregressiveRunner, selectTokenFromLogits
} = require(process.argv[1]);
const tensor = (type, data, dims) => new ort.Tensor(type, data, dims);
const prefillKey = tensor("float32", new Float32Array(12), [1, 1, 3, 4]);
const prefillValue = tensor("float32", new Float32Array(12), [1, 1, 3, 4]);
const decodeKey = tensor("float32", new Float32Array(16), [1, 1, 4, 4]);
const decodeValue = tensor("float32", new Float32Array(16), [1, 1, 4, 4]);
let prefillCalls = 0;
let decodeCalls = 0;
let directBindings = false;
const bundle = {
  cacheDtype: "float32",
  presentNames: ["present_0_key", "present_0_value"],
  contract: {
    cache_slots: [{
      kind: "attn",
      past_inputs: ["past_0_key", "past_0_value"],
      present_outputs: ["present_0_key", "present_0_value"],
    }],
  },
  metadata: {
    vocab_size: 4,
    model: { config: { n_kv_heads: 1, d_model: 8, n_heads: 2, conv_kernel: 3 } },
  },
  prefillSession: {
    async run(feeds) {
      prefillCalls += 1;
      if (feeds.input_ids.dims.join(",") !== "1,3") throw new Error("bad prefill shape");
      return {
        next_token: tensor("int64", BigInt64Array.of(2n), [1]),
        logits: tensor("float32", Float32Array.from([0, 1, 5, 2]), [1, 4]),
        present_0_key: prefillKey,
        present_0_value: prefillValue,
      };
    },
  },
  decodeSession: {
    async run(feeds) {
      decodeCalls += 1;
      directBindings =
        feeds.past_0_key === prefillKey && feeds.past_0_value === prefillValue;
      return {
        next_token: tensor("int64", BigInt64Array.of(3n), [1]),
        logits: tensor("float32", Float32Array.from([0, 1, 2, 6]), [1, 4]),
        present_0_key: decodeKey,
        present_0_value: decodeValue,
      };
    },
  },
};
(async () => {
  const runner = createCachedAutoregressiveRunner(bundle, [1, 2, 3]);
  const first = await runner.prefill();
  const selected = selectTokenFromLogits(first.logits);
  const second = await runner.decode(selected);
  runner.dispose();
  process.stdout.write(JSON.stringify({
    prefillCalls, decodeCalls, directBindings, selected,
    next: selectTokenFromLogits(second.logits),
    prefillCachesDisposed: prefillKey.disposed && prefillValue.disposed,
    finalCachesDisposed: decodeKey.disposed && decodeValue.disposed,
  }));
})().catch((error) => { console.error(error); process.exit(1); });
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "prefillCalls": 1,
        "decodeCalls": 1,
        "directBindings": True,
        "selected": 2,
        "next": 3,
        "prefillCachesDisposed": True,
        "finalCachesDisposed": True,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_ar_parser_separates_parse_and_schema_failures():
    meta = {
        "markers": {
            "tool_call_open": {"text": "<tool_call>"},
            "tool_call_close": {"text": "</tool_call>"},
        },
        "tools": [
            {
                "name": "click",
                "schema": {
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                },
            }
        ],
    }
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { parseGeneratedAction } = require(process.argv[1]);
const meta = JSON.parse(process.argv[2]);
const values = [
  parseGeneratedAction(
    '<tool_call>{"arguments":{"target":"Confirm"},"name":"click"}</tool_call>',
    "eos", meta
  ),
  parseGeneratedAction(
    '<tool_call>{"arguments":{},"name":"click"}</tool_call>',
    "eos", meta
  ),
  parseGeneratedAction('<tool_call>{"arguments":', "max_tokens", meta),
  parseGeneratedAction("The sky is blue.", "eos", meta),
  parseGeneratedAction("", "candidate_terminal", meta),
];
process.stdout.write(JSON.stringify(values));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP), json.dumps(meta)],
        check=True,
        capture_output=True,
        text=True,
    )
    valid, invalid_schema, invalid_json, direct, trie_abstain = json.loads(result.stdout)
    assert valid["parse_failure"] is False
    assert valid["schema_valid"] is True
    assert invalid_schema["parse_failure"] is False
    assert invalid_schema["validation_failure"] is True
    assert invalid_json["parse_failure"] is True
    assert invalid_json["validation_failure"] is False
    assert direct["action"] == {"abstain": True}
    assert direct["parse_kind"] == "direct_text"
    assert trie_abstain["action"] == {"abstain": True}
    assert trie_abstain["parse_failure"] is False
    assert trie_abstain["schema_valid"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_dom_candidate_safety_adapter_prefers_task_matching_observation():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { browserCandidateTargetId, groundFromSchema } = require(process.argv[1]);
const prompt = [
  "Find the results of the most recent NFL games.",
  "Browser DOM candidates: operation=CLICK target_id=213 text=YAHOO PLUS |",
  "target_id=2932 text=Final | target_id=170 id=root_3 text=NFL"
].join(" ");
const schema = {
  properties: { target_id: { type: "string" } },
  required: ["target_id"]
};
process.stdout.write(JSON.stringify({
  target: browserCandidateTargetId(prompt),
  args: groundFromSchema(prompt, schema, { target_id: "2932 text=Final" })
}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"target": "170", "args": {"target_id": "170"}}


def test_browser_benchmark_exposes_all_policy_modes_and_cached_decode_metadata():
    html = WEB_BENCHMARK_HTML.read_text()
    app = WEB_APP.read_text()
    for policy in (
        "structured_one_forward",
        "raw_autoregressive_json",
        "grounded_candidate_trie_autoregressive",
    ):
        assert f'value="{policy}"' in html
    assert "LOGITS_MODEL_URL = META.model_file" in app
    assert "await runner.prefill()" in app
    assert "await runner.decode(generated.at(-1))" in app
    assert "selectTokenFromLogits(" in app
    assert "const CACHED_DECODE_STRATEGY = \"prefill_then_kv_cached_decode\"" in app
    assert app.count("decode_strategy: CACHED_DECODE_STRATEGY") == 2
    assert "decode_cache: true" in app
    assert "next_token_role: \"compatibility_argmax_cross_check\"" in app
    assert "prompt_contract: prompt.promptContract" in app
    assert "tokenTimes[0] - tokenizedAt" in app
    assert "tokenTimes[0] - started" not in app
    assert 'id="benchmark-context-tokens"' in html
    assert 'value="1536"' in html
    assert 'id="benchmark-warmups"' in html
    assert 'value="3"' in html
    assert 'id="benchmark-repetitions"' in html
    assert 'value="30"' in html
    assert 'id="benchmark-seed"' in html
    assert 'value="slmw2026-v2-trailing"' in html
    benchmark = WEB_BENCHMARK.read_text()
    assert 'latency_clock: "harness_ttfa_ms"' in benchmark
    assert "harness_ttfa_ms: harnessTtfaMs" in benchmark
    assert "artifact_hash_contract" in benchmark
    assert "model_byte_evidence: policyByteEvidence" in benchmark
    assert "per_node_placement: \"unknown\"" in benchmark
    assert 'benchmark_version: "rtab-0.4"' in benchmark
    assert "fixed_compute_tokens_natural_decision_feature" in benchmark
    assert "fixed_final_tokenizer_tokens_pre_assistant_stress" in benchmark
    assert "predicted_route: action.route ?? null" in benchmark
    assert "route_confidence: action.conf ?? null" in benchmark
    assert "predicted_action: predictedAction" in benchmark
    assert "expected_action: expectedAction" in benchmark
    assert "independent_schema: independentSchema" in benchmark
    assert "parse_evidence: parseEvidence" in benchmark
    assert "action_timeout_ms: BENCHMARK_ACTION_TIMEOUT_MS" in benchmark
    assert "no_subsequent_policy_call_started: true" in benchmark


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_v04_action_evidence_is_normalized_and_independently_scored():
    script = """
global.window = { __localAgentSkipInit: true };
global.META = {
  tools: [{
    name: "click",
    schema: {
      type: "object",
      properties: { target: { type: "string" } },
      required: ["target"],
      additionalProperties: false,
    },
  }],
};
const {
  normalizeBenchmarkAction, validateBenchmarkActionSchema, scoreBenchmarkAction
} = require(process.argv[1]);
const raw = {
  tool: "click", args: { target: "Confirm" }, timing: { huge: "omitted" }, route: "tool",
};
const predicted = normalizeBenchmarkAction(raw);
const expected = { tool: "click", args: { target: "Confirm" } };
const valid = validateBenchmarkActionSchema(predicted);
const invalid = validateBenchmarkActionSchema({
  tool: "click", args: { target: 4, extra: true },
});
const score = scoreBenchmarkAction(predicted, { expected }, valid.valid);
process.stdout.write(JSON.stringify({ predicted, expected, valid, invalid, score }));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_BENCHMARK)],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(result.stdout)
    assert evidence["predicted"] == {
        "tool": "click",
        "args": {"target": "Confirm"},
    }
    assert evidence["expected"] == evidence["predicted"]
    assert evidence["valid"]["valid"] is True
    assert evidence["valid"]["errors"] == []
    assert evidence["valid"]["schema_tool"] == "click"
    assert evidence["valid"]["tool_schema"]["required"] == ["target"]
    assert evidence["invalid"]["valid"] is False
    assert evidence["invalid"]["errors"] == [
        "$.args.target does not have JSON Schema type string.",
        "$.args.extra is not declared by the tool schema.",
    ]
    assert evidence["score"] == {
        "exact_tool": True,
        "exact_args": True,
        "exact_action": True,
        "success": True,
        "schema_valid": True,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_v04_watchdog_is_finite_and_timeout_is_fail_stop():
    script = """
global.window = { __localAgentSkipInit: true };
const {
  BENCHMARK_ACTION_TIMEOUT_MS, benchmarkWithWatchdog, isBenchmarkActionTimeout
} = require(process.argv[1]);
(async () => {
  const completed = await benchmarkWithWatchdog(() => Promise.resolve("ok"), 20);
  let timeout = null;
  let subsequentStarted = false;
  try {
    await benchmarkWithWatchdog(() => new Promise(() => {}), 5);
    subsequentStarted = true;
  } catch (error) {
    timeout = {
      name: error.name,
      code: error.code,
      timeout_ms: error.timeout_ms,
      recognized: isBenchmarkActionTimeout(error),
    };
  }
  process.stdout.write(JSON.stringify({
    configured: BENCHMARK_ACTION_TIMEOUT_MS, completed, timeout, subsequentStarted
  }));
})().catch((error) => { console.error(error); process.exit(1); });
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_BENCHMARK)],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(result.stdout)
    assert evidence == {
        "configured": 10_000,
        "completed": "ok",
        "timeout": {
            "name": "BenchmarkActionTimeoutError",
            "code": "benchmark_action_timeout",
            "timeout_ms": 5,
            "recognized": True,
        },
        "subsequentStarted": False,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_model_byte_verification_is_fail_closed():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
global.crypto = require("node:crypto").webcrypto;
const { verifyModelBytesAgainstManifest } = require(process.argv[1]);
(async () => {
  const bytes = new TextEncoder().encode("checked model bytes");
  const good = {
    artifacts: {
      "action_model.fp16.onnx": {
        file: "action_model.fp16.onnx",
        bytes: bytes.byteLength,
        sha256: "053d063c235a71096d6ca139e110e34cb77c6a336ed13633581ab27abcbfc533",
      },
    },
  };
  const verified = await verifyModelBytesAgainstManifest(
    "action_model.fp16.onnx", bytes, good, true
  );
  let mismatch = null;
  try {
    await verifyModelBytesAgainstManifest(
      "action_model.fp16.onnx",
      bytes,
      { artifacts: {
        "action_model.fp16.onnx": {
          file: "action_model.fp16.onnx",
          bytes: bytes.byteLength,
          sha256: "0".repeat(64),
        },
      } },
      true
    );
  } catch (error) {
    mismatch = error.message;
  }
  process.stdout.write(JSON.stringify({ verified, mismatch }));
})().catch((error) => { console.error(error); process.exit(1); });
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(result.stdout)
    assert evidence["verified"]["manifest_verified"] is True
    assert evidence["verified"]["session_source"] == "in_memory_verified_bytes"
    assert evidence["mismatch"].startswith("SHA-256 mismatch")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_runtime_asset_byte_verification_checks_hash_and_size():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
global.crypto = require("node:crypto").webcrypto;
const { verifyArtifactBytesAgainstManifest } = require(process.argv[1]);
(async () => {
  const bytes = new TextEncoder().encode("checked heads json");
  const manifest = { artifacts: {
    "heads.json": {
      file: "heads.json",
      bytes: bytes.byteLength,
      sha256: "4c01b9787684a4b4ed843cc0739607b7b0bf85e84a569e224a887a5b8e0da7bb",
    },
  } };
  const verified = await verifyArtifactBytesAgainstManifest(
    "heads.json", bytes, manifest, true
  );
  let sizeMismatch = null;
  manifest.artifacts["heads.json"].bytes += 1;
  try {
    await verifyArtifactBytesAgainstManifest("heads.json", bytes, manifest, true);
  } catch (error) {
    sizeMismatch = error.message;
  }
  process.stdout.write(JSON.stringify({ verified, sizeMismatch }));
})().catch((error) => { console.error(error); process.exit(1); });
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(result.stdout)
    assert evidence["verified"]["manifest_verified"] is True
    assert evidence["verified"]["verification_scope"] == "exact_fetched_response_body_bytes"
    assert evidence["sizeMismatch"].startswith("Byte-length mismatch")


@pytest.mark.parametrize(
    ("suite_name", "suite_path"),
    [
        ("local-realtime-actions", ROOT / "demos" / "webgpu" / "benchmark-cases.json"),
        (
            "local-browser-tasks",
            ROOT / "demos" / "webgpu" / "browser-task-cases.json",
        ),
    ],
)
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_suite_pins_match_decontamination_config_and_reject_tampering(
    suite_name, suite_path
):
    import yaml

    config = yaml.safe_load(PRETRAIN_PAPER_CONFIG.read_text())
    expected = next(
        suite
        for suite in config["evaluation_decontamination"]["required_suites"]
        if suite["name"] == suite_name
    )
    runner = (
        WEB_BENCHMARK
        if suite_name == "local-realtime-actions"
        else ROOT / "demos" / "webgpu" / "browser-tasks.js"
    ).read_text()
    assert f"bytes: {expected['bytes']}" in runner
    assert f'sha256: "{expected["sha256"]}"' in runner
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
global.crypto = require("node:crypto").webcrypto;
const fs = require("node:fs");
const { verifyPinnedArtifactBytes } = require(process.argv[1]);
(async () => {
  const path = process.argv[2];
  const expected = JSON.parse(process.argv[3]);
  const bytes = new Uint8Array(fs.readFileSync(path));
  const verified = await verifyPinnedArtifactBytes(path, bytes, expected);
  const tampered = bytes.slice();
  tampered[tampered.length - 1] ^= 1;
  let tamperError = null;
  try { await verifyPinnedArtifactBytes(path, tampered, expected); }
  catch (error) { tamperError = error.message; }
  process.stdout.write(JSON.stringify({ verified, tamperError }));
})().catch((error) => { console.error(error); process.exit(1); });
"""
    result = subprocess.run(
        [
            shutil.which("node"),
            "-e",
            script,
            str(WEB_APP),
            str(suite_path),
            json.dumps(expected),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(result.stdout)
    assert evidence["verified"]["bytes"] == expected["bytes"] == suite_path.stat().st_size
    assert evidence["verified"]["sha256"] == expected["sha256"]
    assert evidence["verified"]["identity_verified"] is True
    assert evidence["tamperError"].startswith("Pinned SHA-256 mismatch")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_benchmark_manifest_contract_rejects_missing_or_failed_graph_evidence():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { validateBenchmarkBundleContract } = require(process.argv[1]);
const meta = {
  model_file: "model.fp16.onnx",
  action_model_file: "action_model.fp16.onnx",
};
const result = (outputs, bytes, sha256) => ({
  artifact: { bytes, sha256 },
  passed: true,
  expected_outputs: outputs,
  max_abs_diff_by_output: Object.fromEntries(outputs.map((name) => [name, 0.01])),
  threshold_max_abs_diff_by_output: Object.fromEntries(outputs.map((name) => [name, 0.05])),
});
const manifest = {
  schema_version: 3,
  artifacts: {
    "model.fp16.onnx": {
      file: "model.fp16.onnx", bytes: 10, sha256: "a".repeat(64),
    },
    "action_model.fp16.onnx": {
      file: "action_model.fp16.onnx", bytes: 8, sha256: "b".repeat(64),
    },
  },
  parity_gate: {
    hard_gate: true,
    passed: true,
    results: {
      "model.fp16.onnx": result(["logits", "hidden"], 10, "a".repeat(64)),
      "action_model.fp16.onnx": result(["hidden"], 8, "b".repeat(64)),
    },
  },
};
const valid = validateBenchmarkBundleContract(meta, manifest);
const errors = [];
for (const mutate of [
  (copy) => { delete copy.artifacts["action_model.fp16.onnx"]; },
  (copy) => {
    copy.parity_gate.results["action_model.fp16.onnx"]
      .max_abs_diff_by_output.hidden = 0.5;
  },
]) {
  const copy = JSON.parse(JSON.stringify(manifest));
  mutate(copy);
  try { validateBenchmarkBundleContract(meta, copy); }
  catch (error) { errors.push(error.message); }
}
process.stdout.write(JSON.stringify({ valid, errors }));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(result.stdout)
    assert contract["valid"] is True
    assert len(contract["errors"]) == 2
    assert "does not bind required model artifact" in contract["errors"][0]
    assert "parity evidence is invalid" in contract["errors"][1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_production_cached_bundle_fails_closed_on_abi_lineage_and_catalog_drift():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { validateProductionCachedBundle } = require(process.argv[1]);
const sha = (character) => character.repeat(64);
const config = {
  name: "fixture", vocab_size: 384, d_model: 8, n_layers: 1, n_loops: 1,
  n_heads: 2, n_kv_heads: 1, conv_kernel: 3, layer_types: ["attn"],
};
const slot = {
  slot: 0, loop: 0, layer: 0, kind: "attn",
  past_inputs: ["past_0_key", "past_0_value"],
  present_outputs: ["present_0_key", "present_0_value"],
  shape: ["batch", 1, "cache_sequence", 4],
  update: "append_one_token_along_axis_2",
  dtype_by_precision: { fp32: "float32" },
};
const typed = (name, shape, dtype) => ({ name, shape, dtype });
const outputs = [
  typed("next_token", ["batch"], "int64"),
  typed("logits", ["batch", "vocab_size"], "float32"),
  typed("present_0_key", slot.shape, "float32"),
  typed("present_0_value", slot.shape, "float32"),
];
const graphContract = {
  cache_slots: [slot],
  cache_update_strategy:
    "attention K/V append one token; short-conv state replaces its fixed-width tail",
  prefill_projection:
    "only the final normalized prompt feature is projected to vocabulary logits",
  decode_token_axis_fixed_one: true,
  decode_position: {
    caller_position_input: false,
    derived_from: "past_0_key",
    rule: "RoPE position = first attention past-key axis-2 length",
  },
  graphs: {
    fp32: {
      cache_dtype: "float32",
      prefill: {
        file: "prefill.fp32.onnx",
        input_names: ["input_ids"],
        inputs: [typed("input_ids", ["batch", "prompt_sequence"], "int64")],
        output_names: outputs.map((entry) => entry.name),
        outputs,
      },
      decode: {
        file: "decode.fp32.onnx",
        input_names: ["input_ids", "past_0_key", "past_0_value"],
        inputs: [
          typed("input_ids", ["batch", 1], "int64"),
          typed("past_0_key", slot.shape, "float32"),
          typed("past_0_value", slot.shape, "float32"),
        ],
        output_names: outputs.map((entry) => entry.name),
        outputs,
      },
    },
  },
  logits: {
    name: "logits",
    description: "unnormalized LM scores for the final input token only",
    shape: ["batch", 384],
    dtype_by_precision: { fp32: "float32" },
  },
  next_token: {
    name: "next_token", dtype: "int64", shape: ["batch"],
    decode: "compatibility argmax over the exported final-token logits",
  },
};
const markers = {
  user: { text: "<|user|>", ids: [10] },
  assistant: { text: "<|assistant|>", ids: [11] },
  tool: { text: "<|tool|>", ids: [12] },
  tool_call_open: { text: "<tool_call>", ids: [13] },
  tool_call_close: { text: "</tool_call>", ids: [14] },
  tool_response_open: { text: "<tool_response>", ids: [15] },
  tool_response_close: { text: "</tool_response>", ids: [16] },
};
const tools = [{
  name: "click", description: "Click a target.",
  args: ["target"],
  schema: {
    type: "object",
    properties: { target: { type: "string" } },
    required: ["target"],
    additionalProperties: false,
  },
}];
const tokenizer = {
  kind: "bpe", encoding: "bytelevel-bpe", vocab_size: 384,
  eos_id: 0, pad_id: 0, file: "tokenizer.json", verified: true,
  sha256: sha("d"),
};
const lineage = {
  version: 1, stage: "rl", config_sha256: sha("a"), data_sha256: sha("b"),
  model_config_sha256: sha("c"), tokenizer_sha256: tokenizer.sha256,
  parent_checkpoint_sha256: sha("e"),
  git: {
    commit: "f".repeat(40), repository_sha256: sha("1"),
    dirty: false, worktree_sha256: sha("2"),
  },
};
const metadata = {
  artifact_type: "localagent_cached_autoregressive_onnx",
  schema_version: 1,
  default_precision: "fp32",
  vocab_size: 384, d_model: 8, max_seq_len: 2048,
  encoding: "bytelevel-bpe", eos_id: 0, pad_id: 0,
  markers, tools, tokenizer, graph_contract: graphContract,
  model: {
    config,
    config_canonical_sha256: lineage.model_config_sha256,
    config_file: "model-config.yaml",
    parameters: 1000,
  },
  checkpoint: {
    stage: "rl", step: 7, sha256: sha("3"),
    conversation_prompt_contract: "openai_full_catalog_v1",
    lineage,
    lineage_export: {
      file: "training-lineage.json",
      kind: "localagent_training_lineage_export",
      schema_version: 1,
    },
  },
};
const artifact = (file) => ({
  file, bytes: 100, sha256: file.startsWith("prefill") ? sha("4") : sha("5"),
  precision: "fp32",
});
const provenance = {
  schema_version: 1,
  artifact_type: "trained_checkpoint_cached_decode_onnx",
  trained: true,
  weights: {
    source: "strict_lineage_validated_lm_checkpoint",
    checkpoint_sha256: metadata.checkpoint.sha256,
    checkpoint_stage: "rl",
    checkpoint_step: 7,
  },
  checkpoint_lineage: lineage,
  graph_contract: graphContract,
  tokenizer,
  auxiliary_heads: {
    available: false, exported: false, invalidated: [], validated: true,
  },
  artifacts: {
    "prefill.fp32.onnx": artifact("prefill.fp32.onnx"),
    "decode.fp32.onnx": artifact("decode.fp32.onnx"),
  },
  parity: {
    hard_gate: true,
    results: {
      fp32: {
        hard_gate: true, passed: true, greedy_next_token_exact: true,
        cache_dtype: "float32", logits_atol: 0.001, max_logits_abs_diff: 0,
        final_token_logits_shape: ["batch", 384],
        artifacts: {
          prefill: { bytes: 100, sha256: sha("4") },
          decode: { bytes: 100, sha256: sha("5") },
        },
        reference_independence: {
          onnx_logits_vs_pytorch_cached_path: true,
          pytorch_cached_vs_fresh_full_context_logits: true,
        },
      },
    },
  },
};
const actionMetadata = {
  encoding: metadata.encoding, vocab_size: 384, d_model: 8, max_seq_len: 2048,
  eos_id: 0, pad_id: 0,
  markers, tools,
};
const lineageExport = {
  kind: "localagent_training_lineage_export",
  schema_version: 1,
  stage: "rl",
  checkpoint_sha256: metadata.checkpoint.sha256,
  lineage,
  training_artifact_sha256: [sha("6")],
  conversation_prompt_contract: "openai_full_catalog_v1",
};
const clone = (value) => JSON.parse(JSON.stringify(value));
const rejection = (callback) => {
  try { callback(); return null; } catch (error) { return error.message; }
};
const accepted = validateProductionCachedBundle(
  metadata, provenance, actionMetadata, { sha256: tokenizer.sha256 }, lineageExport
);
const staleName = clone(metadata);
staleName.graph_contract.graphs.fp32.decode.output_names[1] = "scores";
const staleDtype = clone(metadata);
staleDtype.graph_contract.graphs.fp32.prefill.outputs[1].dtype = "float16";
const badLineage = clone(metadata);
badLineage.checkpoint.lineage.parent_checkpoint_sha256 = "A".repeat(64);
const badCatalog = clone(actionMetadata);
badCatalog.tools[0].name = "stale_click";
const badSidecar = clone(lineageExport);
badSidecar.checkpoint_sha256 = sha("7");
process.stdout.write(JSON.stringify({
  precision: accepted.precision,
  staleName: rejection(() => validateProductionCachedBundle(
    staleName, provenance, actionMetadata, { sha256: tokenizer.sha256 }, lineageExport
  )),
  staleDtype: rejection(() => validateProductionCachedBundle(
    staleDtype, provenance, actionMetadata, { sha256: tokenizer.sha256 }, lineageExport
  )),
  badLineage: rejection(() => validateProductionCachedBundle(
    badLineage, provenance, actionMetadata, { sha256: tokenizer.sha256 }, lineageExport
  )),
  badCatalog: rejection(() => validateProductionCachedBundle(
    metadata, provenance, badCatalog, { sha256: tokenizer.sha256 }, lineageExport
  )),
  badSidecar: rejection(() => validateProductionCachedBundle(
    metadata, provenance, actionMetadata, { sha256: tokenizer.sha256 }, badSidecar
  )),
}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["precision"] == "fp32"
    assert "names do not match" in payload["staleName"]
    assert "stale dtype or shape" in payload["staleDtype"]
    assert "canonical final-RL lineage" in payload["badLineage"]
    assert "tokenizer/catalog metadata disagrees" in payload["badCatalog"]
    assert "training-lineage sidecar" in payload["badSidecar"]


def test_benchmark_bundle_contract_requires_hidden_only_action_export():
    app = WEB_APP.read_text()
    benchmark = WEB_BENCHMARK.read_text()
    deploy = (ROOT / "demos" / "webgpu" / "DEPLOY.md").read_text()
    assert "Benchmark-grade runs require bundle-manifest.json" in app
    assert "re-export with action_only=True" in app
    assert 'BENCHMARK_GRADE ? ["hidden"] : null' in app
    assert "artifact.bytes.slice()" in app
    assert "action_only=True" in deploy
    assert "action_model.fp16.onnx" in deploy
    assert "bundle-manifest.json" in deploy
    for artifact in ("heads.json", "meta.json", "dispatch_heads.json"):
        assert f'fetchJsonArtifact("{artifact}", BENCHMARK_GRADE)' in app
    assert "const tokenizerDocument = await fetchJsonArtifact(" in app
    assert app.index("await fetchBundleArtifactBytes(artifactUrl") < app.index(
        "value = JSON.parse(text)"
    )
    assert "runtime_asset_evidence: runtimeAssets" in benchmark
    assert "suite_byte_evidence: suiteByteEvidence" in benchmark
    assert "bundle_manifest_byte_evidence: bundleManifestByteEvidence()" in benchmark
    assert "explicit-webgpu-no-whole-session-retry" in benchmark
    assert 'per_node_fallback_status: "unknown"' in benchmark
