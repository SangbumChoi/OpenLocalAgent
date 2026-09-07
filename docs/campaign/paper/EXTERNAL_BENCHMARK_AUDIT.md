# External benchmark provenance and adapter audit

Status: all four revision-pinned prompt-only adapters and v3 suite freezes have now passed strict
raw-chain replay. The 1,880-byte private aggregate manifest binds BFCL, BrowserGym, Mind2Web, and
WebLINX outputs totaling 34,068,955 bytes; its file SHA-256 is
`3466e9242ccc3aadf487fd2c7fa1dc7bdc9ed14a37007955f75cfece0c040ad1` and its canonical self-hash
is `826f53f5699f3c4b8f311a9fe70561f5b7d9aa99ce42ce250548e46aa644010b`. This is
corpus-decontamination provenance only, not benchmark-score or chronological-freshness evidence.
Protected payloads and prompts remain private and ignored rather than vendored. Paper-all
preparation replayed the four protected inputs, bound them with three local exclusions, and produced
a 504,010-document packed corpus. Independent freeze verification passed. Its bounded audit of
19,334 supplied normalized denylist prompts made 8,633,077 candidate checks and removed 15
documents; it is explicitly non-exhaustive and is not native benchmark evaluation. WebLINX still
requires residual privacy/legal review and external receipt archival.

The machine-readable revision/split/use plan is
[`configs/data/evaluation-benchmarks-paper.yaml`](../../configs/data/evaluation-benchmarks-paper.yaml).

## Decision

The four target suites have two separate roles:

1. Before tokenizer fitting, private prompt-only exports are corpus-exclusion inputs. They contain
   no current-step gold action, expected call, positive/negative label, score, or outcome.
2. After checkpoints and all training text are frozen, evaluation uses each benchmark's native
   semantics or an explicitly named derived contract. Old public revisions can establish local
   heldout isolation, but not chronological freshness.

One generic exact-call schema is not faithful to all four suites:

| Suite | Prompt-only exclusion | Faithful paper evaluation |
|---|---|---|
| BFCL v4 | User messages plus model-visible function schemas | Official BFCL checker; a v1 conversion is only a derived unique-ground-truth subset |
| Mind2Web | Private task and exactly formatted step inputs | Multi-positive element and operation/task metrics with a frozen DOM ranker |
| WebLINX | Private model-visible dialogue/action-history/observation inputs | Intent, element-IoU, text-F1, and micro metrics through the pinned evaluator |
| BrowserGym/MiniWoB | Reset-returned dynamic goals for frozen task/seed pairs | Closed-loop episode reward; no single gold trajectory |

The current `localagent_fresh_external_action_eval_contract` v1 may support a clearly labeled
BFCL-derived exact-action slice. It must not be described as an official BFCL, Mind2Web, WebLINX,
or BrowserGym score. A faithful shared v2 would need multiple acceptable calls, mixed text/action
outputs, sharded private inputs, native-scoring plugin identities, ranker identity/recall, and
closed-loop reward records.

## Immutable upstream pins

### BFCL v4

- Gorilla repository:
  [`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`](https://github.com/ShishirPatil/gorilla/commit/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8)
- License:
  [Apache-2.0](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/LICENSE)
- [Data contract](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data/README.md)
- [Official AST checker](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py)

Use the pinned GitHub tree rather than the Hugging Face mirror audited on this date, whose visible
files were still v3-named. Top-level v4 input files are JSONL records with `id`, nested
`question` messages, and `function` specifications; gold files live separately under
`data/possible_answer/`.

For an ordinary stateless comparison, the candidate categories are `simple_python`, `multiple`,
`parallel`, and `parallel_multiple` (1,000 upstream rows). Irrelevance has no call, while live,
multi-turn, web-search, and memory categories require semantics absent from v1. The official
checker permits alternative argument values, optional values, BFCL-specific normalization/types,
and order-insensitive parallel calls. Eligibility for a 200–256-case unique-ground-truth derived
slice remains uncomputed.

Audited uncompressed Git blob sizes:

- complete BFCL data tree: 12,297,003 bytes;
- top-level v4 inputs: 9,580,472 bytes;
- possible-answer files: 978,870 bytes; and
- four candidate stateless input/gold pairs: 1,355,321 bytes.

The four prompt inputs are bound separately in the machine plan: `simple_python` is 283,274 bytes
/ 400 rows / SHA-256 `82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991`;
`multiple` is 316,583 / 200 /
`aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a`; `parallel` is 171,896 /
200 / `19f51a82eff42e5d62541aa500115a056eb78f437c2ba1f10415fd7c8e5dda84`; and
`parallel_multiple` is 347,080 / 200 /
`8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579`. All four omit a final
newline. The adapter refuses production mode unless all byte, row, and hash identities match.
The exact pinned run emitted 2,677 prompt-component rows (1,294,741 bytes, SHA-256
`9d8f2a87e40a3313a1d039cd22dadbbbbd981d7df3b657dc106d79bd76fa5137`). The current-plan v3
production freezer strictly replayed the source and audit chain, removed normalized duplicates, and
retained 2,023 rows (989,817 bytes, SHA-256
`fecb44c23bd00aad0ae65ec9c0bf6457df5125a08bccffeda7d292a785d9668c`). The aggregate manifest
binds that exact final output.

### BrowserGym and MiniWoB++

- BrowserGym:
  [`9e779f087de9a65668b6974d11f9ce9816026e96`](https://github.com/ServiceNow/BrowserGym/commit/9e779f087de9a65668b6974d11f9ce9816026e96),
  version 0.14.3,
  [Apache-2.0](https://github.com/ServiceNow/BrowserGym/blob/9e779f087de9a65668b6974d11f9ce9816026e96/LICENSE)
- MiniWoB++:
  [`7fd85d71a4b60325c6585396ec4f48377d049838`](https://github.com/Farama-Foundation/miniwob-plusplus/commit/7fd85d71a4b60325c6585396ec4f48377d049838),
  [MIT](https://github.com/Farama-Foundation/miniwob-plusplus/blob/7fd85d71a4b60325c6585396ec4f48377d049838/LICENSE)
- [BrowserGym MiniWoB adapter pin](https://github.com/ServiceNow/BrowserGym/blob/9e779f087de9a65668b6974d11f9ce9816026e96/browsergym/miniwob/README.md)
- [Pinned task metadata](https://github.com/ServiceNow/BrowserGym/blob/9e779f087de9a65668b6974d11f9ce9816026e96/browsergym/experiments/src/browsergym/experiments/benchmark/metadata/miniwob.csv)

Use local MiniWoB rather than credentialed, mutable WebArena/WorkArena sites. The prompt is the
dynamic `goal` returned by environment setup, not a task metadata comment. The pinned metadata has
63 test task variants. LocalAgent excludes `click-pie`, `click-pie-nodelay`, and `terminal` as an
operational repeatability policy, leaving 60 variants in 41 similarity groups. This is our policy,
not an upstream nondeterminism annotation; the frozen receipt records them as LocalAgent policy
exclusions. No similarity group crosses the pinned train/test split. With fixed seeds
`[11, 17, 23, 29]`, the prompt-acquisition plan has 240 resets. A future scored comparison must
cluster uncertainty on 41 groups rather than treating the task/seed episodes as independent tasks.

The current repeat-task helper must not define seeds: at the audited pin its `2 ^ 32` expression
is bitwise XOR, yielding 34 rather than the apparent `2**32` range. Set seeds explicitly.

The controlled producer enforces the exact sorted 60-task by four-seed plan, creates
`browsergym/{task_name}`, calls `reset(seed=seed)` once, copies only the returned
`observation["goal"]`, takes no action, and closes every environment. Before and after all resets it
attests clean local BrowserGym and MiniWoB git checkouts plus the Playwright/Chromium installation;
it refuses source/runtime drift and publishes canonical JSONL plus a self-hashed receipt with
strict no-clobber and rollback-on-detected-error behavior. A process interruption can still leave
an incomplete pair, so consumers must require and jointly verify both files. The receipt explicitly
disclaims labels, rewards, scores, and fresh evaluation evidence.

Two isolated controlled acquisitions were then run on the attested stack. Their canonical 240-row
captures were byte-identical: 348,513 bytes with SHA-256
`128f7f6be8d5b52f745523b0bca4517fdaf8107044eee5a76366464ac10079ff`. The adopted producer
receipt is 6,538 bytes with SHA-256
`b04318c36579a05d3f61a40ea09c1f1c0bd1e004a534b2b5d18305b50e68ebea` and self-hash
`e8cece5a8acf0f5e2333e004c33b035b4b31fa7ec3e3d501c43fcbbac341611a`. Production export
reverified that receipt and emitted 240 prompt-only rows. The generic freezer rederived that export
from the raw capture/receipt pair, removed 77 normalized duplicates, and froze 163 unique normalized
prompts (26,402 bytes, SHA-256
`808b4c5206e5d6ebcd0704b88ba87b7c5c9c7e1866ccfe96ab4201228f436120`).

A BrowserGym artifact must additionally pin action set, observation mode, max steps, locale,
timezone, viewport/device scale, headless mode, timeout, OS/architecture, browser executable
identity, Playwright, and Chromium. Playwright 1.44.0 pins Chromium revision 1117 / version
125.0.6422.26. The frozen Darwin/arm64 Python 3.12.2 runtime manifest is 12,996 bytes with SHA-256
`5edf3987b09db987eabbef52324ef6d0eb87d69e7c36e94d5f88cdccddf21382`; the private receipt
also binds the exact browser executable and installation identities. The eventual primary outcome
is paired binary episode reward; steps on successful episodes and timeout rate are secondary. That
closed-loop evaluation has not run, so there is no BrowserGym episode score yet.

### Mind2Web

- Dataset:
  [`osunlp/Mind2Web@17ece8eb89862368edc0cc806acee6fca5163474`](https://huggingface.co/datasets/osunlp/Mind2Web/tree/17ece8eb89862368edc0cc806acee6fca5163474),
  CC BY 4.0
- Code and held-out handling terms:
  [`OSU-NLP-Group/Mind2Web@33bd95caeee7bba22dd08ecc935845e15c5e5dc7`](https://github.com/OSU-NLP-Group/Mind2Web/tree/33bd95caeee7bba22dd08ecc935845e15c5e5dc7),
  MIT

Task records contain `annotation_id`, website/domain fields, `confirmed_task`,
`action_reprs`, and `actions`. Each action contains its UID, raw/cleaned HTML, operation, and
positive/negative candidates. Normalized operations are `CLICK`, `TYPE`, and `SELECT`.

The dataset has 1,009 training tasks and official cross-task (252), cross-website (177), and
cross-domain (912) held-out splits. The 11 training JSON files total 5,931,387,773 bytes. The
protected `test.zip` is 567,745,122 bytes with SHA-256
`8f5fbe72afab942fe97cdf7fb397e179885d89b5c16862288e9a14bc6d41ca89`.
It contains exactly 15 ZipCrypto-encrypted DEFLATE JSON members: ten `test_domain` shards, three
`test_task` shards, and two `test_website` shards, totaling 6,107,912,752 plaintext bytes. In
production mode, the adapter requires that exact archive identity and member set, streams each
member's plaintext, and binds its bytes and SHA-256 to the corresponding immutable extracted
source. The resulting audit also records the archive format, per-member identity, aggregate
member hash, and official 912/252/177 task counts. This path was run on the authorized protected
archive and was also independently replayed from its raw chain.
The generic freezer independently requires the exact 15 path-to-split mappings from the machine
plan, matches every member's byte/hash/task/row identity to the adapter's source ledger, conserves
tasks and emitted prompt rows, and verifies the required audit self-hash. These hashes detect
drift; they are not signatures and do not replace rerunning the adapter on an authorized archive.
The pinned terms require private handling of unzipped held-out content and include a benchmark
canary. Do not commit or redistribute held-out prompts, HTML, identifiers, or labels; publish
hashes, counts, exclusions, and adapter code.

Native evaluation permits multiple positive target elements and reports element accuracy,
operation F1, step success, and task success. The production v2 label-blind lexical ranker exported
9,378 prompt-only rows across 1,341 tasks with a 1,771-byte maximum prompt. The v3 freezer retained
all 9,378 normalized prompts (18,330,792 bytes, SHA-256
`759a19c0135ecd5da3d657c2ca43d2c047148bbba243e34d389070756a113f27`). The earlier full-DOM v1
attempt remains historical fail-closed evidence: its 858,832-byte prompt exceeded the 524,288-byte
cap. Ranker recall has not been measured and must be reported as an evaluation ceiling; the same
frozen, gold-independent ranker must be reused for native evaluation. Treat HTML as inert untrusted
text and never load the optional pickle outside a disposable isolated process.

### WebLINX

- Compact annotated v1.0 dataset, peeled immutable commit:
  [`be2e19d624febb57173e98772c1312d041a6d3b1`](https://huggingface.co/datasets/McGill-NLP/WebLINX/tree/be2e19d624febb57173e98772c1312d041a6d3b1),
  CC BY-NC-SA 4.0
- Code release 0.3.2:
  [`b3f7010ca4677edc8c3ac705ad9bde20f7f29ce6`](https://github.com/McGill-NLP/weblinx/tree/b3f7010ca4677edc8c3ac705ad9bde20f7f29ce6),
  Apache-2.0
- [Pinned splits](https://huggingface.co/datasets/McGill-NLP/WebLINX/resolve/be2e19d624febb57173e98772c1312d041a6d3b1/splits.json)
- [Pinned formatting template](https://huggingface.co/datasets/McGill-NLP/WebLINX/resolve/be2e19d624febb57173e98772c1312d041a6d3b1/template.txt)
- [Pinned dataset card and terms](https://huggingface.co/datasets/McGill-NLP/WebLINX/resolve/be2e19d624febb57173e98772c1312d041a6d3b1/README.md)

The pinned compact `test_web.json.gz` is 2,187,263 bytes with SHA-256
`10d780712da997da9ff2d15d642aa199410ebe5d30d2ea3f9ba56fb044a745db`; `splits.json` is
38,210 bytes with SHA-256
`db6fd50e6b1ba053817ede3f2a8ec61a292ad2710dd7f4e300cf685f70d843e6`.

The annotated tag object is `9b687336b34f06e6b5ddb8b729373cd5d82ac85d`; record the peeled
commit above rather than only a tag name or tag-object hash. Use the compact `chat` release, not
the roughly 152.1 GB full recording or the roughly 65.5 GB BrowserGym derivative. Its train file
has 24,418 rows, 11,201,727 compressed bytes, and about 107.6 MB decoded. All compact chat splits
are 25,486,661 compressed bytes and about 260.2 MB decoded.

Rows contain `demo`, `turn`, current action, action history, utterances, candidates, cleaned HTML,
and viewport. The pinned template supports change, click, load, say, scroll, submit, and text-input
actions. Native metrics combine intent match, element IoU, text F1, and micro scores; exact
UID/string equality is a stricter derived metric. `test_web` is the preferred external slice
because it targets unseen websites.

The data license is noncommercial/share-alike and includes third-party-site and unlawful-use
conditions; the Apache code license does not supersede it. Public trained-weight licensing needs
legal review before WebLINX data is used for training. An audited official training row contains
an apparent email/password instruction. Before any export, exclude an entire demonstration if any
turn triggers the deterministic v1 patterns for email, labeled secret, SSN, labeled phone, or
Luhn-valid payment card, and publish reason counts plus hashes of excluded demo IDs. This is a
bounded sensitive-pattern filter, not a comprehensive PII detector: names, postal addresses,
dates of birth, unlabeled phones, and unlabeled secrets still require private manual/residual
review. The prompt adapter never emits the current action. It preserves bounded prior action
history as inert source text because real histories can contain natural-language quoting that is
not valid Python syntax; that history is never executed or passed to an evaluator. A restricted
AST parser remains available only for controlled utilities. Never evaluate any source action
string with `eval`.

The pinned `test_web` source audit covered all 4,856 compact-chat rows and all 211 declared split
demonstrations. Whole-demo privacy filtering excluded 65 demonstrations and 1,936 rows, leaving
2,920 canonical prompt-only rows from 146 demonstrations. The private derived export was
14,769,342 bytes with SHA-256
`05a5c29f5e188b318931acfa4f9c294578ddc30ad2b47336aa30114151d63688`. The v3 freezer removed
four normalized duplicates and retained 2,916 prompts (14,721,944 bytes, SHA-256
`1bc701bb14dad35fbba4e1f5fd0018249519a0d895f8ec441f30728938e5e979`). Demo-level reason counts
were email 61, labeled secret 43, and payment card 11 (one demo can trigger more than one reason).
Source and derived files remain private and ignored; only permitted aggregate/hash evidence is
published.

## Prompt-only adapter contract

Each source-specific offline adapter emits canonical JSONL rows with exactly:

```json
{"source_case_id":"stable component identifier","prompt":"exact model-visible text"}
```

The generic suite freezer then:

- rejects label/gold/action/output fields;
- hashes the upstream identifier before publication;
- verifies that the source-specific adapter audit names the declared adapter and binds its output
  byte size/SHA-256 to the supplied prompt rows;
- requires the audit's common `freeze_binding` envelope to match benchmark, immutable revision,
  split, production mode, prompt-only status, and output record identity (fixture-mode exports are
  rejected);
- invokes a fail-closed validator for every known paper suite: exact BFCL source/category
  identities and row arithmetic; BrowserGym capture/runtime/episode/group fingerprints; Mind2Web
  archive/member/split/task accounting; and WebLINX raw-source, label-isolation, and deterministic
  privacy-receipt accounting;
- verifies the exact machine-readable benchmark-plan bytes and matches the suite's benchmark,
  revision, adapter, and freeze split against that plan;
- binds the adapter audit and license-evidence bytes/SHA-256;
- canonicalizes ordering and records normalized-prompt deduplication;
- enforces file, row, and record caps; and
- writes a portable self-hashed provenance manifest plus a prompt JSONL accepted by
  `prepare_corpus.py`.

Protected outputs stay private. The corpus list manifest binds their exact byte/hash identities,
requires every component suite to bind the same benchmark-plan byte/hash identity, while the paper
artifact publishes only permitted provenance, counts, transformations, and hashes. These hashes
establish integrity of the selected local artifacts, not authorship or upstream authenticity;
production receipts still require acquisition from the pinned sources and external review.

The source adapters are executable and intentionally offline:

```bash
# BFCL: first author a content-addressed source manifest using only the four pinned v4 inputs.
PYTHONPATH=src python scripts/export_bfcl_prompt_rows.py \
  private/bfcl/source-manifest.json \
  --out private/bfcl/adapter-prompts.jsonl \
  --audit private/bfcl/adapter-audit.json

# BrowserGym controlled-acquisition reproduction command. Use already-installed clean, pinned
# source checkouts, the frozen runtime manifest, and the pinned Playwright Chromium installation.
PYTHONPATH=src python scripts/capture_browsergym_goals.py \
  --browsergym-checkout <BrowserGym-checkout> \
  --miniwob-checkout <miniwob-plusplus-checkout> \
  --browser-executable <chromium-executable> \
  --browser-installation <chromium-1117-directory> \
  --runtime-manifest configs/data/browsergym-capture-runtime-darwin-arm64-py312.json \
  --capture private/browsergym/reset-capture.jsonl \
  --receipt private/browsergym/reset-capture.receipt.json

# Production export is bound to the jointly verified capture and producer receipt.
PYTHONPATH=src python scripts/export_browsergym_prompt_rows.py \
  private/browsergym/reset-capture.jsonl \
  --receipt private/browsergym/reset-capture.receipt.json \
  --capture-bytes 348513 \
  --capture-sha256 128f7f6be8d5b52f745523b0bca4517fdaf8107044eee5a76366464ac10079ff \
  --out private/browsergym/adapter-prompts.jsonl \
  --audit private/browsergym/adapter-audit.json

# Mind2Web and WebLINX remain private; each source argument includes its precomputed identity.
PYTHONPATH=src python scripts/export_mind2web_prompt_rows.py \
  --revision 17ece8eb89862368edc0cc806acee6fca5163474 \
  --split cross_domain+cross_task+cross_website \
  --archive private/mind2web/test.zip 567745122 \
    8f5fbe72afab942fe97cdf7fb397e179885d89b5c16862288e9a14bc6d41ca89 \
  --member-source test_domain/test_domain_0.json \
    private/mind2web/test_domain/test_domain_0.json <bytes> <sha256> \
  --member-source test_domain/test_domain_1.json \
    private/mind2web/test_domain/test_domain_1.json <bytes> <sha256> \
  --member-source test_domain/test_domain_2.json \
    private/mind2web/test_domain/test_domain_2.json <bytes> <sha256> \
  --member-source test_domain/test_domain_3.json \
    private/mind2web/test_domain/test_domain_3.json <bytes> <sha256> \
  --member-source test_domain/test_domain_4.json \
    private/mind2web/test_domain/test_domain_4.json <bytes> <sha256> \
  --member-source test_domain/test_domain_5.json \
    private/mind2web/test_domain/test_domain_5.json <bytes> <sha256> \
  --member-source test_domain/test_domain_6.json \
    private/mind2web/test_domain/test_domain_6.json <bytes> <sha256> \
  --member-source test_domain/test_domain_7.json \
    private/mind2web/test_domain/test_domain_7.json <bytes> <sha256> \
  --member-source test_domain/test_domain_8.json \
    private/mind2web/test_domain/test_domain_8.json <bytes> <sha256> \
  --member-source test_domain/test_domain_9.json \
    private/mind2web/test_domain/test_domain_9.json <bytes> <sha256> \
  --member-source test_task/test_task_0.json \
    private/mind2web/test_task/test_task_0.json <bytes> <sha256> \
  --member-source test_task/test_task_1.json \
    private/mind2web/test_task/test_task_1.json <bytes> <sha256> \
  --member-source test_task/test_task_2.json \
    private/mind2web/test_task/test_task_2.json <bytes> <sha256> \
  --member-source test_website/test_website_0.json \
    private/mind2web/test_website/test_website_0.json <bytes> <sha256> \
  --member-source test_website/test_website_1.json \
    private/mind2web/test_website/test_website_1.json <bytes> <sha256> \
  --out private/mind2web/adapter-prompts.jsonl \
  --audit private/mind2web/adapter-audit.json
PYTHONPATH=src python scripts/export_weblinx_prompt_rows.py \
  --revision be2e19d624febb57173e98772c1312d041a6d3b1 --split test_web \
  --chat <test_web.json.gz> <bytes> <sha256> \
  --splits <splits.json> <bytes> <sha256> \
  --out private/weblinx/adapter-prompts.jsonl \
  --audit private/weblinx/adapter-audit.json
```

For each suite, author a strict freezer contract containing the prompt JSONL under `sources`, the
source-specific audit under `adapter_provenance`, the exact
`configs/data/evaluation-benchmarks-paper.yaml` bytes under `benchmark_plan`, and archived
upstream license bytes under `license_evidence`. The freezer selects that suite's machine-plan
entry and exact-matches benchmark, revision, adapter, and split before freezing:

The BrowserGym contract additionally requires exactly two `raw_artifacts` entries. The role
`browsergym_capture` must identify the canonical capture and `browsergym_receipt` must identify its
matching producer receipt; missing, duplicate, or additional roles fail closed. The production
contract uses:

```json
{
  "raw_artifacts": [
    {
      "bytes": 348513,
      "name": "browsergym-reset-capture",
      "path": "browsergym-miniwob-reset-goals.jsonl",
      "role": "browsergym_capture",
      "sha256": "128f7f6be8d5b52f745523b0bca4517fdaf8107044eee5a76366464ac10079ff"
    },
    {
      "bytes": 6538,
      "name": "browsergym-reset-capture-receipt",
      "path": "browsergym-miniwob-reset-goals.receipt.json",
      "role": "browsergym_receipt",
      "sha256": "b04318c36579a05d3f61a40ea09c1f1c0bd1e004a534b2b5d18305b50e68ebea"
    }
  ]
}
```

```bash
PYTHONPATH=src python scripts/freeze_evaluation_denylist_suite.py \
  private/<suite>/freeze-contract.json \
  --output private/<suite>/prompts.jsonl \
  --manifest private/<suite>/prompts.provenance.json

PYTHONPATH=src python scripts/build_evaluation_denylist_manifest.py \
  --suite-provenance private/bfcl/prompts.provenance.json \
  --suite-provenance private/browsergym/prompts.provenance.json \
  --suite-provenance private/mind2web/prompts.provenance.json \
  --suite-provenance private/weblinx/prompts.provenance.json \
  --out private/paper-external-denylists-v3.json
```

The final command is deterministic and non-clobbering. Its completed v3 manifest has 1,880 bytes,
the four required suites, 34,068,955 combined suite-output bytes, and self-hash
`826f53f5699f3c4b8f311a9fe70561f5b7d9aa99ce42ce250548e46aa644010b`. The composer and
`prepare_corpus.py` both reverify each per-suite self-hash, provenance identity, prompt-output
identity, prompt-only isolation flags, and the shared benchmark-plan identity.

## Pretraining and evaluation chronology

1. Pin benchmark revisions and adapters.
2. Produce private prompt-only exports before tokenizer fitting.
3. Build the four-external-suite provenance manifest, supply the three config-hash-pinned local
   suites alongside it, and use the same seven inputs for every pretrain/midtrain repack.
4. Freeze data, tokenizer, checkpoints, and all post-training text.
5. For ordinary external evaluation, run the pinned native evaluators and call the result
   revision-frozen/locally held out.
6. For a chronological-freshness claim, acquire a different post-training immutable revision,
   hidden steward export, or newly instantiated procedural seed set.
7. Freeze and externally timestamp its source/selection/analysis identities before running either
   system.

Hash selection after training controls selection determinism; it does not make an old public
benchmark new.

## Remaining blockers

- Do not generalize the completed supplied-denylist audit into an exhaustive contamination or
  benchmark-canary guarantee; its frozen limitations explicitly describe the bounded coverage.
- Confirm the BFCL unique-ground-truth subset has enough cases/clusters and passes scorer
  conformance, or use only the official evaluator.
- Implement a private sharded multi-gold/native-metric path for Mind2Web and WebLINX.
- Implement BrowserGym episode-plan, isolated runner, finite timeout, paired reward comparison,
  and 41-group bootstrap; the completed reset-only prompt freeze is not episode-score evidence.
- Measure and publish the frozen Mind2Web ranker's held-out recall ceiling; the v2/v3 decontamination
  path is complete, but it is not native-evaluation evidence.
- Externally archive the private WebLINX exclusion receipt with its aggregate reason counts and
  hashed excluded demonstration IDs for final artifact review.
- Obtain legal review before training on or publicly releasing weights derived from WebLINX.
- Externally timestamp the final protocols before outcomes are inspected.
