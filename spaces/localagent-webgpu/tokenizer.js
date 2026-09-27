/* LocalAgent browser tokenizers.
 *
 * Supports the two tokenizer contracts emitted by the Python exporter:
 *   - utf-8-bytes: one token id per UTF-8 byte.
 *   - bytelevel-bpe: Hugging Face tokenizers ByteLevel + BPE JSON, without transformers.js.
 *
 * The BPE implementation intentionally supports only the exact training configuration in
 * localagent.model.tokenizer.train_bpe. It rejects incompatible tokenizer JSON rather than
 * producing plausible but wrong token ids.
 */
(function installLocalAgentTokenizer(root) {
  "use strict";

  const textEncoder = new TextEncoder();
  const textDecoder = new TextDecoder("utf-8", { fatal: false });

  function byteUnicodeTables() {
    const visibleBytes = [];
    for (let value = 33; value <= 126; value++) visibleBytes.push(value);
    for (let value = 161; value <= 172; value++) visibleBytes.push(value);
    for (let value = 174; value <= 255; value++) visibleBytes.push(value);

    const bytes = [...visibleBytes];
    const codePoints = [...visibleBytes];
    let extra = 0;
    for (let value = 0; value <= 255; value++) {
      if (!visibleBytes.includes(value)) {
        bytes.push(value);
        codePoints.push(256 + extra);
        extra += 1;
      }
    }

    const encode = new Array(256);
    const decode = new Map();
    for (let index = 0; index < bytes.length; index++) {
      const symbol = String.fromCodePoint(codePoints[index]);
      encode[bytes[index]] = symbol;
      decode.set(symbol, bytes[index]);
    }
    return { encode, decode };
  }

  const BYTE_UNICODE = byteUnicodeTables();
  const GPT2_PATTERN =
    /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  function regexEscape(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  class Utf8ByteTokenizer {
    constructor(meta) {
      this.encoding = "utf-8-bytes";
      this.eosId = meta.eos_id ?? 0;
      this.vocabSize = meta.vocab_size;
    }

    encode(text, addEos = false) {
      const ids = Array.from(textEncoder.encode(text));
      if (addEos) ids.push(this.eosId);
      return ids;
    }

    decode(ids, stopAtEos = true) {
      const bytes = [];
      for (const id of ids) {
        if (stopAtEos && id === this.eosId) break;
        bytes.push(id);
      }
      return textDecoder.decode(new Uint8Array(bytes));
    }
  }

  class ByteLevelBPETokenizer {
    constructor(meta, spec) {
      this.encoding = "bytelevel-bpe";
      this.eosId = meta.eos_id ?? 0;
      this.vocabSize = meta.vocab_size;
      this.vocab = spec.model?.vocab || {};
      this.idToToken = new Map(
        Object.entries(this.vocab).map(([token, id]) => [Number(id), token])
      );
      this.addedByText = new Map();
      this.addedById = new Map();
      for (const token of spec.added_tokens || []) {
        this.addedByText.set(token.content, token.id);
        this.addedById.set(token.id, token.content);
      }
      this.mergeRanks = new Map();
      for (const [rank, rawMerge] of (spec.model?.merges || []).entries()) {
        const pair = Array.isArray(rawMerge)
          ? rawMerge
          : [rawMerge.slice(0, rawMerge.indexOf(" ")), rawMerge.slice(rawMerge.indexOf(" ") + 1)];
        this.mergeRanks.set(this.pairKey(pair[0], pair[1]), rank);
      }
      const specials = [...this.addedByText.keys()].sort((a, b) => b.length - a.length);
      this.specialPattern = specials.length
        ? new RegExp(`(${specials.map(regexEscape).join("|")})`, "gu")
        : null;
      this.assertCompatible(spec);
    }

    assertCompatible(spec) {
      if (spec.model?.type !== "BPE") {
        throw new Error(`Unsupported tokenizer model ${spec.model?.type}; expected BPE.`);
      }
      if (spec.pre_tokenizer?.type !== "ByteLevel" || spec.pre_tokenizer?.add_prefix_space) {
        throw new Error("Browser BPE requires ByteLevel(add_prefix_space=false).");
      }
      if (spec.pre_tokenizer?.use_regex === false) {
        throw new Error("Browser BPE requires the ByteLevel GPT-2 regex pre-tokenizer.");
      }
      if (spec.decoder?.type !== "ByteLevel") {
        throw new Error("Browser BPE requires a ByteLevel decoder.");
      }
      if (this.addedById.get(this.eosId) !== "<|end|>") {
        throw new Error(`BPE eos id ${this.eosId} is not <|end|>.`);
      }
      if (Object.keys(this.vocab).length !== this.vocabSize) {
        throw new Error(
          `Tokenizer vocab has ${Object.keys(this.vocab).length} entries; model expects ${this.vocabSize}.`
        );
      }
    }

    pairKey(left, right) {
      return `${left}\u0000${right}`;
    }

    encodePiece(piece) {
      let symbols = Array.from(textEncoder.encode(piece), (byte) => BYTE_UNICODE.encode[byte]);
      while (symbols.length > 1) {
        let bestRank = Infinity;
        let bestLeft = null;
        let bestRight = null;
        for (let index = 0; index < symbols.length - 1; index++) {
          const rank = this.mergeRanks.get(this.pairKey(symbols[index], symbols[index + 1]));
          if (rank != null && rank < bestRank) {
            bestRank = rank;
            bestLeft = symbols[index];
            bestRight = symbols[index + 1];
          }
        }
        if (!Number.isFinite(bestRank)) break;
        const merged = [];
        for (let index = 0; index < symbols.length;) {
          if (
            index < symbols.length - 1 &&
            symbols[index] === bestLeft &&
            symbols[index + 1] === bestRight
          ) {
            merged.push(bestLeft + bestRight);
            index += 2;
          } else {
            merged.push(symbols[index]);
            index += 1;
          }
        }
        symbols = merged;
      }
      return symbols.map((symbol) => {
        const id = this.vocab[symbol];
        if (id == null) {
          throw new Error(`Tokenizer JSON has no id for byte-level BPE symbol ${JSON.stringify(symbol)}.`);
        }
        return id;
      });
    }

    encodeNormal(text) {
      const ids = [];
      for (const piece of text.match(GPT2_PATTERN) || []) {
        ids.push(...this.encodePiece(piece));
      }
      return ids;
    }

    encode(text, addEos = false) {
      const ids = [];
      if (!this.specialPattern) {
        ids.push(...this.encodeNormal(text));
      } else {
        this.specialPattern.lastIndex = 0;
        let position = 0;
        for (const match of text.matchAll(this.specialPattern)) {
          if (match.index > position) {
            ids.push(...this.encodeNormal(text.slice(position, match.index)));
          }
          ids.push(this.addedByText.get(match[0]));
          position = match.index + match[0].length;
        }
        if (position < text.length) ids.push(...this.encodeNormal(text.slice(position)));
      }
      if (addEos) ids.push(this.eosId);
      return ids;
    }

    decode(ids, stopAtEos = true) {
      let output = "";
      let bytes = [];
      const flush = () => {
        if (bytes.length) {
          output += textDecoder.decode(new Uint8Array(bytes));
          bytes = [];
        }
      };
      for (const id of ids) {
        if (stopAtEos && id === this.eosId) break;
        if (this.addedById.has(id)) {
          flush();
          output += this.addedById.get(id);
          continue;
        }
        const token = this.idToToken.get(id);
        if (token == null) throw new Error(`Unknown BPE token id ${id}.`);
        for (const symbol of Array.from(token)) {
          const byte = BYTE_UNICODE.decode.get(symbol);
          if (byte == null) {
            throw new Error(`BPE token contains a non-ByteLevel symbol ${JSON.stringify(symbol)}.`);
          }
          bytes.push(byte);
        }
      }
      flush();
      return output;
    }
  }

  async function fromMeta(meta, loadJson = async (path) => {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to fetch ${path}: HTTP ${response.status}`);
    return response.json();
  }) {
    if (meta.encoding === "utf-8-bytes") return new Utf8ByteTokenizer(meta);
    if (meta.encoding === "bytelevel-bpe") {
      if (!meta.tokenizer_file) throw new Error("BPE metadata is missing tokenizer_file.");
      return new ByteLevelBPETokenizer(meta, await loadJson(meta.tokenizer_file));
    }
    throw new Error(`Unsupported tokenizer encoding ${JSON.stringify(meta.encoding)}.`);
  }

  const api = { Utf8ByteTokenizer, ByteLevelBPETokenizer, fromMeta };
  root.LocalAgentTokenizer = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
