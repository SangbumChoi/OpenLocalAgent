"""Generate LocalAgent promo/listing image assets at exact pixel sizes (PIL, no app capture needed).

Outputs (runs/promo/):
  portrait_1_pipeline.png   636x1048  - generable dispatch pipeline
  portrait_2_toolcall.png   636x1048  - natural language -> tool call (terminal)
  portrait_3_results.png    636x1048  - held-out accuracy + on-device facts
  landscape_1_flow.png     1504x741   - from-scratch training pipeline + flywheel
  thumbnail_logo.png       1942x828   - wordmark + tagline (logo-style feature graphic)

Dark terminal theme; Korean titles + English code/metrics (mixed). Fonts under runs/promo/fonts/.
"""
from PIL import Image, ImageDraw, ImageFont

OUT = "runs/promo"
KR_B = f"{OUT}/fonts/NotoKR-Bold.otf"
KR_R = f"{OUT}/fonts/NotoKR-Reg.otf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# palette (GitHub-dark inspired)
BG = (13, 17, 23)
PANEL = (22, 27, 34)
PANEL2 = (28, 34, 43)
BORDER = (48, 54, 61)
TX = (230, 237, 243)
MUTED = (139, 148, 158)
ACCENT = (255, 210, 30)   # HF yellow
GREEN = (63, 185, 80)
BLUE = (88, 166, 255)
PURPLE = (188, 140, 255)
PINK = (247, 129, 102)

_fc = {}


def F(path, size):
    k = (path, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]


def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def T(d, xy, s, font, fill=TX, anchor="la", spacing=6):
    d.multiline_text(xy, s, font=font, fill=fill, anchor=anchor, spacing=spacing)


def w(d, s, font):
    return d.textbbox((0, 0), s, font=font)[2]


def has_kr(s):
    return any("가" <= c <= "힣" for c in s)


def codefont(s, size):
    """Mono for ASCII code; Korean font when the string contains Hangul (mono has no CJK glyphs)."""
    return F(KR_R, size) if has_kr(s) else F(MONO, size)


def bg(W, H, top=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # subtle top vignette band
    if top:
        for y in range(160):
            a = int(18 * (1 - y / 160))
            d.line([(0, y), (W, y)], fill=(BG[0] + a, BG[1] + a, BG[2] + a + 4))
    return img, d


def chip(d, x, y, s, font, fg, bgc):
    pad = 14
    tw = w(d, s, font)
    h = font.size + 14
    rrect(d, [x, y, x + tw + pad * 2, y + h], h // 2, fill=bgc)
    T(d, (x + pad, y + h / 2), s, font, fill=fg, anchor="lm")
    return x + tw + pad * 2


def logo_mark(d, cx, cy, s, accent=ACCENT):
    """A rounded terminal tile with a '>_' prompt."""
    rrect(d, [cx - s, cy - s, cx + s, cy + s], s * 0.28, fill=PANEL2, outline=BORDER, width=max(2, s // 22))
    f = F(MONO_B, int(s * 1.15))
    T(d, (cx - s * 0.06, cy), ">", f, fill=accent, anchor="mm")
    d.line([(cx + s * 0.18, cy + s * 0.42), (cx + s * 0.62, cy + s * 0.42)], fill=accent, width=max(2, s // 9))


# ---------- A. portrait 1 — pipeline ----------
def portrait_pipeline():
    W, H = 636, 1048
    img, d = bg(W, H, top=True)
    logo_mark(d, 60, 70, 30)
    T(d, (104, 56), "LocalAgent", F(KR_B, 34), fill=TX)
    T(d, (106, 96), "28M · byte-level · from-scratch", F(MONO, 16), fill=MUTED)

    T(d, (44, 168), "툴을 추가해도\n재학습이 필요 없다", F(KR_B, 44), fill=TX, spacing=10)
    T(d, (46, 286), "자연어 요청을 한 줄짜리 새 도구까지\n생성형으로 디스패치하는 파이프라인", F(KR_R, 19), fill=MUTED, spacing=8)

    steps = [
        ("request", "\"서울 날씨 알려줘\"", MUTED, MONO),
        ("route head", "5-way modality gate", BLUE, KR_B),
        ("two-tower selector", "scores ANY tool by description", PURPLE, KR_B),
        ("pointer-copy args", "slots <- input spans", GREEN, KR_B),
        ("tool(args)", "get_weather(city=\"Seoul\")", ACCENT, MONO),
    ]
    x0, x1 = 46, W - 46
    y = 370
    bh = 104
    gap = 30
    for i, (title, sub, c, tf) in enumerate(steps):
        rrect(d, [x0, y, x1, y + bh], 16, fill=PANEL, outline=BORDER, width=2)
        d.line([(x0, y + 14), (x0, y + bh - 14)], fill=c, width=5)
        T(d, (x0 + 26, y + 28), title, F(KR_B if tf is KR_B else MONO_B, 23), fill=c)
        T(d, (x0 + 26, y + 62), sub, codefont(sub, 16), fill=MUTED)
        if i < len(steps) - 1:
            ay = y + bh + gap / 2
            d.line([(W / 2, y + bh + 4), (W / 2, y + bh + gap - 4)], fill=BORDER, width=3)
            d.polygon([(W / 2 - 7, ay - 1), (W / 2 + 7, ay - 1), (W / 2, ay + 8)], fill=BORDER)
        y += bh + gap

    T(d, (W / 2, H - 40), "request → route → select → args → tool", F(MONO, 17), fill=MUTED, anchor="mm")
    img.save(f"{OUT}/portrait_1_pipeline.png")


# ---------- B. portrait 2 — tool call terminal ----------
def portrait_toolcall():
    W, H = 636, 1048
    img, d = bg(W, H, top=True)
    T(d, (44, 60), "자연어 → 툴 호출", F(KR_B, 42), fill=TX)
    T(d, (46, 124), "임의의 도구 집합에서 in-context로 호출을 생성", F(KR_R, 19), fill=MUTED)

    def terminal(x, y, ww, hh, prompt, calls):
        rrect(d, [x, y, x + ww, y + hh], 14, fill=(11, 14, 20), outline=BORDER, width=2)
        for i, c in enumerate([PINK, ACCENT, GREEN]):
            d.ellipse([x + 18 + i * 22, y + 16, x + 30 + i * 22, y + 28], fill=c)
        T(d, (x + ww - 16, y + 22), "localagent", F(MONO, 14), fill=MUTED, anchor="rm")
        cy = y + 56
        T(d, (x + 20, cy), "$ user", F(MONO_B, 15), fill=GREEN)
        T(d, (x + 20, cy + 24), prompt, F(KR_R, 18), fill=TX)
        cy += 64
        T(d, (x + 20, cy), "→ tool_call", F(MONO_B, 15), fill=BLUE)
        cy += 26
        for ln, col in calls:
            T(d, (x + 28, cy), ln, codefont(ln, 15), fill=col)
            cy += 22

    terminal(46, 180, W - 92, 250, "서울 날씨 알려줘", [
        ('{"name": "get_weather",', TX),
        ('   "arguments": {"city": "Seoul"}}', ACCENT),
    ])
    terminal(46, 456, W - 92, 250, "이 PDF 요약하고 한국어로 번역해줘", [
        ('{"name": "summarize",', TX),
        ('   "arguments": {"path": "report.pdf"}}', ACCENT),
        ('{"name": "translate",', TX),
        ('   "arguments": {"to": "ko"}}', ACCENT),
    ])
    terminal(46, 732, W - 92, 232, "지금 몇 시야? (도구 없이)", [
        ("// no matching tool → abstain", MUTED),
        ('"현재 시각 도구가 없습니다."', TX),
    ])
    T(d, (W / 2, H - 32), "parallel calls · abstention · pointer-copied args", F(MONO, 15), fill=MUTED, anchor="mm")
    img.save(f"{OUT}/portrait_2_toolcall.png")


# ---------- C. portrait 3 — results ----------
def portrait_results():
    W, H = 636, 1048
    img, d = bg(W, H, top=True)
    T(d, (44, 60), "작지만, 진짜 동작한다", F(KR_B, 42), fill=TX)
    T(d, (46, 124), "Held-out (disjoint phrasings & slot values)", F(MONO, 16), fill=MUTED)

    stats = [
        ("53%", "free-form OOD\ncall-name (45)", BLUE),
        ("63%", "paraphrase\nselection (100)", PURPLE),
        ("72%", "referent-conditioned\nselection (46)", GREEN),
    ]
    y = 196
    ch = 150
    for big, lab, c in stats:
        rrect(d, [46, y, W - 46, y + ch], 18, fill=PANEL, outline=BORDER, width=2)
        d.line([(46, y + 16), (46, y + ch - 16)], fill=c, width=6)
        T(d, (74, y + ch / 2), big, F(KR_B, 64), fill=c, anchor="lm")
        T(d, (240, y + ch / 2), lab, F(KR_R, 21), fill=TX, anchor="lm", spacing=6)
        y += ch + 24

    y += 8
    rrect(d, [46, y, W - 46, y + 210], 18, fill=PANEL2, outline=BORDER, width=2)
    T(d, (74, y + 26), "on-device, from scratch", F(KR_B, 24), fill=ACCENT)
    facts = ["28M params · vocab 256 (byte-level)",
             "CPU / GPU / NPU",
             "PyTorch · GGUF · ONNX · ExecuTorch",
             "data flywheel: improves from its own runs"]
    fy = y + 70
    for fct in facts:
        d.ellipse([74, fy + 7, 84, fy + 17], fill=GREEN)
        T(d, (98, fy), fct, F(MONO, 16), fill=TX)
        fy += 32
    T(d, (W / 2, H - 32), "no fixed-N classifier — adding a tool is one row", F(MONO, 15), fill=MUTED, anchor="mm")
    img.save(f"{OUT}/portrait_3_results.png")


# ---------- D. landscape — training flow + flywheel ----------
def landscape_flow():
    W, H = 1504, 741
    img, d = bg(W, H, top=True)
    logo_mark(d, 64, 70, 34)
    T(d, (114, 50), "from-scratch 학습 파이프라인", F(KR_B, 40), fill=TX)
    T(d, (116, 100), "pretrain → SFT → GRPO, then a self-improving data flywheel", F(MONO, 18), fill=MUTED)

    stages = [
        ("Pretrain", "FineWeb-edu\nnext-byte LM", BLUE),
        ("SFT", "synthetic + Hermes\nfunction-calling", PURPLE),
        ("GRPO", "verifiable reward\nAST-match = 1", GREEN),
        ("Deploy", "Agent.from_checkpoint\nCPU/GPU/NPU", ACCENT),
    ]
    n = len(stages)
    pad = 70
    gap = 46
    bw = (W - pad * 2 - gap * (n - 1)) / n
    y = 230
    bh = 200
    cxs = []
    for i, (t, s, c) in enumerate(stages):
        x = pad + i * (bw + gap)
        rrect(d, [x, y, x + bw, y + bh], 18, fill=PANEL, outline=BORDER, width=2)
        d.line([(x + 18, y), (x + bw - 18, y)], fill=c, width=6)
        T(d, (x + bw / 2, y + 56), t, F(KR_B, 34), fill=c, anchor="mm")
        T(d, (x + bw / 2, y + 130), s, F(MONO, 17), fill=MUTED, anchor="mm", spacing=8)
        cxs.append((x, x + bw))
        if i < n - 1:
            mx = x + bw + gap / 2
            d.line([(x + bw + 8, y + bh / 2), (x + bw + gap - 8, y + bh / 2)], fill=BORDER, width=4)
            d.polygon([(mx + 4, y + bh / 2 - 8), (mx + 4, y + bh / 2 + 8), (mx + 14, y + bh / 2)], fill=BORDER)

    # flywheel loop from Deploy back to SFT
    fy = y + bh + 78
    x_dep = (cxs[3][0] + cxs[3][1]) / 2
    x_sft = (cxs[1][0] + cxs[1][1]) / 2
    d.line([(x_dep, y + bh + 6), (x_dep, fy)], fill=PINK, width=4)
    d.line([(x_dep, fy), (x_sft, fy)], fill=PINK, width=4)
    d.line([(x_sft, fy), (x_sft, y + bh + 6)], fill=PINK, width=4)
    d.polygon([(x_sft - 8, y + bh + 18), (x_sft + 8, y + bh + 18), (x_sft, y + bh + 6)], fill=PINK)
    T(d, ((x_dep + x_sft) / 2, fy + 22), "data flywheel — failures become new training data",
      F(KR_R, 20), fill=PINK, anchor="mm")

    img.save(f"{OUT}/landscape_1_flow.png")


# ---------- E. thumbnail — wordmark ----------
def thumbnail():
    W, H = 1942, 828
    img, d = bg(W, H)
    # faint grid / scanline texture on the right
    for x in range(W // 2, W, 46):
        d.line([(x, 0), (x, H)], fill=(18, 23, 30))
    for yy in range(0, H, 46):
        d.line([(W // 2, yy), (W, yy)], fill=(18, 23, 30))
    # accent corner glow
    rrect(d, [-200, H - 260, 360, H + 200], 200, fill=(20, 25, 33))

    logo_mark(d, 250, H // 2, 120)
    x = 410
    T(d, (x, H // 2 - 96), "LocalAgent", F(KR_B, 132), fill=TX)
    T(d, (x + 4, H // 2 + 64), "A  < 100M  from-scratch  tool-calling  agent", F(MONO, 40), fill=ACCENT)
    T(d, (x + 6, H // 2 + 128), "온디바이스 · 순수 PyTorch · 바이트 단위 · 데이터 플라이휠",
      F(KR_R, 34), fill=MUTED)

    # chips row
    cy = H - 96
    cx = x + 6
    for s, c in [("28M params", BLUE), ("CPU/GPU/NPU", GREEN), ("GGUF·ONNX·ExecuTorch", PURPLE),
                 ("tool calling", ACCENT)]:
        cx = chip(d, cx, cy, s, F(MONO, 26), BG if c is ACCENT else TX,
                  c if c is ACCENT else PANEL2) + 16
    img.save(f"{OUT}/thumbnail_logo.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    portrait_pipeline()
    portrait_toolcall()
    portrait_results()
    landscape_flow()
    thumbnail()
    print("SAVED 5 assets to", OUT, flush=True)
