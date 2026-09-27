#!/usr/bin/env python3
"""Emit the paper as NeurIPS 2026 LaTeX from the same receipt-driven content module.

The HTML pipeline stays the fast-iteration path; this emitter produces the submission PDF:
main sections (9-page budget), references, the official checklist, then the appendices — one
document, as the 2026 call requires.
"""

import html.parser
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import content

UNICODE = {
    "—": "---", "–": "--", "−": r"$-$", "‖": r"$\|$", "×": r"$\times$", "≈": r"$\approx$",
    "≥": r"$\geq$", "≤": r"$\leq$", "±": r"$\pm$", "→": r"$\rightarrow$",
    "·": r"$\cdot$", "⅓": r"$\tfrac{1}{3}$", "α": r"$\alpha$", "β": r"$\beta$",
    "θ": r"$\theta$", "Δ": r"$\Delta$",
    "⁻³": r"$^{-3}$", "⁻¹": r"$^{-1}$", "¹⁰": r"$^{10}$",
    "⁶": r"$^{6}$", "³": r"$^{3}$", "⁻": r"$^{-}$", "¹": r"$^{1}$", "⁰": r"$^{0}$",
    "§": r"\S", "†": r"$\dagger$", "‡": r"$\ddagger$", "…": r"\ldots", "ﬁ": "fi", "ﬂ": "fl",
    "Ż": r"\.{Z}", "ż": r"\.{z}", "ü": r"\"u", "é": r"\'e", "’": "'", "‘": "`",
    "“": "``", "”": "''",
}


SUPER_MAP = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
             "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
             "\u207b": "-"}


def _collapse_superscripts(text: str) -> str:
    """A run of unicode superscript chars becomes one math superscript group.

    Char-by-char mapping produced $^{-1}$$^{0}$ for 10^-10, which XeTeX rejects as a double
    superscript once the $$ pairs collapse. Handling the whole run first keeps multi-digit
    exponents legal."""
    pattern = "[" + "".join(map(re.escape, SUPER_MAP)) + "]+"
    # Emit a placeholder that survives esc()'s $/^ escaping; restored at the end of esc().
    return re.sub(pattern,
                  lambda m: "@@SUP" + "".join(SUPER_MAP[c] for c in m.group(0)) + "PUS@@", text)


def esc(text: str) -> str:
    text = _collapse_superscripts(text)
    text = text.replace("\\", r"\textbackslash{}")
    for char, repl in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                      ("$", r"\$"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        text = text.replace(char, repl)
    for char, repl in UNICODE.items():
        text = text.replace(char, repl)
    # ×10⁻³-style leftovers: superscript digits already mapped; collapse $..$$..$ pairs
    text = re.sub(r"\$\s*\$", "", text)
    text = re.sub(r"@@SUP(-?[0-9]+)PUS@@", lambda m: "$^{" + m.group(1) + "}$", text)
    return text


def prose(body: str) -> str:
    """HTML-ish paragraph markup to LaTeX."""
    out = body
    # Every entity the source actually uses maps to the character the SYMBOLS table knows;
    # an unmapped one reaches the PDF verbatim (&mdash; and &minus; did, 26 times).
    for entity, char in (("&rarr;", "→"), ("&times;", "×"), ("&ndash;", "–"),
                         ("&mdash;", "—"), ("&minus;", "−"), ("&Delta;", "Δ"),
                         ("&asymp;", "≈"), ("&plusmn;", "±"), ("&nbsp;", " "),
                         ("&alpha;", "α"), ("&beta;", "β"), ("&theta;", "θ"),
                         ("&gt;", ">"), ("&lt;", "<"), ("&le;", "≤"), ("&ge;", "≥"),
                         ("&hellip;", "…"), ("&iuml;", "ï")):
        out = out.replace(entity, char)
    out = out.replace("&amp;", "&")
    # protect tags, escape, restore
    tokens = {}
    def stash(match):
        key = f"\x00{len(tokens)}\x00"
        tokens[key] = match.group(0)
        return key
    out = re.sub(r"</?(?:p|b|em|u|span[^>]*)>", stash, out)
    out = esc(out)
    for key, tag in tokens.items():
        if tag == "<b>": repl = r"\textbf{"
        elif tag == "</b>": repl = "}"
        elif tag == "<em>": repl = r"\emph{"
        elif tag == "</em>": repl = "}"
        elif tag == "<u>": repl = r"\underline{"
        elif tag == "</u>": repl = "}"
        elif tag.startswith("<span"): repl = r"\texttt{"
        elif tag == "</span>": repl = "}"
        elif tag == "<p>": repl = ""
        elif tag == "</p>": repl = "\n"
        else: repl = ""
        out = out.replace(key, repl)
    return out


class TableParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.caption = ""
        self.klass = ""
        self.head, self.body = [], []
        self.row, self.cell = None, None
        self.in_head = False
        self.cell_attrs = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self.caption = a.get("data-caption", "")
            self.klass = a.get("class", "")
        elif tag == "thead": self.in_head = True
        elif tag == "tr": self.row = []
        elif tag in ("td", "th"):
            self.cell = []
            self.cell_attrs = a
        elif tag == "b" and self.cell is not None: self.cell.append("\x01")
        elif tag == "span" and self.cell is not None: self.cell.append("\x02")
        elif tag == "u" and self.cell is not None: self.cell.append("\x05")

    def handle_endtag(self, tag):
        if tag == "thead": self.in_head = False
        elif tag == "tr" and self.row is not None:
            (self.head if self.in_head else self.body).append(self.row)
            self.row = None
        elif tag in ("td", "th") and self.row is not None:
            text = "".join(self.cell or [])
            self.row.append((text, self.cell_attrs))
            self.cell = None
        elif tag == "b" and self.cell is not None: self.cell.append("\x03")
        elif tag == "span" and self.cell is not None: self.cell.append("\x04")
        elif tag == "u" and self.cell is not None: self.cell.append("\x06")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def cell_tex(text: str) -> str:
    out = esc(text.strip())
    out = out.replace("\x01", r"\textbf{").replace("\x03", "}")
    out = out.replace("\x02", r"{\scriptsize ").replace("\x04", "}")
    out = out.replace("\x05", r"\underline{").replace("\x06", "}")
    return out


def table_tex(html_table: str, key: str) -> str:
    parser = TableParser()
    parser.feed(html_table)
    all_rows = parser.head + parser.body
    ncols = max(
        sum(int(a.get("colspan", 1)) for _, a in row) for row in all_rows) if all_rows else 1
    # Column-group divider bars: a first-header-row cell spanning several columns marks a
    # group; a vertical rule goes at each group's left edge (matching the HTML dividers).
    bars = set()
    if parser.head:
        col = 0
        for _text, a in parser.head[0]:
            cs = int(a.get("colspan", 1))
            if cs > 1 and col > 0:
                bars.add(col)
            col += cs
    colspec = "".join(("|" if index in bars else "") + ("l" if index == 0 else "r")
                      for index in range(ncols))
    lines = []
    carry = 0   # columns owned by rowspan cells from the previous header row
    for row in parser.head:
        cells = [""] * carry
        col = carry
        row_carry = 0
        for text, a in row:
            t = cell_tex(text)
            cs = int(a.get("colspan", 1))
            if int(a.get("rowspan", 1)) > 1:
                row_carry += cs
            cells.append(rf"\multicolumn{{{cs}}}{{c}}{{{t}}}" if cs > 1 else t)
            col += cs
        carry = row_carry
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for row in parser.body:
        if len(row) == 1 and int(row[0][1].get("colspan", 1)) > 1:
            if lines and lines[-1] != r"\midrule":
                lines.append(r"\midrule")   # rule above each block (open models / this work)
            lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{{cell_tex(row[0][0])}}}}} \\")
            lines.append(r"\midrule")
            continue
        lines.append(" & ".join(cell_tex(t) for t, _ in row) + r" \\")
    body = "\n".join(lines)
    caption = prose(parser.caption)
    wide = ncols >= 8
    # Prose-heavy tables (long text cells, too few columns for the resizebox path) get
    # tabularx with wrapping X columns so they hold the text width instead of overflowing.
    col_max = [0] * ncols
    for row in parser.body:
        if len(row) == 1 and int(row[0][1].get("colspan", 1)) > 1:
            continue
        for index, (text, _a) in enumerate(row):
            if index < ncols:
                col_max[index] = max(col_max[index], len(re.sub(r"<[^>]+>", "", text)))
    prose_wrap = not wide and sum(col_max) > 110
    open_size = (r"\scriptsize\setlength{\tabcolsep}{2.6pt}" if wide
                 else r"\footnotesize\setlength{\tabcolsep}{3pt}" if prose_wrap
                 else r"\footnotesize")
    rule_fix = ("\\setlength{\\aboverulesep}{0pt}\\setlength{\\belowrulesep}{0pt}"
                if bars else "")
    if prose_wrap:
        colspec = "".join(r">{\raggedright\arraybackslash}X" if width > 30 else "l"
                          for width in col_max)
        tabular = (f"\\begin{{tabularx}}{{\\textwidth}}{{{colspec}}}\n\\toprule\n{body}\n"
                   f"\\bottomrule\n\\end{{tabularx}}")
    else:
        tabular = (f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{body}\n\\bottomrule\n"
                   f"\\end{{tabular}}")
    if wide:
        tabular = f"\\resizebox{{\\textwidth}}{{!}}{{%\n{tabular}}}"
    return (f"\\begin{{table}}[htbp]\n\\caption{{{caption}}}\n\\label{{tab:{key}}}\n"
            f"\\centering\n{open_size}{rule_fix}\n{tabular}\n\\end{{table}}\n")


def expand(body: str, figures: dict, tables: dict) -> str:
    blocks = {}

    def keep(latex: str) -> str:
        key = f"\x10{len(blocks)}\x10"
        blocks[key] = latex
        return key

    def fig(match):
        key, _, caption = match.group(1).partition("|")
        path = figures.get(key.strip())
        if not path or not Path(path).exists():
            return ""
        # The pipeline figure is 16:9 and full width costs ~3.1in of a page it does not need;
        # the wide layer-profile figure is already short enough to leave alone.
        width = {"pipeline": "0.72\\linewidth", "layers": "0.50\\linewidth",
                 "losses": "0.72\\linewidth"}.get(key.strip(), "\\linewidth")
        return keep(f"\\begin{{figure}}[htbp]\n\\centering\n"
                    f"\\includegraphics[width={width}]{{{path}}}\n"
                    f"\\caption{{{prose(caption.strip())}}}\n\\label{{fig:{key.strip()}}}\n\\end{{figure}}\n")

    def tab(match):
        key = match.group(1).strip()
        builder = tables.get(key)
        if builder is None:
            return ""
        html_out = builder() if callable(builder) else builder
        return keep(table_tex(html_out, key))

    body = re.sub(r"\{\{figure:([^}]+)\}\}", fig, body)
    body = re.sub(r"\{\{table:([^}]+)\}\}", tab, body)
    body = re.sub(r"<table.*?</table>", lambda m: keep(table_tex(m.group(0), "inline")), body, flags=re.S)
    out = prose(body)
    for key, latex in blocks.items():
        out = out.replace(key, latex)
    return out


def main() -> None:
    paper = content.PAPER
    out = []
    out.append(r"""\PassOptionsToPackage{numbers}{natbib}
\documentclass{article}
\usepackage{neurips_2026}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{hyperref}
""")
    out.append(f"\\title{{{esc(paper['title'])}}}\n")
    out.append(r"""\author{Anonymous}
\begin{document}
\maketitle
""")
    out.append(f"\\begin{{abstract}}\n{prose(paper['abstract'])}\n\\end{{abstract}}\n")
    for section in paper["sections"]:
        out.append(f"\\section{{{esc(section['title'])}}}\n")
        out.append(expand(section["body"], paper["figures"], paper["tables"]))
    # references (exempt from the page budget)
    out.append("\\clearpage\n\\begin{thebibliography}{99}\n\\footnotesize\n")
    for number, ref in enumerate(paper["references"], start=1):
        out.append(f"\\bibitem{{ref{number}}} {esc(ref)}\n")
    out.append("\\end{thebibliography}\n")
    out.append("\\appendix\n")
    for i, section in enumerate(paper["supplementary"]):
        out.append(f"\\section{{{esc(section['title'])}}}\n")
        out.append(expand(section["body"], paper["figures"], paper["tables"]))
    # checklist last, after the appendices, so appendix floats cannot drift into it
    out.append("\\clearpage\n\\input{checklist_filled}\n")
    out.append("\\end{document}\n")
    Path("neurips/main.tex").parent.mkdir(exist_ok=True)
    Path("neurips/main.tex").write_text("".join(out))
    print("wrote neurips/main.tex")


if __name__ == "__main__":
    main()
