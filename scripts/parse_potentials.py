"""Turn the downloaded wikitext of the standard electrode potential data page
into data/standard_potentials.csv.

Run with:  .venv/bin/python -I scripts/parse_potentials.py data/raw/potentials.wikitext data/standard_potentials.csv

The page lists each half-reaction as one table row: element, oxidant side,
the arrow, reductant side, E0 in volts vs SHE, electrons, reference.
Wiki templates are reduced to plain chemical text. The reference column is
kept as the reference name used on the page so every value can be traced.
"""

import csv
import re
import sys

SOURCE = "https://en.wikipedia.org/wiki/Standard_electrode_potential_(data_page)"


def clean(cell: str) -> str:
    s = cell
    s = re.sub(r"_\{([^{}]*)\}", r"\1", s)
    s = s.replace("&alpha;", "alpha-")
    s = s.replace("&nbsp;", " ").replace("&mdash;", "-").replace("'''", "")
    s = re.sub(r"<sup>\s*[–-]\s*</sup>", "-", s)
    s = re.sub(r"<sup>([^<]*)</sup>", r"^\1", s)
    s = re.sub(r"<sub>([^<]*)</sub>", r"\1", s)
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\{\{H\+[^}]*\}\}", "H+", s)
    s = re.sub(r"\{\{OH-[^}]*\}\}", "OH-", s)
    s = re.sub(r"\{\{H2O[^}]*\}\}", "H2O(l)", s)
    s = re.sub(r"\{\{CO2[^}]*\}\}", "CO2", s)
    s = re.sub(r"\{\{H2[^}]*\}\}", "H2", s)
    # {{chem2|link=X|FORMULA}} or {{chem2|FORMULA}}
    s = re.sub(r"\{\{chem2\|(?:link=[^|}]*\|)?([^}]*)\}\}", r"\1", s)
    s = re.sub(r"\{\{rp\|[^}]*\}\}", "", s)
    s = re.sub(r"\{\{[^}]*\}\}", "", s)
    s = re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = s.replace("_{", "").replace("}", "").replace("{", "")
    s = s.replace("–", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ref_name(cell: str) -> str:
    names = re.findall(r'<ref name\s*=\s*"?([^"/>]+?)"?\s*/?>', cell)
    pages = re.findall(r"\{\{rp\|([^}]*)\}\}", cell)
    out = ";".join(n.strip() for n in names)
    if pages:
        out += " p." + ",".join(pages)
    if not out and "<ref" in cell:
        out = "inline citation"
    return out


def number(cell: str):
    s = clean(cell).replace("+", "")
    m = re.match(r"^-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else None


def rows(text: str):
    start = text.index("==Table of standard electrode potentials==")
    body = text[start:]
    end = body.find("\n|}")
    body = body[: end if end > 0 else len(body)]
    for chunk in body.split("\n|-")[1:]:
        cells = [c for c in re.split(r"\n\|(?!\})", "\n" + chunk.strip("\n")) if c.strip()]
        cells = [re.sub(r"^\s*class\s*=\s*\"?noprint\"?\s*\|", "", c) for c in cells]
        if len(cells) < 7 or cells[0].lstrip().startswith("!"):
            continue
        yield cells


def main(src: str, dst: str) -> None:
    text = open(src, encoding="utf-8").read()
    out = []
    for cells in rows(text):
        element, ox, arrow, red, e0, n, ref = cells[:7]
        if "⇌" not in arrow:
            continue
        e = number(e0)
        if e is None:
            continue
        nv = number(n)
        out.append(
            {
                "element": clean(element),
                "oxidant": clean(ox),
                "reductant": clean(red),
                "e0_v_she": e,
                "electrons": int(nv) if nv is not None else "",
                "reference": ref_name(ref),
                "source": SOURCE,
            }
        )
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} half-reactions -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
