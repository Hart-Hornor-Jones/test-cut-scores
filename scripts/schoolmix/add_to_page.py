"""Add the school-mixture estimates to index.html as a selectable test,
'CAASPP Math (school mixture)', leaving the default (current) estimates and
everything else on the page unchanged.  Re-runnable: replaces the key if it
is already there.

Usage: python add_to_page.py
"""
from __future__ import annotations

import json
import os
import re
import numpy as np
import pandas as pd

from common import DATA, REPO

KEY = "CAASPP Math (school mixture)"


def build_entry():
    sm = pd.read_csv(os.path.join(DATA, "estimated_share_above_cut_schoolmix.csv"), encoding="utf-8")
    out = {}
    for year, dy in sm.groupby("year"):
        grid = sorted(dy.score.unique())
        allrow = dy[dy.group_type == "All"].sort_values("score")
        entry = {}
        for scheme, gtype in [("race", "Race/ethnicity"), ("ses", "Economic status")]:
            groups = {}
            for gname, dg in dy[dy.group_type == gtype].groupby("group"):
                dg = dg.sort_values("score")
                assert list(dg.score) == grid
                groups[gname] = {"n": int(dg.n.iloc[0]), "tail": [round(float(v), 5) for v in dg.share_at_or_above]}
            entry[scheme] = {"grid": [int(v) for v in grid], "all_n": int(allrow.n.iloc[0]),
                             "all_tail": [round(float(v), 5) for v in allrow.share_at_or_above], "groups": groups}
        out[str(year)] = entry
    return out


def main():
    path = os.path.join(REPO, "index.html")
    with open(path, encoding="utf-8") as f:
        html = f.read()
    m = re.search(r"^const DATA=(\{.*\});$", html, flags=re.M)
    assert m, "DATA literal not found"
    data = json.loads(m.group(1))
    existing = {k: v for k, v in data.items() if k != KEY}
    # keep the existing literal text byte-for-byte; append the new key before the closing brace
    orig = m.group(1)
    if KEY in data:
        # rebuild without the key, then append (the existing part is re-serialized only in this case)
        base = json.dumps(existing, separators=(",", ":"), ensure_ascii=False)
    else:
        base = orig
    new_json = json.dumps(build_entry(), separators=(",", ":"), ensure_ascii=False)
    literal = base[:-1] + "," + json.dumps(KEY) + ":" + new_json + "}"
    html = html[:m.start(1)] + literal + html[m.end(1):]
    # gender grouping is not available for the school-mixture estimate: disable it like SES for the SAT
    old_line = '''  const so=$("scheme").querySelector('option[value="ses"]');so.disabled=state.test.startsWith("SAT");if(so.disabled&&state.scheme==="ses"){state.scheme="race";$("scheme").value="race"}'''
    new_line = old_line + '''
  const go=$("scheme").querySelector('option[value="gender"]');go.disabled=!DATA[state.test][state.year].gender;if(go.disabled&&state.scheme==="gender"){state.scheme="race";$("scheme").value="race"}'''
    if new_line not in html:
        assert old_line in html, "scheme line not found"
        html = html.replace(old_line, new_line)
    old_label = '''o.textContent=t==="SAT Total"?"SAT (total score)":t.replace("CAASPP","CAASPP grade 11")'''
    new_label = '''o.textContent=t==="SAT Total"?"SAT (total score)":t===%s?"CAASPP grade 11 Math — school-mixture estimate":t.replace("CAASPP","CAASPP grade 11")''' % json.dumps(KEY)
    if new_label not in html:
        assert old_label in html
        html = html.replace(old_label, new_label)
    note = '''<p>"CAASPP grade 11 Math — school-mixture estimate" is an alternative estimate of the same math distributions, built from school-level results (one near-normal component per school and student group) and reconciled to the same published statewide figures. It is not the default. It has no gender grouping. Its bounds are much narrower in the upper tail; see <code>METHOD.md</code>, section 11, and <code>data/estimated_*_schoolmix.csv</code>.</p>'''
    if note not in html:
        anchor = '''<p>Source tables: CDE Smarter Balanced technical reports'''
        assert anchor in html
        html = html.replace(anchor, note + "\n" + anchor)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print("index.html updated; DATA keys:", list(json.loads(re.search(r"^const DATA=(\{.*\});$", html, flags=re.M).group(1)).keys()))


if __name__ == "__main__":
    main()
