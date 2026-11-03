#!/usr/bin/env python3
"""Labels the camera shots in data/ham item by item, in the browser.

The app saves a crop every half second while an item is in view, so the
shots come in runs: one run = one item. This groups the runs, writes a
single self-contained HTML page with the shots of every run, and turns the
labels exported from that page into kendi/<bin>/ for prepare_dataset.py.
A run is labelled as a whole, or shot by shot when another item slipped in:

    python3 label_page.py build --ham ../data/ham --out ../data/etiketle.html
    (open the page, label, "Etiketleri indir" -> etiketler.json)
    python3 label_page.py apply --ham ../data/ham --labels ~/İndirilenler/etiketler.json --out kendi

apply holds out a fifth of each bin's items (whole runs, binned by the
majority of their shots) in kendi_test/, so the camera accuracy is measured
on items the model never saw.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import random
import shutil
import sys

from PIL import Image

BINS = ["plastik", "organik", "kagit", "cam", "metal", "diger"]
SKIP = "atla"


def shots(ham):
    """(time, file, model label, COCO name) for every shot, oldest first."""
    out = []
    for f in os.listdir(ham):
        if not f.lower().endswith(".jpg"):
            continue
        parts = f[:-4].split("_", 2)
        if len(parts) != 3:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        out.append((t, f, parts[1], parts[2]))
    return sorted(out)


def runs(items, gap=1.5, longest=40):
    """Splits at pauses longer than `gap` s; long runs are cut into pieces
    of `longest` shots, since the item may have been swapped without a pause."""
    groups = []
    for it in items:
        if groups and it[0] - groups[-1][-1][0] <= gap and len(groups[-1]) < longest:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def thumb(path, size=150):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def tally(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return ", ".join(f"{k} {n}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def build(args):
    groups = runs(shots(args.ham), args.gap, args.longest)
    data = []
    for i, g in enumerate(groups):
        k = min(len(g), args.thumbs or len(g))
        picks = [g[round(j * (len(g) - 1) / max(1, k - 1))] for j in range(k)]
        data.append({
            "id": f"g{i:04d}",
            "files": [it[1] for it in g],
            "thumbs": [{"f": it[1], "src": thumb(os.path.join(args.ham, it[1]))} for it in picks],
            "model": tally(it[2] for it in g),
            "coco": tally(it[3].replace("-", " ") for it in g),
        })
    # Pages from different shot sets must not share saved labels: group ids
    # restart at g0000 in every page, and file:// pages share one storage.
    key = "atik-etiket-" + hashlib.sha1("".join(d["files"][0] for d in data).encode()).hexdigest()[:12]
    page = (PAGE.replace("/*DATA*/null", json.dumps(data)).replace("/*BINS*/null", json.dumps(BINS))
            .replace('"atik-etiket-v1"', json.dumps(key)))
    with open(args.out, "w") as f:
        f.write(page)
    n = sum(len(g) for g in groups)
    print(f"{n} kare, {len(groups)} grup -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


def apply(args):
    """Copies labelled shots into <out>/<bin>/. A shot labelled on its own in
    the page overrides its group. With --test, that share of each bin's groups
    (whole runs, never single shots) goes to <out>_test/ instead."""
    with open(args.labels) as f:
        labels = json.load(f)
    groups = {f"g{i:04d}": [it[1] for it in g]
              for i, g in enumerate(runs(shots(args.ham), args.gap, args.longest))}
    # A group counts for the bin most of its shots carry: runs are often
    # labelled shot by shot, leaving the group itself on "atla".
    by_bin = {b: [] for b in BINS}
    for gid, names in sorted(groups.items()):
        votes = [labels["kareler"][n] for n in names if labels["kareler"].get(n) in BINS]
        if votes:
            by_bin[max(BINS, key=votes.count)].append(gid)
    rnd = random.Random(args.seed)
    held = set()
    for b, gids in by_bin.items():
        k = round(len(gids) * args.test)
        if args.test and len(gids) >= 2:
            k = max(1, k)
        held |= set(rnd.sample(gids, min(k, len(gids) - 1)) if len(gids) > 1 else [])
    missing = set(labels["kareler"]) - {n for names in groups.values() for n in names}
    if missing:
        sys.exit(f"{sorted(missing)[:3]} ham klasöründe yok: etiketler başka bir sayfadan mı?")
    counts = {}
    for gid, names in groups.items():
        out = args.out + "_test" if gid in held else args.out
        for name in names:
            b = labels["kareler"].get(name)
            if b not in BINS:
                continue
            d = os.path.join(out, b)
            os.makedirs(d, exist_ok=True)
            shutil.copy2(os.path.join(args.ham, name), d)
            counts[(out, b)] = counts.get((out, b), 0) + 1
    for out in sorted({o for o, _ in counts}):
        print(f"{out}/: " + "  ".join(f"{b} {counts.get((out, b), 0)}" for b in BINS))


PAGE = r"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atık Etiketleme (kare seçimli)</title>
<style>
:root { --bg: #fafafa; --card: #fff; --ink: #1a1a1a; --quiet: #666; --line: #ddd; --accent: #2a6fdb; --sel: #e8f0fd; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161616; --card: #202020; --ink: #eee; --quiet: #999; --line: #333; --accent: #6ea0ff; --sel: #1d2a40; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.4 system-ui, sans-serif; }
header { position: sticky; top: 0; z-index: 2; background: var(--bg); border-bottom: 1px solid var(--line);
         padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
header h1 { font-size: 16px; margin: 0 8px 0 0; }
#progress { color: var(--quiet); }
#counts { color: var(--quiet); font-size: 12px; flex: 1; }
button { font: inherit; border: 1px solid var(--line); background: var(--card); color: var(--ink);
         border-radius: 6px; padding: 5px 10px; cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
label.f { color: var(--quiet); font-size: 13px; }
main { padding: 12px 16px 80px; display: grid; gap: 10px; }
.g { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }
.g.cur { outline: 2px solid var(--accent); }
.g.done { opacity: .55; }
.g.done.cur { opacity: 1; }
.meta { display: flex; flex-wrap: wrap; gap: 4px 14px; color: var(--quiet); font-size: 12px; margin-bottom: 6px; }
.meta b { color: var(--ink); }
.th { display: flex; gap: 4px; overflow-x: auto; }
.th { flex-wrap: wrap; }
.th figure { margin: 0; position: relative; cursor: pointer; }
.th img { height: 96px; border-radius: 4px; display: block; border: 3px solid transparent; }
.th figure.sel img { border-color: var(--accent); outline: 3px solid var(--accent); opacity: .8; }
.th figure.sel::after { content: "✓"; position: absolute; top: 4px; right: 6px; background: var(--accent); color: #fff;
                        font-weight: 700; border-radius: 50%; width: 22px; height: 22px; text-align: center; line-height: 22px; }
#selinfo { color: var(--accent); font-weight: 600; }
.th figcaption { position: absolute; left: 3px; bottom: 3px; background: var(--accent); color: #fff;
                 font-size: 11px; padding: 0 4px; border-radius: 3px; }
.bins { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.bins button.on { background: var(--sel); border-color: var(--accent); font-weight: 600; }
.help { color: var(--quiet); font-size: 12px; padding: 0 16px; }
</style>
</head>
<body>
<header>
  <h1>Atık etiketleme</h1>
  <span id="progress"></span>
  <span id="counts"></span>
  <span id="selinfo"></span>
  <label class="f"><input type="checkbox" id="hide"> Etiketlileri gizle</label>
  <button class="primary" id="export">Etiketleri indir</button>
</header>
<p class="help">Her kart, kameranın art arda kaydettiği kareler, yani büyük olasılıkla tek bir nesne.
Kısayollar: <b>1</b> plastik · <b>2</b> organik · <b>3</b> kağıt · <b>4</b> cam · <b>5</b> metal · <b>6</b> diğer ·
<b>0</b> atla (boş, el, bulanık) · <b>↑/↓</b> önceki/sonraki. Grupta farklı bir nesne varsa o karelere tıklayıp
seçin, sonra numaraya basın: yalnızca seçili kareler etiketlenir (<b>Esc</b> seçimi bırakır). Etiketler bu tarayıcıda otomatik saklanır.
"model" uygulamanın o anki tahmini, "coco" dedektörün verdiği isim; ikisi de ipucu, doğru olmak zorunda değil.</p>
<main id="list"></main>
<script>
const DATA = /*DATA*/null;
const BINS = /*BINS*/null;
const NAMES = {plastik: "Plastik", organik: "Organik", kagit: "Kağıt/Karton", cam: "Cam", metal: "Metal", diger: "Diğer", atla: "Atla"};
const CHOICES = [...BINS, "atla"];
const KEY = "atik-etiket-v1";
let labels = {}, over = {};
try {
  const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
  // Older pages stored only {group id: bin}.
  if (saved.gruplar) { labels = saved.gruplar; over = saved.kareler || {}; } else labels = saved;
} catch (e) {}
let cur = 0, sel = new Set();
const list = document.getElementById("list");

function save() { try { localStorage.setItem(KEY, JSON.stringify({gruplar: labels, kareler: over})); } catch (e) {} }

function render() {
  list.innerHTML = "";
  DATA.forEach((g, i) => {
    const el = document.createElement("section");
    el.className = "g"; el.id = g.id; el.dataset.i = i;
    const imgs = g.thumbs.map(t => `<figure data-f="${t.f}"><img src="${t.src}" alt=""><figcaption hidden></figcaption></figure>`).join("");
    const btns = CHOICES.map((b, k) => `<button data-b="${b}">${b === "atla" ? 0 : k + 1} · ${NAMES[b]}</button>`).join("");
    el.innerHTML = `<div class="meta"><b>#${i + 1}</b><span>${g.files.length} kare</span>
      <span>model: ${g.model}</span><span>coco: ${g.coco}</span></div>
      <div class="th">${imgs}</div><div class="bins">${btns}</div>`;
    el.addEventListener("click", e => {
      if (cur !== i) sel.clear();
      cur = i;
      const b = e.target.closest("button"), fig = e.target.closest("figure");
      if (b) set(i, b.dataset.b);
      else { if (fig) sel.has(fig.dataset.f) ? sel.delete(fig.dataset.f) : sel.add(fig.dataset.f); update(); }
    });
    list.appendChild(el);
  });
  update();
}

function set(i, b) {
  if (sel.size) {
    // Only the picked shots: another item in the same run.
    sel.forEach(f => { if (b === labels[DATA[i].id]) delete over[f]; else over[f] = b; });
    sel.clear(); save(); update(); return;
  }
  labels[DATA[i].id] = b; save();
  const next = DATA.findIndex((g, j) => j > i && !labels[g.id]);
  cur = next >= 0 ? next : Math.min(i + 1, DATA.length - 1);
  update(true);
}

function update(scroll) {
  const hide = document.getElementById("hide").checked;
  const counts = {};
  DATA.forEach((g, i) => {
    const el = list.children[i], l = labels[g.id];
    el.classList.toggle("done", !!l);
    el.classList.toggle("cur", i === cur);
    el.style.display = hide && l && i !== cur ? "none" : "";
    el.querySelectorAll(".bins button").forEach(b => b.classList.toggle("on", b.dataset.b === l));
    el.querySelectorAll("figure").forEach(fig => {
      const o = over[fig.dataset.f], cap = fig.querySelector("figcaption");
      fig.classList.toggle("sel", i === cur && sel.has(fig.dataset.f));
      cap.hidden = !o; cap.textContent = o ? NAMES[o] : "";
    });
    g.files.forEach(f => { const k = over[f] || l; if (k) counts[k] = (counts[k] || 0) + 1; });
  });
  document.getElementById("selinfo").textContent =
    sel.size ? `#${cur + 1}: ${sel.size} kare seçili, numaraya basın` : "";
  const done = DATA.filter(g => labels[g.id]).length;
  document.getElementById("progress").textContent = `${done} / ${DATA.length} grup`;
  document.getElementById("counts").textContent =
    CHOICES.filter(b => counts[b]).map(b => `${NAMES[b]} ${counts[b]}`).join(" · ") + (done ? " kare" : "");
  if (scroll) list.children[cur].scrollIntoView({block: "center", behavior: "smooth"});
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key >= "1" && e.key <= String(BINS.length)) { set(cur, BINS[+e.key - 1]); e.preventDefault(); }
  else if (e.key === "0") { set(cur, "atla"); e.preventDefault(); }
  else if (e.key === "Escape") { sel.clear(); update(); }
  else if (e.key === "ArrowDown" || e.key === "j") { sel.clear(); cur = Math.min(cur + 1, DATA.length - 1); update(true); e.preventDefault(); }
  else if (e.key === "ArrowUp" || e.key === "k") { sel.clear(); cur = Math.max(cur - 1, 0); update(true); e.preventDefault(); }
});
document.getElementById("hide").addEventListener("change", () => update());

document.getElementById("export").addEventListener("click", () => {
  const kareler = {};
  DATA.forEach(g => g.files.forEach(f => { const l = over[f] || labels[g.id]; if (l) kareler[f] = l; }));
  const blob = new Blob([JSON.stringify({gruplar: labels, kareler}, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "etiketler.json"; a.click();
  URL.revokeObjectURL(a.href);
});

render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="data/ham -> etiketleme sayfası")
    b.add_argument("--ham", default="../data/ham")
    b.add_argument("--out", default="../data/etiketle.html")
    b.add_argument("--gap", type=float, default=1.5, help="bundan uzun ara yeni nesne sayılır (sn)")
    b.add_argument("--longest", type=int, default=40, help="bir gruptaki en fazla kare")
    b.add_argument("--thumbs", type=int, default=0, help="grup başına küçük resim (0 = hepsi)")
    a = sub.add_parser("apply", help="etiketler.json -> kendi/<kutu>/")
    a.add_argument("--ham", default="../data/ham")
    a.add_argument("--labels", required=True)
    a.add_argument("--out", default="kendi")
    a.add_argument("--test", type=float, default=0.2, help="test için ayrılan grup oranı (sınıf başına)")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--gap", type=float, default=1.5)
    a.add_argument("--longest", type=int, default=40)
    args = ap.parse_args()
    build(args) if args.cmd == "build" else apply(args)


if __name__ == "__main__":
    main()
