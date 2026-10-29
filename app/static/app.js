"use strict";
const $ = (id) => document.getElementById(id);

/* build DOM safely: text only, never innerHTML with server data */
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "style") n.setAttribute("style", v);
    else n.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) n.append(k.nodeType ? k : document.createTextNode(k));
  return n;
}
const pct = (p) => (p * 100).toFixed(p < 0.1 ? 1 : 0) + "%";
const fmt = (n) => Number(n).toLocaleString("en-IN", { maximumFractionDigits: 2 });

/* ---------- tabs ---------- */
const tabs = ["leaf", "yield", "report"];
function showTab(name) {
  tabs.forEach((t) => {
    $("t-" + t).setAttribute("aria-selected", t === name);
    $("p-" + t).hidden = t !== name;
  });
  if (name === "report") loadReport();
  if (name === "yield") initYield();
}
tabs.forEach((t) => $("t-" + t).addEventListener("click", () => showTab(t)));

/* ---------- leaf check ---------- */
let chosen = null;
const drop = $("drop");
function setFile(f) {
  $("err-leaf").hidden = true;
  if (!f) return;
  if (!/^image\/(jpeg|png)$/.test(f.type)) return showErr("err-leaf", "Use a JPG or PNG photo.");
  if (f.size > 6 * 1024 * 1024) return showErr("err-leaf", "That image is larger than 6 MB. Use a smaller photo.");
  chosen = f;
  const url = URL.createObjectURL(f);
  $("preview").src = url;
  $("preview").hidden = false;
  $("dropText").hidden = true;
  $("go-leaf").disabled = false;
  $("clear-leaf").hidden = false;
}
function showErr(id, msg) { const e = $(id); e.textContent = msg; e.hidden = false; }
$("file").addEventListener("change", (e) => setFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
$("clear-leaf").addEventListener("click", () => {
  chosen = null; $("file").value = ""; $("preview").hidden = true; $("dropText").hidden = false;
  $("go-leaf").disabled = true; $("clear-leaf").hidden = true; $("err-leaf").hidden = true;
  $("label").replaceChildren(el("p", { class: "empty" }, "Add a leaf photo and press Check leaf. The diagnosis appears here."));
});

$("go-leaf").addEventListener("click", async () => {
  if (!chosen) return;
  $("go-leaf").disabled = true; $("go-leaf").textContent = "Checking…"; $("err-leaf").hidden = true;
  try {
    const fd = new FormData(); fd.append("image", chosen);
    const r = await fetch("/api/disease", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Something went wrong. Try again.");
    renderLeaf(j);
  } catch (e) {
    showErr("err-leaf", e.message);
  } finally {
    $("go-leaf").disabled = false; $("go-leaf").textContent = "Check leaf";
  }
});

function ruler(conf, thr) {
  const t = el("div", { class: "track" },
    el("div", { class: "fill" + (conf < thr ? " low" : ""), style: `width:${(conf * 100).toFixed(1)}%` }),
    el("div", { class: "tick", style: `left:${thr * 100}%`, title: "Minimum confidence to give a diagnosis" }));
  return el("div", { class: "ruler" }, t,
    el("div", { class: "ends" }, el("span", {}, "0%"), el("span", {}, "100%")),
    el("p", { class: "fine" }, `The black line marks ${Math.round(thr * 100)}%, the minimum confidence for a diagnosis.`));
}

function renderLeaf(j) {
  const p = j.prediction, ok = j.confident;
  const chipClass = !ok ? "warn" : p.healthy ? "healthy" : "sick";
  const chipText = !ok ? "Not sure" : p.kind_label;
  const body = el("div", { class: "body" });
  body.append(el("h3", {}, `Confidence: ${pct(p.confidence)}`), ruler(p.confidence, j.threshold));

  if (!ok) {
    body.append(el("p", { class: "note" }, "The model is not confident enough to call this one. Retake the photo: one leaf, plain background, even daylight, leaf filling the frame."));
  } else {
    if (p.summary) body.append(el("h3", {}, "What it looks like"), el("p", {}, p.summary));
    if (p.actions.length) {
      body.append(el("h3", {}, p.healthy ? "Keep it that way" : "What to do"), el("ul", {}, ...p.actions.map((a) => el("li", {}, a))));
    }
  }
  body.append(el("h3", {}, "Other possibilities"),
    el("ul", { class: "alts" }, ...j.top3.map((t) => el("li", {},
      el("div", {}, `${t.crop}: ${t.condition}`, el("div", { class: "bar" }, el("i", { style: `width:${(t.probability * 100).toFixed(1)}%` }))),
      el("span", { class: "pct" }, pct(t.probability))))));
  body.append(el("p", { class: "fine" }, "General guidance only. Confirm with your local agriculture office (for example a Krishi Vigyan Kendra) before spraying. Trained on lab-style photos, so field photos may be less accurate."));

  $("label").replaceChildren(
    el("div", { class: "head" },
      el("p", { class: "crop" }, ok ? p.crop : "Best guess: " + p.crop),
      el("p", { class: "cond" }, ok ? p.condition : p.condition),
      el("span", { class: "chip " + chipClass }, chipText)),
    body);
}

/* ---------- yield ---------- */
let META = null;
async function initYield() {
  if (META) return;
  try {
    const r = await fetch("/api/meta");
    META = await r.json();
    if (!r.ok) throw new Error(META.error);
  } catch (e) { return showErr("err-yield", e.message || "Could not load options."); }
  const fill = (id, items, sel) => $(id).replaceChildren(...items.map((v) => { const o = el("option", { value: v }, v); if (v === sel) o.selected = true; return o; }));
  fill("f-crop", META.crops, "Rice");
  fill("f-state", META.states, "Uttar Pradesh");
  fill("f-season", META.seasons, "Kharif");
  $("f-year").value = new Date().getFullYear();
  applyDefaults();
  $("f-crop").addEventListener("change", applyDefaults);
  $("f-state").addEventListener("change", () => { const r = META.state_rain[$("f-state").value]; if (r) $("f-rain").value = Math.round(r); });
}
function applyDefaults() {
  const d = META.defaults.find((x) => x.Crop === $("f-crop").value);
  if (!d) return;
  $("f-fert").value = Math.round(d.fert * 10) / 10;
  $("f-pest").value = Math.round(d.pest * 100) / 100;
  $("f-area").value = 1;
  const sr = META.state_rain[$("f-state").value];
  $("f-rain").value = Math.round(sr || d.rain);
}

$("yform").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("err-yield").hidden = true;
  const body = {
    Crop: $("f-crop").value, State: $("f-state").value, Season: $("f-season").value,
    Crop_Year: $("f-year").value, Area: $("f-area").value, Annual_Rainfall: $("f-rain").value,
    Fert_per_ha: $("f-fert").value, Pest_per_ha: $("f-pest").value,
  };
  $("go-yield").disabled = true; $("go-yield").textContent = "Estimating…";
  try {
    const r = await fetch("/api/yield", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Something went wrong. Try again.");
    renderYield(j, body);
  } catch (err) { showErr("err-yield", err.message); }
  finally { $("go-yield").disabled = false; $("go-yield").textContent = "Estimate yield"; }
});

function renderYield(j, body) {
  const b = el("div", { class: "body" });
  b.append(el("h3", {}, "Expected production"),
    el("p", {}, el("b", {}, `${fmt(j.production)} ${j.production_unit}`), ` from ${fmt(body.Area)} ha`));
  if (j.typical_low != null) {
    const lo = j.typical_low, hi = j.typical_high, v = j.yield_per_ha;
    const max = Math.max(hi, v) * 1.1, x = (n) => Math.min(100, (n / max) * 100).toFixed(1);
    b.append(el("h3", {}, `Compared with other ${body.Crop} records`),
      el("div", { class: "range" },
        el("div", { class: "rail" }),
        el("div", { class: "band", style: `left:${x(lo)}%;width:${(x(hi) - x(lo)).toFixed(1)}%` }),
        el("div", { class: "pin", style: `left:${x(v)}%`, title: "This estimate" })),
      el("p", { class: "fine" }, `Green band: the middle 80% of historical ${body.Crop} yields (${fmt(lo)} to ${fmt(hi)}); median ${fmt(j.typical_median)}. Black pin: this estimate.`));
  }
  j.notes.forEach((n) => b.append(el("p", { class: "note" }, n)));
  b.append(el("p", { class: "fine" }, "This is a statistical estimate from past state-level records, not a guarantee. Weather in the season, pests and soil can move the real result. On held-out future years the typical error was about 16%."));
  $("ylabel").replaceChildren(
    el("div", { class: "head" },
      el("p", { class: "crop" }, `${body.Crop} · ${body.State} · ${body.Season} ${body.Crop_Year}`),
      el("p", { class: "cond" }, fmt(j.yield_per_ha), " ", el("small", { style: "font:600 1rem var(--sans);color:var(--ink-2)" }, j.unit))),
    b);
}

/* ---------- report ---------- */
let reportLoaded = false;
async function loadReport() {
  if (reportLoaded) return;
  const box = $("report-body");
  try {
    const j = await (await fetch("/api/report")).json();
    const d = j.disease, y = j.yield_;
    const out = [];
    if (d) {
      out.push(el("h2", {}, "Leaf disease model"),
        el("div", { class: "stats" },
          stat(pct(d.test_accuracy), "accuracy on unseen test images"),
          stat(pct(d.test_macro_f1), "macro F1 (all classes equal)"),
          stat(String(d.n_classes), "crop and disease classes"),
          stat(fmt(d.n_images), "images used")),
        el("p", { class: "fine" }, `Split: ${fmt(d.split.train)} train, ${fmt(d.split.val)} validation, ${fmt(d.split.test)} test. The test set was used once, after choosing the model on validation.`));
      const rows = Object.entries(d.validation_accuracy).map(([k, v]) =>
        el("tr", { class: k === d.chosen_model ? "pick" : "" }, el("td", {}, k + (k === d.chosen_model ? " (chosen)" : "")), el("td", { class: "n" }, pct(v))));
      out.push(el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Candidate"), el("th", { class: "n" }, "Validation accuracy"))), el("tbody", {}, ...rows)));
      const tr = Object.entries(d.confidence_threshold_tradeoff).map(([k, v]) =>
        el("tr", {}, el("td", {}, `${Math.round(k * 100)}% or higher`), el("td", { class: "n" }, pct(v.coverage)), el("td", { class: "n" }, v.accuracy_when_answered == null ? "–" : pct(v.accuracy_when_answered))));
      out.push(el("h2", {}, "Confidence versus accuracy"),
        el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Only answer when confidence is"), el("th", { class: "n" }, "Photos answered"), el("th", { class: "n" }, "Accuracy of those"))), el("tbody", {}, ...tr)),
        el("details", {}, el("summary", {}, "Confusion matrix"), el("img", { src: "/reports/disease_confusion_matrix.png", alt: "Confusion matrix on the test set" })));
    }
    if (y) {
      out.push(el("h2", {}, "Yield model"),
        el("p", { class: "fine" }, `${fmt(y.dataset.rows_used)} records, ${y.dataset.crops} crops, ${y.dataset.states} states, ${y.dataset.years[0]}–${y.dataset.years[1]}.`));
      const rows = y.random_split.map((m) => el("tr", { class: m.model.includes("chosen") ? "pick" : "" },
        el("td", {}, m.model), el("td", { class: "n" }, m.R2_log.toFixed(3)), el("td", { class: "n" }, m.R2_yield.toFixed(3)), el("td", { class: "n" }, m.median_abs_pct_error + "%")));
      const t = y.temporal_split[0];
      rows.push(el("tr", {}, el("td", {}, "Chosen model, trained to 2014, tested on 2015–2020"), el("td", { class: "n" }, t.R2_log.toFixed(3)), el("td", { class: "n" }, t.R2_yield.toFixed(3)), el("td", { class: "n" }, t.median_abs_pct_error + "%")));
      out.push(el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Model and test"), el("th", { class: "n" }, "R² (log yield)"), el("th", { class: "n" }, "R² (yield)"), el("th", { class: "n" }, "Typical error"))), el("tbody", {}, ...rows)),
        el("p", { class: "fine" }, "What drives the estimate most: " + y.permutation_importance.slice(0, 4).map((p) => p[0].replace("_per_ha", " per ha").replace("Annual_", "").replace("Crop_Year", "Year")).join(", ") + "."));
    }
    out.push(el("h2", {}, "Limits to know about"),
      el("ul", { class: "limits" },
        el("li", {}, "Leaf photos in the training data are lab-style (one leaf, plain background). Field photos with soil, hands or several leaves will be less accurate. The app says “Not sure” when confidence is low."),
        el("li", {}, "The same physical leaf can appear in more than one image in this dataset. Random splits can therefore flatter the accuracy; treat the number as an upper bound."),
        el("li", {}, "The disease model uses hand-crafted colour, texture and edge features with a scikit-learn classifier. A deep network such as MobileNet is the natural next step."),
        el("li", {}, "The yield model learns from state-level averages, so it describes typical outcomes for a crop, state and season rather than one particular farm. Rainfall, fertiliser and pesticide add little beyond crop and state in this data."),
        el("li", {}, "A photo check refuses blank, flat-colour, noisy or plant-free images. It is a sanity check, not a leaf detector: a colourful photo of something else can still pass and get a label."),
        el("li", {}, "The yield model and the disease model are separate: a detected disease does not change the yield estimate.")));
    box.replaceChildren(...out);
    // let wide tables scroll sideways on a phone instead of overflowing the page
    box.querySelectorAll("table").forEach((t) => { const w = el("div", { class: "tablewrap" }); t.replaceWith(w); w.append(t); });
    reportLoaded = true;
  } catch (e) { box.replaceChildren(el("p", { class: "err" }, "Could not load results. Train the models first.")); }
}
function stat(v, l) { return el("div", { class: "stat" }, el("b", {}, v), el("span", {}, l)); }
