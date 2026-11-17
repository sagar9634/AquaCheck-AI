/* AquaCheckAI front end: builds the sliders from /api/meta, calls /api/predict, runs the AquaBot chat. */
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => (Math.abs(n) >= 1000 ? Math.round(n).toLocaleString() : Number(n.toFixed(2)).toString());

let META = null;
let lastResult = null;
const values = {};
const history = [];
const API_BASE = "https://aquacheck-ai-backend.onrender.com";
/* ---------- live glass (visual only) ---------- */
const PH_STOPS = [[3, "#e0463a"], [5, "#f08a3c"], [6.2, "#e9d04a"], [7, "#8fcf6a"], [8, "#4fbfb0"], [9.5, "#3d8fd1"], [11, "#7a5bc7"]];
const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
function phColor(ph) {
  const p = Math.min(11, Math.max(3, ph));
  for (let i = 1; i < PH_STOPS.length; i++) {
    const [p1, c1] = PH_STOPS[i - 1], [p2, c2] = PH_STOPS[i];
    if (p <= p2) {
      const t = (p - p1) / (p2 - p1), a = hex2rgb(c1), b = hex2rgb(c2);
      return "rgb(" + a.map((v, k) => Math.round(v + (b[k] - v) * t)).join(",") + ")";
    }
  }
  return PH_STOPS[PH_STOPS.length - 1][1];
}
function updateGlass() {
  if (!META || values.ph === undefined || values.Turbidity === undefined) return;
  const ph = values.ph, turb = values.Turbidity;
  const col = phColor(ph);
  ["water-body", "wave"].forEach((id) => $("#" + id).setAttribute("fill", col));
  const t = META.features.find((f) => f.key === "Turbidity");
  const cloud = Math.min(1, Math.max(0, (turb - t.min) / (t.max - t.min)));
  $("#cloud").setAttribute("opacity", (cloud * 0.6).toFixed(2));
  const pos = Math.min(11, Math.max(3, ph));
  $("#strip-marker").style.left = ((pos - 3) / 8) * 100 + "%";
  $("#glass-note").textContent = `pH ${fmt(ph)} tints the water. Turbidity ${fmt(turb)} NTU makes it ${cloud > 0.66 ? "very cloudy" : cloud > 0.33 ? "slightly cloudy" : "clear"}.`;
}

/* ---------- sliders ---------- */
function setZone(range, f) {
  const span = f.max - f.min;
  const pct = (v) => Math.min(100, Math.max(0, ((v - f.min) / span) * 100)) + "%";
  range.style.setProperty("--lo", pct(f.safe_low));
  range.style.setProperty("--hi", pct(f.safe_high));
}
function buildSliders() {
  const form = $("#form");
  form.innerHTML = "";
  META.features.forEach((f) => {
    const id = "f-" + f.key;
    const step = (f.max - f.min) / 400;
    const wrap = document.createElement("div");
    wrap.className = "field";
    wrap.innerHTML = `
      <div class="field-top">
        <label for="${id}-n">${esc(f.label)}</label>
        <div class="num"><input id="${id}-n" type="number" step="any" inputmode="decimal" aria-label="${esc(f.label)} value"><span>${esc(f.unit)}</span></div>
      </div>
      <p class="hint">${esc(f.hint)}</p>
      <input id="${id}" type="range" min="${f.min}" max="${f.max}" step="${step}" aria-label="${esc(f.label)} slider">`;
    form.appendChild(wrap);
    const range = $("#" + id, wrap), num = $("#" + id + "-n", wrap);
    setZone(range, f);
    const set = (v, from) => {
      if (Number.isNaN(v)) return;
      values[f.key] = v;
      if (from !== "range") range.value = v;
      if (from !== "num") num.value = Number(v.toFixed(2));
      updateGlass();
    };
    range.addEventListener("input", () => set(parseFloat(range.value), "range"));
    num.addEventListener("input", () => set(parseFloat(num.value), "num"));
    wrap._set = set;
  });
}
function setAll(sample) {
  META.features.forEach((f, i) => {
    const wrap = $("#form").children[i];
    wrap._set(sample[f.key] !== undefined ? sample[f.key] : f.median);
  });
}
function buildPresets() {
  const box = $("#presets");
  box.innerHTML = "";
  Object.keys(META.presets).forEach((name) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = name;
    b.addEventListener("click", () => setAll(META.presets[name]));
    box.appendChild(b);
  });
}

/* ---------- prediction ---------- */
async function checkWater() {
  const btn = $("#check-btn");
  btn.disabled = true;
  btn.textContent = "Checking...";
  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ values }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "The server could not check this sample.");
    lastResult = data;
    showResult(data);
    $("#chat-dot").hidden = $("#chat").hidden ? false : true;
  } catch (err) {
    showError("Could not check the water: " + err.message + " Make sure the server is running (python app.py).");
  } finally {
    btn.disabled = false;
    btn.textContent = "Check water";
  }
}
function showResult(d) {
  const box = $("#result");
  box.hidden = false;
  box.className = "result " + (d.potable ? "safe" : "unsafe");
  box.style.animation = "none"; void box.offsetWidth; box.style.animation = "";
  $("#verdict-title").textContent = d.potable ? "Likely safe to drink" : "Likely not safe to drink";
  $("#verdict-sub").textContent = `The model gives this sample a ${d.probability}% chance of being potable.`;
  const C = 326.7;
  $("#ring-fg").style.strokeDashoffset = C;
  requestAnimationFrame(() => requestAnimationFrame(() => { $("#ring-fg").style.strokeDashoffset = C * (1 - d.probability / 100); }));
  animateNumber($("#pct"), d.probability);

  const max = Math.max(1, ...d.contributions.map((c) => Math.abs(c.impact)));
  $("#drivers").innerHTML = d.contributions.map((c) => {
    const up = c.impact >= 0, w = (Math.abs(c.impact) / max) * 50;
    return `<div class="driver">
      <div class="lbl">${esc(c.label)}<small>${esc(fmt(c.value))} ${esc(c.unit)}</small></div>
      <div class="track"><div class="bar ${up ? "up" : "down"}" data-w="${w}"></div></div>
      <div class="val ${up ? "up" : "down"}">${up ? "+" : ""}${c.impact.toFixed(1)}</div></div>`;
  }).join("");
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll("#drivers .bar").forEach((b) => (b.style.width = b.dataset.w + "%"));
  }));

  const words = { ok: "in range", low: "too low", high: "too high" };
  $("#chips").innerHTML = d.status.map((s) => `<li class="chip ${s.state}">${esc(s.label)} ${words[s.state]}</li>`).join("");
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function showError(msg) {
  const n = $("#load-error");
  n.textContent = msg;
  n.hidden = false;
}
function animateNumber(el, to) {
  const start = performance.now(), dur = 900;
  const tick = (now) => {
    const t = Math.min(1, (now - start) / dur);
    el.textContent = (to * (1 - Math.pow(1 - t, 3))).toFixed(1);
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ---------- model section ---------- */
function buildModel() {
  const m = META.metrics;
  $("#model-lede").textContent =
    `A ${META.best_model} trained from scratch on ${META.dataset_rows.toLocaleString()} water samples. About ${Math.round(META.potable_share * 100)}% of them were labelled safe to drink.`;
  const s = [["Accuracy", m.accuracy], ["Precision", m.precision], ["Recall", m.recall], ["F1 score", m.f1]];
  $("#scores").innerHTML = s.map(([k, v]) => `<div><dt>${k}</dt><dd>${(v * 100).toFixed(0)}%</dd></div>`).join("");

  const labels = Object.fromEntries(META.features.map((f) => [f.key, f.label]));
  const imp = Object.entries(META.importance).sort((a, b) => b[1] - a[1]);
  const top = imp[0][1];
  $("#importance").innerHTML = imp.map(([k, v]) =>
    `<div class="imp"><span>${esc(labels[k] || k)}</span><div class="t"><div class="f" style="width:${(v / top) * 100}%"></div></div><span>${(v * 100).toFixed(0)}%</span></div>`).join("");

  const [[tn, fp], [fn, tp]] = m.confusion_matrix;
  $("#matrix").innerHTML = `
    <span></span><span class="h">Predicted unsafe</span><span class="h">Predicted safe</span>
    <span class="h">Actually unsafe</span><div class="cell right">${tn}<small>correct</small></div><div class="cell wrong">${fp}<small>false alarm</small></div>
    <span class="h">Actually safe</span><div class="cell wrong">${fn}<small>missed</small></div><div class="cell right">${tp}<small>correct</small></div>`;
}

/* ---------- AquaBot chat ---------- */
function md(text) {
  const lines = esc(text).split("\n");
  let html = "", inList = false;
  const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|\W)_(.+?)_(?=\W|$)/g, "$1<em>$2</em>");
  for (const line of lines) {
    if (/^\s*-\s+/.test(line)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += "<li>" + inline(line.replace(/^\s*-\s+/, "")) + "</li>";
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (line.trim()) html += "<p>" + inline(line) + "</p>";
    }
  }
  return html + (inList ? "</ul>" : "");
}
function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = "msg " + (role === "user" ? "me" : "bot");
  el.innerHTML = role === "user" ? esc(text) : md(text);
  $("#msgs").appendChild(el);
  $("#msgs").scrollTop = $("#msgs").scrollHeight;
  return el;
}
function setSuggestions(list) {
  const box = $("#suggest");
  box.innerHTML = "";
  (list || []).forEach((q) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = q;
    b.addEventListener("click", () => sendMessage(q));
    box.appendChild(b);
  });
}
async function sendMessage(text) {
  text = (text || "").trim();
  if (!text) return;
  addMsg("user", text);
  $("#chat-input").value = "";
  setSuggestions([]);
  const typing = document.createElement("div");
  typing.className = "msg bot";
  typing.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
  $("#msgs").appendChild(typing);
  $("#msgs").scrollTop = $("#msgs").scrollHeight;
  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history: history.slice(-8), context: lastResult }),
    });
    const data = await res.json();
    typing.remove();
    addMsg("assistant", data.reply);
    history.push({ role: "user", content: text }, { role: "assistant", content: data.reply });
    $("#chat-mode").textContent = data.mode === "llm" ? "AI model" : "offline mode";
    setSuggestions(data.suggestions);
  } catch {
    typing.remove();
    addMsg("assistant", "I could not reach the server. Start it with `python app.py` and try again.");
  }
}
function toggleChat(open) {
  const chat = $("#chat");
  const show = open === undefined ? chat.hidden : open;
  chat.hidden = !show;
  $("#chat-fab").setAttribute("aria-expanded", String(show));
  if (show) { $("#chat-dot").hidden = true; $("#chat-input").focus(); }
}

/* ---------- start ---------- */
async function init() {
  $("#chat-fab").addEventListener("click", () => toggleChat());
  $("#chat-close").addEventListener("click", () => toggleChat(false));
  $("#open-chat-nav").addEventListener("click", () => toggleChat(true));
  $("#open-chat-hero").addEventListener("click", () => toggleChat(true));
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); sendMessage($("#chat-input").value); });
  $("#explain-btn").addEventListener("click", () => { toggleChat(true); sendMessage("Why is my water " + (lastResult && lastResult.potable ? "safe" : "unsafe") + "?"); });
  $("#check-btn").addEventListener("click", checkWater);
  $("#reset-btn").addEventListener("click", () => setAll(META.presets["Average sample"] || {}));
  addMsg("assistant", "Hi, I'm **AquaBot**. Check a water sample, then ask me why it scored the way it did. I can also explain terms like pH, turbidity or trihalomethanes.");
  setSuggestions(["What is turbidity?", "How accurate is the model?", "How can I improve water quality?"]);

  try {
    const res = await fetch(`${API_BASE}/api/meta`);
    if (!res.ok) throw new Error();
    META = await res.json();
  } catch {
    showError("Could not load the model. Run `python train_model.py`, then `python app.py` in the backend folder, and open http://127.0.0.1:5000.");
    $("#check-btn").disabled = true;
    return;
  }
  buildSliders();
  buildPresets();
  buildModel();
  setAll(META.presets["Average sample"] || {});
}
init();
