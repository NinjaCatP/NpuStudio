// NpuStudio Companion web UI. Served by the desktop at /app; the Android shell (window.NpuShell) is optional.
"use strict";
const $ = id => document.getElementById(id);
const UI_VERSION = "1.0.0";
const Shell = window.NpuShell || null;
const shellVersion = Shell && Shell.shellVersion ? Number(Shell.shellVersion()) : 0;

// ---------- key ----------
(function takeKeyFromHash() {             // the shell opens /app/#k=<key>; keep it, then clean the URL
  const t = location.hash.match(/t=(\w+)/); if (t) window.__startTab = t[1];
  const m = location.hash.match(/k=([^&]+)/);
  if (m || t) { if (m) localStorage.setItem("npuKey", decodeURIComponent(m[1])); history.replaceState(null, "", location.pathname); }
})();
const key = () => localStorage.getItem("npuKey") || "";

// ---------- API ----------
async function api(path, body, method) {
  const r = await fetch("/api/v1/" + path, {
    method: method || (body !== undefined ? "POST" : "GET"),
    headers: { "X-Key": key(), "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined, cache: "no-store"
  });
  let d = {}; try { d = await r.json(); } catch (e) {}
  if (!r.ok) { const e = new Error(d.error || ("HTTP " + r.status)); e.status = r.status; throw e; }
  return d;
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

function notifyDone(title, text) {
  if (!document.hidden) return;
  if (Shell && Shell.notify) Shell.notify(title, text);
  else if (window.Notification && Notification.permission === "granted") new Notification(title, { body: text });
}

// Runs a job with a progress card (bar, message, Cancel). onUpdate(status) for streaming text; onConfirm(q)->bool.
async function runJob(body, box, label, opts = {}) {
  const card = document.createElement("div"); card.className = "job";
  card.innerHTML = `<div class="row"><b class="grow">${label}</b><button class="ghost danger">Cancel</button></div><div class="bar ind"><i></i></div><div class="muted small msg"></div>`;
  if (box) { box.innerHTML = ""; box.appendChild(card); }
  const bar = card.querySelector(".bar"), msg = card.querySelector(".msg"), cancel = card.querySelector("button");
  const t0 = Date.now(); let id, cancelled = false, confirming = false;
  cancel.onclick = async () => { cancelled = true; cancel.disabled = true; msg.textContent = "Cancelling…"; if (id) try { await api(`jobs/${id}/cancel`, {}); } catch (e) {} };
  try {
    id = (await api("jobs", body)).id;
    while (true) {
      await sleep(600);
      const s = await api("jobs/" + id);
      if (s.total > 0) { bar.classList.remove("ind"); bar.firstChild.style.width = (100 * s.progress / s.total) + "%"; }
      const secs = ((Date.now() - t0) / 1000).toFixed(0);
      if (!cancelled) msg.textContent = (s.message || "working…") + ` · ${secs}s`;
      if (opts.onUpdate) opts.onUpdate(s);
      if (s.state === "confirm" && !confirming) {
        confirming = true;
        const yes = await ask(s.confirm || "Allow this action?");
        try { await api(`jobs/${id}/confirm`, { yes }); } catch (e) {}
        confirming = false;
      }
      if (s.state === "done" || s.state === "error" || s.state === "cancelled") break;
    }
    const r = await api(`jobs/${id}/result`);
    card.remove(); notifyDone("NpuStudio: " + label + " done", `Finished in ${r.seconds || "?"} s`);
    return r;
  } catch (e) {
    if (e.status === 410 || cancelled) { msg.textContent = "Cancelled."; bar.remove(); cancel.remove(); return null; }
    msg.innerHTML = `<span class="err">${esc(e.message)}</span>`; bar.remove(); cancel.remove();
    notifyDone("NpuStudio: " + label + " failed", e.message); return null;
  }
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function ask(q) {
  return new Promise(res => {
    $("dlgQ").textContent = q; $("dlg").classList.remove("hidden");
    const done = v => { $("dlg").classList.add("hidden"); res(v); };
    $("dlgYes").onclick = () => done(true); $("dlgNo").onclick = () => done(false);
  });
}

// ---------- images: pick / resize / save / share ----------
let cbSeq = 0; const cbs = {};
window.npuCallback = (id, data) => { const f = cbs[id]; delete cbs[id]; if (f) f(data); };   // called by the Android shell
function pickImage() {
  if (Shell && Shell.pickImage) return new Promise(res => { const id = "c" + (++cbSeq); cbs[id] = d => res(d ? "data:image/png;base64," + d.split(",").pop() : null); Shell.pickImage(id); });
  return new Promise(res => {
    const f = $("fileIn"); f.value = "";
    f.onchange = () => { const file = f.files[0]; if (!file) return res(null); const rd = new FileReader(); rd.onload = () => res(rd.result); rd.readAsDataURL(file); };
    f.click();
  });
}
function loadImg(src) { return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(new Error("could not read that image")); i.src = src; }); }
async function shrink(dataUrl, maxSide, type = "image/jpeg") {     // phone photos are huge; send a sensible size
  const im = await loadImg(dataUrl); const s = Math.min(1, maxSide / Math.max(im.naturalWidth, im.naturalHeight));
  const c = document.createElement("canvas"); c.width = Math.round(im.naturalWidth * s); c.height = Math.round(im.naturalHeight * s);
  c.getContext("2d").drawImage(im, 0, 0, c.width, c.height);
  return { b64: c.toDataURL(type, 0.92).split(",")[1], w: c.width, h: c.height };
}
function saveImg(b64, name) {
  if (Shell && Shell.saveImage) { Shell.saveImage(b64, name); return; }
  const a = document.createElement("a"); a.href = "data:image/png;base64," + b64; a.download = name; a.click();
}
async function shareImg(b64, name) {
  if (Shell && Shell.share) { Shell.share(b64, name); return; }
  try { const blob = await (await fetch("data:image/png;base64," + b64)).blob(); await navigator.share({ files: [new File([blob], name, { type: blob.type })] }); }
  catch (e) { saveImg(b64, name); }
}
function showResult(box, r, base, extra = "") {
  const name = `${base}-${new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19)}.png`;
  box.innerHTML = `<img src="data:image/png;base64,${r.image}"><div class="row"><button class="primary grow sv">💾 Save</button><button class="grow sh">📤 Share</button></div>
    <p class="muted small">${esc(r.device || "")} · ${r.seconds || "?"} s ${extra}</p>`;
  box.querySelector(".sv").onclick = () => saveImg(r.image, name);
  box.querySelector(".sh").onclick = () => shareImg(r.image, name);
}

// ---------- tabs ----------
let tab = "chat", monTimer = null;
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => {
  tab = b.dataset.t;
  document.querySelectorAll("#tabs button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll("main .page").forEach(p => p.classList.toggle("hidden", p.id !== "tab-" + tab));
  clearInterval(monTimer);
  if (tab === "mon") { monitor(); monTimer = setInterval(monitor, 2000); }
  if (tab === "desk") desk();
});

// ---------- chat ----------
function addMsg(cls, text) { const d = document.createElement("div"); d.className = "msg " + cls; d.textContent = text; $("chatLog").appendChild(d); d.scrollIntoView({ block: "end" }); return d; }
async function chatSend() {
  const m = $("chatIn").value.trim(); if (!m) return;
  $("chatIn").value = ""; addMsg("me", m); $("chatSend").disabled = true;
  const bot = addMsg("bot", "…"); const box = document.createElement("div"); $("chatLog").appendChild(box);
  const r = await runJob({ type: "chat", message: m }, box, "Reply", { onUpdate: s => { if (s.text) bot.textContent = s.text; } });
  box.remove(); $("chatSend").disabled = false;
  if (!r) { if (bot.textContent === "…") bot.textContent = "(no reply)"; return; }
  bot.textContent = r.text || bot.textContent;
  if (r.action) { const a = document.createElement("div"); a.className = "act"; a.textContent = `⚙ ${r.action}${r.action_result ? " → " + r.action_result : ""}`; bot.appendChild(a); }
  bot.scrollIntoView({ block: "end" });
}
$("chatSend").onclick = chatSend;
$("chatIn").onkeydown = e => { if (e.key === "Enter") chatSend(); };
$("chatNew").onclick = async () => { try { await api("chat/new", {}); } catch (e) {} $("chatLog").innerHTML = ""; };

// ---------- image gen ----------
$("genRand").onclick = () => $("genSeed").value = Math.floor(Math.random() * 1e9);
$("genGo").onclick = async () => {
  const p = $("genPrompt").value.trim(); if (!p) return $("genPrompt").focus();
  let [w, h] = $("genSize").value === "device"
    ? [Math.round(screen.width * devicePixelRatio), Math.round(screen.height * devicePixelRatio)]
    : $("genSize").value.split("x").map(Number);
  $("genGo").disabled = true;
  const r = await runJob({ type: "gen", prompt: p, width: w, height: h, steps: +$("genSteps").value || 4, seed: +$("genSeed").value || 0 }, $("genJob"), "Image Gen");
  $("genGo").disabled = false; if (r) showResult($("genOut"), r, "npustudio-gen", `· ${w}×${h}`);
};

// ---------- edit / AI change ----------
let editSrc = null, upSrc = null;
$("editPick").onclick = async () => { const d = await pickImage(); if (!d) return; editSrc = d; $("editIn").src = d; $("editIn").classList.remove("hidden"); $("editOut").innerHTML = ""; };
async function editRun(kind) {
  if (!editSrc) return alert("Pick a photo first.");
  const instr = $("editInstr").value.trim(); if (kind === "change" && !instr) return $("editInstr").focus();
  const im = await shrink(editSrc, 1536);
  const body = kind === "change" ? { type: "change", image: im.b64, instruction: instr, method: $("editMethod").value } : { type: "remove_bg", image: im.b64 };
  $("editGo").disabled = $("editBg").disabled = true;
  const r = await runJob(body, $("editJob"), kind === "change" ? "AI Change" : "Remove background");
  $("editGo").disabled = $("editBg").disabled = false;
  if (r) showResult($("editOut"), r, kind === "change" ? "npustudio-edit" : "npustudio-cutout", r.method ? "· " + esc(r.method) : "");
}
$("editGo").onclick = () => editRun("change");
$("editBg").onclick = () => editRun("remove_bg");

// ---------- upscale ----------
$("upPick").onclick = async () => {
  const d = await pickImage(); if (!d) return; upSrc = d; $("upIn").src = d; $("upIn").classList.remove("hidden"); $("upOut").innerHTML = "";
  const i = await loadImg(d); $("upInfo").textContent = `${i.naturalWidth}×${i.naturalHeight}`;
};
$("upGo").onclick = async () => {
  if (!upSrc) return alert("Pick a photo first.");
  const sc = +$("upScale").value, im = await shrink(upSrc, Math.max(512, Math.floor(8192 / sc)), "image/png");
  $("upGo").disabled = true;
  const r = await runJob({ type: "upscale", image: im.b64, scale: sc, model: $("upModel").value, format: "PNG" }, $("upJob"), "Upscale");
  $("upGo").disabled = false; if (r) showResult($("upOut"), r, "npustudio-upscale", `· ${im.w * sc}×${im.h * sc}`);
};

// ---------- monitor ----------
const na = v => (v === null || v === undefined || v === "N/A") ? "N/A" : (typeof v === "number" ? (Number.isInteger(v) ? v : +v.toFixed(1)) : v);
const mb = v => typeof v === "number" ? Math.round(v).toLocaleString() + " MB" : na(v);
function chipRows(chips) {
  return Object.entries(chips || {}).filter(([n, c]) => !n.startsWith("_") && c && typeof c === "object").map(([n, c]) =>
    `<h3>${esc(n)}</h3><div class="kv"><b>Use</b><span>${na(c.use)}${c.use != null && c.use !== "N/A" ? "%" : ""}</span>
     <b>Dedicated mem</b><span>${mb(c.ded)}</span><b>Shared mem</b><span>${mb(c.shared)}</span>
     <b>Power</b><span>${na(c.watts)}${typeof c.watts === "number" ? " W" : ""}</span><b>Temp</b><span>${na(c.temp)}${typeof c.temp === "number" ? " °C" : ""}</span></div>`).join("");
}
async function monitor() {
  try {
    const m = await api("monitor"), d = m.desktop;
    let h = `<h2>🖥️ Desktop · ${esc(d.name || "")}</h2><div class="kv"><b>GPU</b><span>${esc(d.gpu || "N/A")}</span><b>Use</b><span>${na(d.use)}%</span>
      <b>VRAM</b><span>${na(d.vram_used)} / ${na(d.vram_total)} MB</span><b>Power</b><span>${na(d.watts)} W</span><b>Temp</b><span>${na(d.temp)} °C</span><b>Jobs running</b><span>${d.busy}</span></div>`;
    h += chipRows(m.chips).replace(/<h3>Intel Arc<\/h3>/, "<h3>Intel UHD 770 (desktop iGPU)</h3>").replace(/<h3>NVIDIA<\/h3>/, "<h3>NVIDIA (desktop)</h3>");
    const others = Object.entries(m.others || {});
    h += others.length ? others.map(([n, o]) => `<h2>💻 ${esc(n)}</h2>${chipRows(o.chips)}`).join("") : `<h2>💻 Laptop</h2><p class="muted">Offline (NpuStudio isn't open on the laptop).</p>`;
    $("monBox").innerHTML = h;
  } catch (e) { $("monBox").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

// ---------- desktop status ----------
async function desk() {
  try {
    const [i, hl] = await Promise.all([api("info"), api("health")]);
    $("deskBox").innerHTML = `<div class="kv"><b>PC</b><span>${esc(i.name)}</span><b>Home address</b><span>${esc(i.lan)}:${i.port}</span>
      <b>Tailscale</b><span>${i.tailscale && i.tailscale.length ? esc(i.tailscale.join(", ")) : "not set up"}</span>
      <b>GPU</b><span>${esc(hl.gpu || "N/A")}</span><b>Jobs running</b><span>${i.busy}</span>
      <b>Web UI</b><span>${UI_VERSION} (server has ${esc((i.ui || {}).ui_version || "?")})</span><b>Android shell</b><span>${shellVersion || "browser"}</span></div>
      ${(i.jobs || []).length ? "<h3>Jobs</h3>" + i.jobs.map(j => `<div class="row"><span class="grow">${esc(j.type)} · ${esc(j.state)}</span>${j.state === "running" || j.state === "confirm" ? `<button class="ghost danger" data-id="${j.id}">Cancel</button>` : ""}</div>`).join("") : ""}`;
    $("deskBox").querySelectorAll("button[data-id]").forEach(b => b.onclick = async () => { await api(`jobs/${b.dataset.id}/cancel`, {}); desk(); });
  } catch (e) { $("deskBox").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}
$("deskRe").onclick = desk;
$("unpair").onclick = () => { if (!confirm("Forget the pairing key?")) return; localStorage.removeItem("npuKey"); if (Shell && Shell.unpair) Shell.unpair(); else location.reload(); };

// ---------- start ----------
async function start() {
  if (!key()) { $("pair").classList.remove("hidden"); return; }
  try {
    await api("health"); $("conn").textContent = "Desktop online"; $("conn").className = "pill ok";
    $("pair").classList.add("hidden"); $("main").classList.remove("hidden"); $("tabs").classList.remove("hidden");
    try { const v = await (await fetch("version.json", { cache: "no-store" })).json(); if (Shell && v.min_shell > shellVersion) alert("Please update the NpuStudio Companion app (this UI needs shell v" + v.min_shell + ")."); } catch (e) {}
    if (window.__startTab) { const b = document.querySelector(`#tabs button[data-t="${window.__startTab}"]`); if (b) b.click(); }
    if (!Shell && window.Notification && Notification.permission === "default") Notification.requestPermission();
  } catch (e) {
    if (e.status === 403) { $("pairMsg").textContent = "That key was refused."; $("pair").classList.remove("hidden"); $("main").classList.add("hidden"); $("tabs").classList.add("hidden"); }
    else { $("conn").textContent = "Offline"; $("conn").className = "pill bad"; }
  }
}
$("pairBtn").onclick = () => { localStorage.setItem("npuKey", $("pairKey").value.trim()); $("pairMsg").textContent = ""; start(); };
setInterval(async () => { if (!key()) return; try { await api("health"); $("conn").textContent = "Desktop online"; $("conn").className = "pill ok"; } catch (e) { $("conn").textContent = "Offline"; $("conn").className = "pill bad"; } }, 10000);
start();
