// NpuStudio Companion web UI. Served by the desktop at /app; the Android shell (window.NpuShell) is optional.
"use strict";
const $ = id => document.getElementById(id);
const UI_VERSION = "1.6.8";
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
// Identify this phone to the desktop so it shows in every Control picker (same id as the v2.1 background link).
function devHeaders() {
  if (!/Android/.test(navigator.userAgent)) return {};
  let id = null, name = null;
  try { if (typeof Shell !== "undefined" && Shell && Shell.getBackground) { const b = JSON.parse(Shell.getBackground()); id = b.id; name = b.name; } } catch (e) {}
  if (!id) { id = localStorage.getItem("npuDevId") || ("w" + Math.random().toString(36).slice(2, 9)); localStorage.setItem("npuDevId", id); }
  const h = { "X-Device-Id": id }; if (name) h["X-Device-Name"] = name;
  try { if (typeof shellVersion !== "undefined" && shellVersion) h["X-Shell-Version"] = String(shellVersion); } catch (e) {}
  return h;
}
async function api(path, body, method) {
  const r = await fetch("/api/v1/" + path, {
    method: method || (body !== undefined ? "POST" : "GET"),
    headers: { "X-Key": key(), "Content-Type": "application/json", ...devHeaders() },
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

// XHR so the card can show upload / download percentages.
function xhrJson(method, path, body, onProg) {
  return new Promise((res, rej) => {
    const x = new XMLHttpRequest(); x.open(method, "/api/v1/" + path); x.setRequestHeader("X-Key", key());
    if (body !== undefined) x.setRequestHeader("Content-Type", "application/json");
    if (onProg) { x.upload.onprogress = e => e.lengthComputable && onProg("up", e.loaded / e.total); x.onprogress = e => onProg("down", e.lengthComputable ? e.loaded / e.total : -e.loaded); }
    x.onload = () => { let d = {}; try { d = JSON.parse(x.responseText); } catch (e) {}
      if (x.status >= 200 && x.status < 300) res(d); else { const e = new Error(d.error || ("HTTP " + x.status)); e.status = x.status; rej(e); } };
    x.onerror = () => rej(new Error("Network error - is the desktop reachable?"));
    x.send(body !== undefined ? JSON.stringify(body) : null);
  });
}
const fmtMB = n => n < 1048576 ? Math.max(1, Math.round(n / 1024)) + " KB" : (n / 1048576).toFixed(n < 10485760 ? 1 : 0) + " MB";
// Runs a job with a status card: Preparing → Uploading x% → Starting → server message + progress → Downloading result → Done / error.
// body may be an object or an async function returning one (e.g. shrinking a photo). opts: onUpdate(status), btn (disabled while running),
// done(r) -> text for the "Done" line (card stays), keep (leave the Done line visible).
async function runJob(body, box, label, opts = {}) {
  const card = document.createElement("div"); card.className = "job";
  card.innerHTML = `<div class="row"><b class="grow">${label}</b><span class="muted small el"></span><button class="ghost danger">Cancel</button></div><div class="bar ind"><i></i></div><div class="small msg st"></div>`;
  if (box) { box.innerHTML = ""; box.appendChild(card); }
  const bar = card.querySelector(".bar"), msg = card.querySelector(".msg"), cancel = card.querySelector("button"), el = card.querySelector(".el");
  const setBar = f => { if (f == null) bar.classList.add("ind"); else { bar.classList.remove("ind"); bar.firstChild.style.width = Math.round(100 * Math.max(0, Math.min(1, f))) + "%"; } };
  const t0 = Date.now(); let id, cancelled = false, confirming = false, phase = "Preparing…";
  const tick = setInterval(() => { el.textContent = ((Date.now() - t0) / 1000).toFixed(0) + " s"; }, 500);
  const say = t => { if (!cancelled) msg.textContent = t; };
  if (opts.btn) opts.btn.disabled = true;
  cancel.onclick = async () => { cancelled = true; cancel.disabled = true; msg.textContent = "Cancelling…"; if (id) try { await api(`jobs/${id}/cancel`, {}); } catch (e) {} };
  try {
    say("Preparing…");
    if (typeof body === "function") body = await body();
    if (cancelled) throw Object.assign(new Error("cancelled"), { status: 410 });
    const size = JSON.stringify(body).length;
    say(size > 200000 ? `Uploading ${fmtMB(size)}… 0%` : "Sending…"); setBar(size > 200000 ? 0 : null);
    id = (await xhrJson("POST", "jobs", body, (k, f) => { if (k === "up") { say(`Uploading ${fmtMB(size)}… ${Math.round(f * 100)}%`); setBar(f); } })).id;
    say("Queued - waiting for the desktop to start…"); setBar(null);
    while (true) {
      await sleep(600);
      const s = await api("jobs/" + id);
      if (s.total > 0) setBar(s.progress / s.total); else setBar(null);
      say((s.message || "Starting…") + (s.total > 0 ? ` (${Math.round(100 * s.progress / s.total)}%)` : ""));
      if (opts.onUpdate) opts.onUpdate(s);
      if (s.state === "confirm" && !confirming) {
        confirming = true;
        const yes = await ask(s.confirm || "Allow this action?");
        try { await api(`jobs/${id}/confirm`, { yes }); } catch (e) {}
        confirming = false;
      }
      if (s.state === "done" || s.state === "error" || s.state === "cancelled") break;
    }
    say("Downloading result…"); setBar(null);
    const r = await xhrJson("GET", `jobs/${id}/result`, undefined, (k, f) => { if (k === "down") { if (f >= 0) { setBar(f); say(`Downloading result… ${Math.round(f * 100)}%`); } else say(`Downloading result… ${fmtMB(-f)}`); } });
    clearInterval(tick); cancel.remove(); bar.remove();
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    if (opts.done || opts.keep) { msg.className = "small msg ok"; msg.textContent = "✓ " + (opts.done ? opts.done(r, secs) : `Done in ${secs} s`) + savedLine(r); el.textContent = ""; }
    else card.remove();
    notifyDone("NpuStudio: " + label + " done", `Finished in ${secs} s`);
    return r;
  } catch (e) {
    clearInterval(tick); bar.remove(); cancel.remove();
    if (e.status === 410 || cancelled) { msg.textContent = "Cancelled."; return null; }
    msg.innerHTML = `<span class="err">⚠ ${esc(e.message)}</span>`;
    notifyDone("NpuStudio: " + label + " failed", e.message); return null;
  } finally { if (opts.btn) opts.btn.disabled = false; }
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// Every desktop job keeps its full-size result in the desktop output folder.
function savedLine(r) {
  if (!r) return "";
  if (r.saved_path) return `\nFull size ${r.full_width}×${r.full_height} saved on desktop: ${r.saved_path}` + (r.full_note ? `\n(${r.full_note})` : "");
  return r.save_error ? `\n⚠ Couldn't save on the desktop: ${r.save_error}` : "";
}
// In-page dialogs: Android WebView silently drops window.alert/confirm, so never use them.
function notice(q) { return ask(q, true); }
function ask(q, okOnly) {
  return new Promise(res => {
    $("dlgQ").textContent = q; $("dlg").classList.remove("hidden");
    $("dlgNo").classList.toggle("hidden", !!okOnly); $("dlgYes").textContent = okOnly ? "OK" : "Yes";
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
  window.LAST = { b64: r.image, name };
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
  const bot = addMsg("bot", "Thinking…"); bot.classList.add("thinking"); const box = document.createElement("div"); $("chatLog").appendChild(box);
  const target = $("chatTarget").value;
  const r = await runJob({ type: "chat", message: m, device: target }, box, "Reply", { onUpdate: s => { if (s.text) { bot.textContent = s.text; bot.classList.remove("thinking"); } } });
  bot.classList.remove("thinking");
  box.remove(); $("chatSend").disabled = false;
  if (!r) { if (bot.textContent === "Thinking…") bot.textContent = "(no reply)"; return; }
  bot.textContent = r.text || (bot.textContent === "Thinking…" ? "" : bot.textContent);
  if (r.phone_action) {
    const pa = r.phone_action, a = document.createElement("div"); a.className = "act"; bot.appendChild(a);
    a.textContent = "📱 " + pa.name + " …";
    if (r.needs_confirm && !(await ask(`Do you want me to ${r.confirm_text || pa.name}?`))) a.textContent = "📱 Cancelled: " + (r.confirm_text || pa.name);
    else a.textContent = "📱 " + pa.name + " → " + await phoneAction(pa);
  } else if (!r.text && !r.action && bot.textContent === "") bot.textContent = "(no reply)";
  if (r.action) { const a = document.createElement("div"); a.className = "act"; a.textContent = `⚙ ${r.action}${r.action_result ? " → " + r.action_result : ""}`; bot.appendChild(a); }
  bot.scrollIntoView({ block: "end" });
}
// Phone actions: allowlisted, run by the Android app (shell v2+). Nothing here can run arbitrary code.
async function phoneAction(pa) {
  const g = pa.args || {};
  if (pa.name === "save_image" || pa.name === "share_image") {
    if (!window.LAST) return "No picture yet — make or open one first.";
    (pa.name === "save_image" ? saveImg : shareImg)(window.LAST.b64, window.LAST.name); return pa.name === "save_image" ? "Saving the last picture." : "Sharing the last picture.";
  }
  if (pa.name === "time") return new Date().toLocaleString();
  const V3 = ["brightness", "brightness_change", "brightness_auto", "dnd", "ring_mode", "rotation_lock", "settings_panel", "keep_screen_on", "camera", "navigate",
    "maps_search", "call", "sms", "calendar_event", "media", "storage", "network", "note", "notes", "speak", "app_settings", "find_phone", "stop_sound"];
  if (Shell && V3.includes(pa.name) && shellVersion < 3) {
    if (pa.name === "navigate" || pa.name === "maps_search") { location.href = "geo:0,0?q=" + encodeURIComponent(g.place || g.query || ""); return "Opened maps."; }
    return `Update the app to use ${pa.name.replace(/_/g, " ")} (Companion v2.1). See the update banner at the top.`;
  }
  if (!(Shell && typeof Shell.phoneAction === "function")) {
    if (pa.name === "open_url") { window.open(/^https?:/.test(g.url) ? g.url : "https://" + g.url); return "Opened."; }
    if (pa.name === "web_search") { window.open("https://www.bing.com/search?q=" + encodeURIComponent(g.query || "")); return "Searched."; }
    if (pa.name === "notify") { notifyDone(g.title || "NpuStudio", g.text || ""); return "Sent."; }
    return "This needs the updated Android app (v2). " + (Shell ? "Install the new APK." : "Open NpuStudio in the Android app.");
  }
  return new Promise(res => { const id = "c" + (++cbSeq); cbs[id] = d => res(d || "Done."); Shell.phoneAction(JSON.stringify(pa), id); setTimeout(() => res("No answer from the phone app."), 15000); });
}
const HINTS = {
  phone: "Commands run on this phone: apps, flashlight, volume (media/ring/alarm/notification/call), brightness and auto-brightness, Do Not Disturb, ring/vibrate/silent, rotation lock, keep screen on, Wi-Fi/Bluetooth/data/NFC/location panels, camera, navigate/maps, call or text (you tap send), calendar events, play/pause/next, alarms/timers, battery/storage/network, notes, read aloud, app settings, find my phone, links, clipboard, save/share the last picture.",
  remotephone: "Commands are sent to that phone's background link: volume, brightness, Do Not Disturb, ring modes, flashlight, media, alarms/timers, battery/storage/network, notes, read aloud, find my phone, open apps or links. Calls/texts are prefilled there and ask here first.",
  pc: "Commands run on that PC: apps, volume, brightness, media, screenshot, lock, files, recent files, clipboard, system info, Wi-Fi/Bluetooth/battery saver/night light settings, empty the Recycle Bin, find my phone. Risky ones ask you here first."
};
(function chatSetup() {
  const sel = $("chatTarget");
  const me = devHeaders()["X-Device-Id"] || null;
  const hint = () => { const v = sel.value; $("chatHint").textContent = v === "phone" ? HINTS.phone : v.startsWith("phone:") ? HINTS.remotephone : HINTS.pc; };
  const fill = devs => {
    const want = localStorage.getItem("npuTarget") || (Shell ? "phone" : "pc"); sel.innerHTML = "";
    const add = (v, label, on, kind) => { const o = document.createElement("option"); o.value = v; o.textContent = (kind === "phone" ? "📱 " : "🖥 ") + label + (on ? "" : " (offline)"); o.disabled = !on; o.style.color = on ? "" : "#999"; sel.appendChild(o); };
    if (Shell) add("phone", "This phone", true, "phone");
    for (const d of devs) {
      if (d.id === "phone:" + me) continue;
      add(d.id === "desktop" ? "pc" : d.id, d.id === "desktop" ? "Desktop PC" : d.name, d.online, d.kind);
    }
    const ok = [...sel.options].find(o => o.value === want && !o.disabled); sel.value = ok ? want : sel.options[0].value; hint();
  };
  const load = async () => { try { const r = await api("devices"); if (r && r.devices) fill(r.devices); } catch (e) {} };
  fill([{ id: "desktop", name: "Desktop PC", kind: "pc", online: true }]); load(); setInterval(load, 20000);
  sel.addEventListener("focus", load); sel.addEventListener("mousedown", load);
  sel.onchange = () => { localStorage.setItem("npuTarget", sel.value); hint(); };
})();
async function shellUpdateCheck() {
  try {
    const v = await (await fetch("version.json", { cache: "no-store" })).json();
    if (Shell && shellVersion < (v.latest_shell || v.min_shell || 1)) {
      const u = location.origin + "/app/NpuStudio-Companion.apk";
      $("shellUpd").innerHTML = `📲 An Android app update (v${v.latest_shell === 3 ? "2.1" : v.latest_shell}) adds brightness, Do Not Disturb, ring modes, calls/texts, maps, notes, read aloud and Find my phone. Install <b>NpuStudio-Companion.apk</b> over this one: open <a href="${u}" target="_blank">${u}</a> in Chrome on this phone, or copy it from H:\\downloads on the desktop. Your pairing is kept.`;
      $("shellUpd").classList.remove("hidden");
    }
  } catch (e) {}
}
$("chatSend").onclick = chatSend;
$("chatIn").onkeydown = e => { if (e.key === "Enter") chatSend(); };
$("chatNew").onclick = async () => { try { await api("chat/new", {}); } catch (e) {} $("chatLog").innerHTML = ""; };

// ---------- image gen ----------
let OPTS = null;
async function loadOptions() {
  try { OPTS = await api("options"); } catch (e) { return; }
  const g = $("genPcSizes"); g.innerHTML = "";
  (OPTS.gen_sizes || []).forEach(v => { const o = document.createElement("option"); o.value = v; o.textContent = v.replace("x", "×"); g.appendChild(o); });
}
function genTarget() {
  const v = $("genSize").value;
  if (v === "device") return [Math.round(screen.width * devicePixelRatio), Math.round(screen.height * devicePixelRatio)];
  if (v === "custom") return [+$("genW").value || 1024, +$("genH").value || 1024];
  return v.split("x").map(Number);
}
function genPlan() {
  $("genCustom").classList.toggle("hidden", $("genSize").value !== "custom");
  const [w, h] = genTarget(); $("genPlan").textContent = `Output ${w}×${h}. Drawn at a base size, then upscaled with ${$("genUp").value === "fast" ? "Fast" : "Sharp"} and cropped to fit.`;
}
["genSize", "genUp", "genW", "genH"].forEach(id => $(id).addEventListener("input", genPlan));
$("genRand").onclick = () => $("genSeed").value = Math.floor(Math.random() * 2147483647);
$("genGo").onclick = async () => {
  const p = $("genPrompt").value.trim(); if (!p) return $("genPrompt").focus();
  const [w, h] = genTarget();
  if (!(w >= 64 && h >= 64 && w <= 16384 && h <= 16384)) return notice("Size must be between 64 and 16384 on each side.");
  const r = await runJob({ type: "gen", prompt: p, width: w, height: h, steps: +$("genSteps").value || 4, seed: +$("genSeed").value || 0, upscaler: $("genUp").value, autofix: $("genFix").checked }, $("genJob"), "Image Gen",
    { btn: $("genGo"), done: (r, s) => `Done in ${s} s · ${w}×${h}` });
  if (r) { showResult($("genOut"), r, "npustudio-gen", `· ${w}×${h}`); addSend($("genOut"), r.image, r.gen_meta); }
};
function addSend(box, b64, meta) {
  const row = document.createElement("div"); row.className = "row";
  row.innerHTML = `<button class="grow">✏️ Send to Edit</button><button class="grow">🔍 Send to Upscale</button>`;
  row.children[0].onclick = async () => { await editLoad("data:image/png;base64," + b64, true); setGen(meta || null); tabTo("edit"); };
  row.children[1].onclick = () => { upLoad("data:image/png;base64," + b64); tabTo("up"); };
  box.appendChild(row);
}
// ---------- Analyze tab: description + Gen-ready prompt ----------
let anaSrc = null, anaDim = null;
async function anaLoad(d) { anaSrc = d; $("anaIn").src = d; $("anaIn").classList.remove("hidden"); const i = await loadImg(d); anaDim = [i.naturalWidth, i.naturalHeight]; $("anaSize").textContent = `Size: ${anaDim[0]}×${anaDim[1]}`; $("anaDesc").value = ""; $("anaPrompt").value = ""; $("anaPose").textContent = ""; }
$("anaPick").onclick = async () => { const d = await pickImage(); if (d) { await anaLoad(d); $("anaGo").click(); } };
$("anaGo").onclick = async () => {
  if (!anaSrc) return notice("Pick a picture first.");
  const s = await shrinkUrl(anaSrc, 1024);
  const r = await runJob({ type: "analyze", image: s.split(",")[1], device: $("anaDev").value, width: anaDim[0], height: anaDim[1], save_txt: $("anaTxt").checked }, $("anaJob"), "Analyze",
    { btn: $("anaGo"), done: (r, s) => `Analyzed in ${s} s${r.device ? " on " + r.device : ""}${r.saved_path ? " · saved .txt" : ""}` });
  if (!r) return;
  $("anaDesc").value = r.description || ""; $("anaPrompt").value = r.prompt || ""; $("anaSize").textContent = `Size: ${r.width}×${r.height}`; $("anaPose").textContent = "Pose: " + (r.pose || "-"); $("anaGo").textContent = "🔄 Re-analyze";
};
const copyTxt = async (t, what) => { try { await navigator.clipboard.writeText(t); notice(what + " copied."); } catch { const x = document.createElement("textarea"); x.value = t; document.body.appendChild(x); x.select(); document.execCommand("copy"); x.remove(); notice(what + " copied."); } };
$("anaCopyD").onclick = () => $("anaDesc").value && copyTxt($("anaDesc").value, "Description");
$("anaCopyP").onclick = () => $("anaPrompt").value && copyTxt($("anaPrompt").value, "Prompt");
$("anaUse").onclick = () => { if (!$("anaPrompt").value) return; $("genPrompt").value = $("anaPrompt").value; tabTo("gen"); };
function tabTo(t) { const b = document.querySelector(`#tabs button[data-t="${t}"]`); if (b) b.click(); }

// ---------- edit (laptop Edit tab: transform, crop, resize, adjust, brush mask, AI erase / remove bg / AI Change) ----------
const ED = { cur: null, undo: [], redo: [], w: 0, h: 0, gen: null };
// "From Gen": pictures sent here from Image Gen get a looser AI Change for non-body edits (safety rules unchanged)
function setGen(m) { ED.gen = m; $("genBadge").classList.toggle("hidden", !m); $("editRegen").classList.toggle("hidden", !m); }
const EDIT_MAX = 2048;   // working size sent to the desktop
async function editLoad(src, fromResult) {
  const s = await shrinkUrl(src, EDIT_MAX);
  if (ED.cur && !fromResult) { ED.undo.push(ED.cur); ED.redo = []; }
  else if (ED.cur) { ED.undo.push(ED.cur); ED.redo = []; }
  setEdit(s);
}
async function shrinkUrl(src, max) { const r = await shrink(src, max, "image/png"); return "data:image/png;base64," + r.b64; }
function setEdit(url) {
  ED.cur = url; const img = $("editIn"); img.onload = () => {
    ED.w = img.naturalWidth; ED.h = img.naturalHeight; const cv = $("maskCv");
    cv.width = ED.w; cv.height = ED.h; fitMask();
    $("editInfo").textContent = `${ED.w}×${ED.h}`; previewAdj();
  };
  img.src = url; $("stage").classList.remove("hidden"); $("editTools").classList.remove("hidden");
  $("editUndo").disabled = !ED.undo.length; $("editRedo").disabled = !ED.redo.length;
}
function fitMask() { const img = $("editIn"), cv = $("maskCv"); cv.style.left = img.offsetLeft + "px"; cv.style.width = img.clientWidth + "px"; cv.style.height = img.clientHeight + "px"; }
window.addEventListener("resize", fitMask);
function pushEdit(url) { ED.undo.push(ED.cur); ED.redo = []; clearMask(); setEdit(url); }
$("editUndo").onclick = () => { if (!ED.undo.length) return; ED.redo.push(ED.cur); setEdit(ED.undo.pop()); };
$("editRedo").onclick = () => { if (!ED.redo.length) return; ED.undo.push(ED.cur); setEdit(ED.redo.pop()); };
$("editPick").onclick = async () => { const d = await pickImage(); if (!d) return; ED.undo = []; ED.redo = []; ED.cur = null; setGen(null); clearMask(); await editLoad(d); $("editOut").innerHTML = ""; };
async function canvasOp(fn) {      // draws the current image through fn(ctx, img) on a new canvas
  const im = await loadImg(ED.cur); const c = document.createElement("canvas"); fn(c, c.getContext("2d"), im);
  pushEdit(c.toDataURL("image/png"));
}
document.querySelectorAll("[data-tf]").forEach(b => b.onclick = () => canvasOp((c, x, im) => {
  const t = b.dataset.tf, w = im.naturalWidth, h = im.naturalHeight, rot = t === "rotl" || t === "rotr";
  c.width = rot ? h : w; c.height = rot ? w : h; x.translate(c.width / 2, c.height / 2);
  if (t === "rotl") x.rotate(-Math.PI / 2); if (t === "rotr") x.rotate(Math.PI / 2); if (t === "r180") x.rotate(Math.PI);
  if (t === "fliph") x.scale(-1, 1); if (t === "flipv") x.scale(1, -1);
  x.drawImage(im, -w / 2, -h / 2);
}));
$("cropGo").onclick = () => { const r = +$("cropAsp").value; if (!r) return; canvasOp((c, x, im) => {
  const w = im.naturalWidth, h = im.naturalHeight; let cw = w, ch = Math.round(w / r); if (ch > h) { ch = h; cw = Math.round(h * r); }
  c.width = cw; c.height = ch; x.drawImage(im, (w - cw) / 2, (h - ch) / 2, cw, ch, 0, 0, cw, ch);
}); };
$("rsGo").onclick = () => { const p = Math.max(5, Math.min(400, +$("rsPct").value || 100)) / 100; canvasOp((c, x, im) => {
  c.width = Math.max(1, Math.round(im.naturalWidth * p)); c.height = Math.max(1, Math.round(im.naturalHeight * p)); x.imageSmoothingQuality = "high"; x.drawImage(im, 0, 0, c.width, c.height);
}); };
const FILT = { "None": "", "B&W": "grayscale(1)", "Sepia": "sepia(1)", "Blur": "blur(2px)", "Sharpen": "contrast(1.15)", "Warm": "sepia(0.25) saturate(1.2)", "Cool": "hue-rotate(12deg) saturate(0.9) brightness(1.03)" };
function adjFilter() { return `brightness(${$("adjB").value}) contrast(${$("adjC").value}) saturate(${$("adjS").value}) ${FILT[$("adjF").value] || ""}`; }
function previewAdj() { $("bV").textContent = (+$("adjB").value).toFixed(2); $("cV").textContent = (+$("adjC").value).toFixed(2); $("sV").textContent = (+$("adjS").value).toFixed(2); $("editIn").style.filter = adjFilter(); }
["adjB", "adjC", "adjS", "adjF"].forEach(id => $(id).addEventListener("input", previewAdj));
function resetAdj() { $("adjB").value = $("adjC").value = $("adjS").value = 1; $("adjF").value = "None"; previewAdj(); }
$("adjReset").onclick = resetAdj;
$("adjGo").onclick = async () => { const f = adjFilter(); await canvasOp((c, x, im) => { c.width = im.naturalWidth; c.height = im.naturalHeight; x.filter = f; x.drawImage(im, 0, 0); }); resetAdj(); };
// brush mask
let painting = false;
$("brush").oninput = () => $("brV").textContent = $("brush").value;
$("brOn").onchange = () => $("stage").classList.toggle("paint", $("brOn").checked);
function clearMask() { const cv = $("maskCv"); cv.getContext("2d").clearRect(0, 0, cv.width, cv.height); cv.dataset.dirty = ""; }
$("maskClear").onclick = clearMask;
function dab(e) {
  const cv = $("maskCv"), r = cv.getBoundingClientRect(), s = cv.width / r.width, x = cv.getContext("2d");
  x.globalCompositeOperation = $("penErase").checked ? "destination-out" : "source-over";
  x.fillStyle = "#00dcff"; x.beginPath(); x.arc((e.clientX - r.left) * s, (e.clientY - r.top) * s, (+$("brush").value / 2) * s, 0, 7); x.fill();
  x.globalCompositeOperation = "source-over"; cv.dataset.dirty = "1";
}
const strokes = [];
$("maskCv").addEventListener("pointerdown", e => { const cv = $("maskCv"); strokes.push(cv.getContext("2d").getImageData(0, 0, cv.width, cv.height)); if (strokes.length > 20) strokes.shift(); painting = true; $("maskCv").setPointerCapture(e.pointerId); dab(e); });
$("maskCv").addEventListener("pointermove", e => { if (painting) dab(e); });
["pointerup", "pointercancel"].forEach(t => $("maskCv").addEventListener(t, () => painting = false));
$("maskUndo").onclick = () => { const d = strokes.pop(); if (d) { const cv = $("maskCv"); cv.getContext("2d").putImageData(d, 0, 0); } };
["penAdd", "penErase"].forEach(id => $(id).onchange = () => { $("brOn").checked = true; $("brOn").onchange(); });
// Load the auto-mask (cyan) onto the brush layer so it can be fixed with Add / Erase before AI Change
$("maskAuto").onclick = async () => {
  const instr = $("editInstr").value.trim(); if (!ED.cur) return notice("Pick a photo first."); if (!instr) return notice("Type what to change first (e.g. 'change the outfit to a red dress').");
  const r = await runJob({ type: "find_mask", image: curB64(), instruction: instr }, $("editJob"), "Find target", { done: (r, s) => r.mask ? `Found (${r.conf}) - ${r.how}` : `Not found - ${r.how}` });
  if (!r || !r.mask) return;
  const m = await loadImg("data:image/png;base64," + r.mask), cv = $("maskCv"), x = cv.getContext("2d");
  const t = document.createElement("canvas"); t.width = cv.width; t.height = cv.height; const tx = t.getContext("2d"); tx.drawImage(m, 0, 0, t.width, t.height);
  const d = tx.getImageData(0, 0, t.width, t.height), o = x.getImageData(0, 0, cv.width, cv.height);
  strokes.push(x.getImageData(0, 0, cv.width, cv.height));
  for (let i = 0; i < d.data.length; i += 4) if (d.data[i] > 127) { o.data[i] = 0; o.data[i + 1] = 220; o.data[i + 2] = 255; o.data[i + 3] = 255; }
  x.putImageData(o, 0, 0); cv.dataset.dirty = "1"; $("brOn").checked = true; $("brOn").onchange();
  $("editOut").innerHTML = `<p class="muted small">Cyan = area that will change. Use ➕ Add / ➖ Erase to fix it, then tap AI Change.</p>`;
};
function maskB64() {
  const cv = $("maskCv"); if (!cv.dataset.dirty) return null;
  const m = document.createElement("canvas"); m.width = cv.width; m.height = cv.height; const x = m.getContext("2d");
  x.fillStyle = "#000"; x.fillRect(0, 0, m.width, m.height); x.globalCompositeOperation = "source-over";
  const src = cv.getContext("2d").getImageData(0, 0, cv.width, cv.height), d = x.getImageData(0, 0, m.width, m.height);
  for (let i = 3; i < src.data.length; i += 4) if (src.data[i] > 0) { d.data[i - 3] = d.data[i - 2] = d.data[i - 1] = 255; }
  x.putImageData(d, 0, 0); return m.toDataURL("image/png").split(",")[1];
}
const curB64 = () => ED.cur.split(",")[1];
async function editJob(body, label) {
  if (!ED.cur) return notice("Pick a photo first.");
  document.querySelectorAll("#editTools button").forEach(b => b.disabled = true);
  const r = await runJob(body, $("editJob"), label, { done: (r, s) => `${label} done in ${s} s${r.device ? " on " + r.device : ""}` });
  document.querySelectorAll("#editTools button").forEach(b => b.disabled = false);
  if (!r) return;
  const before = ED.cur, url = "data:image/png;base64," + r.image;
  if (r.unchanged) { $("editOut").innerHTML = `<p class="muted small">${esc(label)}: picture unchanged${r.method ? " · " + esc(r.method) : ""}</p>`; return; }
  pushEdit(url);
  $("editOut").innerHTML = `<p class="muted small">${esc(label)} done on ${esc(r.device || "")} in ${r.seconds || "?"} s${r.method ? " · " + esc(r.method) : ""}. Undo to revert.</p>`;
  if (r.changed_box) showChange(before, url, r.changed_box);
  $("stage").scrollIntoView({ behavior: "smooth", block: "start" });
}
// Zoomed before/after of the changed area: small changes (eye fixes) are invisible on the full picture at phone size.
async function showChange(beforeUrl, afterUrl, box) {
  try {
    const [a, b] = await Promise.all([loadImg(beforeUrl), loadImg(afterUrl)]);
    let [x0, y0, x1, y1] = box; const pw = (x1 - x0) * 0.25 + 16, ph = (y1 - y0) * 0.25 + 16;
    x0 = Math.max(0, x0 - pw); y0 = Math.max(0, y0 - ph); x1 = Math.min(b.naturalWidth, x1 + pw); y1 = Math.min(b.naturalHeight, y1 + ph);
    if ((x1 - x0) * (y1 - y0) > 0.6 * b.naturalWidth * b.naturalHeight) return;   // big change: the main preview shows it
    const mk = (im, t) => { const c = document.createElement("canvas"); c.width = x1 - x0; c.height = y1 - y0;
      c.getContext("2d").drawImage(im, x0, y0, c.width, c.height, 0, 0, c.width, c.height);
      return `<figure style="margin:0;flex:1"><img src="${c.toDataURL("image/png")}" style="width:100%;image-rendering:auto;border-radius:6px"><figcaption class="small muted">${t}</figcaption></figure>`; };
    $("editOut").insertAdjacentHTML("beforeend", `<div class="small muted">Changed area (zoomed):</div><div class="row" style="gap:6px;align-items:flex-start">${mk(a, "Before")}${mk(b, "After")}</div>`);
  } catch (e) {}
}
$("eraseGo").onclick = () => { const m = maskB64(); if (!m) return notice("Paint over what to erase first (tick Paint)."); editJob({ type: "erase", image: curB64(), mask: m }, "AI erase"); };
$("editBg").onclick = () => editJob({ type: "remove_bg", image: curB64() }, "Remove background");
$("editStr").oninput = () => $("stV").textContent = (+$("editStr").value).toFixed(2);
$("editGrow").oninput = () => $("grV").textContent = (+$("editGrow").value > 0 ? "+" : "") + $("editGrow").value;
$("editWhole").onchange = () => { $("editStr").value = $("editWhole").checked ? 0.4 : 1; $("editStr").oninput(); };
["editFixEyes", "editFixFace", "editFixBody"].forEach(id => $(id) && ($(id).onclick = () => { $("editInstr").value = { editFixEyes: "fix eyes", editFixFace: "fix face", editFixBody: "fix arms, hands, legs and feet" }[id]; $("editGo").click(); }));
$("editGo").onclick = () => {
  const instr = $("editInstr").value.trim(); if (!instr) return $("editInstr").focus();
  const m = $("editWhole").checked ? null : maskB64();
  editJob({ type: "change", image: curB64(), instruction: instr, method: $("editMethod").value, strength: +$("editStr").value,
            grow: +$("editGrow").value, whole: $("editWhole").checked, ...(m ? { mask: m } : {}), ...(ED.gen ? { gen_meta: ED.gen } : {}) }, "AI Change");
};
$("editRegen").onclick = async () => {
  const instr = $("editInstr").value.trim(); if (!ED.gen) return; if (!instr) return notice("Type the change first (e.g. 'a red sweater dress', 'beach background', 'watercolor style').");
  const r = await runJob({ type: "regen", instruction: instr, gen_meta: ED.gen, upscaler: "sharp" }, $("editJob"), "Regenerate", { done: (r, s) => `Regenerated in ${s} s` });
  if (r) { await editLoad("data:image/png;base64," + r.image, true); setGen(r.gen_meta || ED.gen); $("editOut").innerHTML = `<p class="muted small">Regenerated from the original Gen prompt + your change (Undo goes back).</p>`; }
};
$("editSave").onclick = () => ED.cur && saveImg(curB64(), `npustudio-edit-${Date.now()}.png`);
$("editShare").onclick = () => ED.cur && shareImg(curB64(), `npustudio-edit-${Date.now()}.png`);
$("editToUp").onclick = () => { if (ED.cur) { upLoad(ED.cur); tabTo("up"); } };

// ---------- upscale ----------
let upSrc = null, upDim = null, upScale = 4;
async function upLoad(d) { upSrc = d; $("upIn").src = d; $("upIn").classList.remove("hidden"); $("upOut").innerHTML = ""; const i = await loadImg(d); upDim = [i.naturalWidth, i.naturalHeight]; upEstimate(); }
document.querySelectorAll("#upScale button").forEach(b => b.onclick = () => {
  document.querySelectorAll("#upScale button").forEach(x => x.classList.toggle("on", x === b)); upScale = +b.dataset.v; upEstimate();
});
$("upPick").onclick = async () => { const d = await pickImage(); if (d) upLoad(d); };
let upChk = null;
// "Limit to my screen": cap the result at the phone's native resolution (long edge vs long edge, so either orientation fits).
$("upScreen").checked = localStorage.getItem("npuUpScreen") !== "0";
$("upScreen").onchange = () => { localStorage.setItem("npuUpScreen", $("upScreen").checked ? "1" : "0"); upEstimate(); };
function screenPx() { const d = window.devicePixelRatio || 1, a = Math.round(screen.width * d), b = Math.round(screen.height * d); return [Math.max(a, b), Math.min(a, b)]; }
$("upScreenSz").textContent = "(" + screenPx().join("×") + ")";
// -> {scale, w, h, limited, already}: the smallest model pass that reaches the screen size; the server resizes the result to fit.
const PHONE_SIDE = 16384, PHONE_MP = 160;
// -> {scale (model passes to run), w, h (true final size), limited, capped, already}. Same rules as the server's phone_cap().
function upPlan() {
  const sc = upScale; if (!upDim) return null;
  const [w, h] = upDim;
  let f = Math.min(sc, PHONE_SIDE / Math.max(w, h), Math.sqrt(PHONE_MP * 1e6 / (w * h))); const capped = f < sc;
  let limited = false;
  if ($("upScreen").checked) {
    const [L, S] = screenPx(), fs = Math.min(L / Math.max(w, h), S / Math.min(w, h));
    if (fs <= 1) return { scale: 0, w, h, limited: true, capped: false, already: true };
    if (fs < f) { f = fs; limited = true; }
  }
  if (f >= sc) return { scale: sc, w: w * sc, h: h * sc, limited: false, capped: false, already: false };
  const run = [2, 4, 8, 16].find(x => x >= f - 1e-9 && x <= sc) || sc;
  return { scale: run, w: Math.min(PHONE_SIDE, Math.round(w * f)), h: Math.min(PHONE_SIDE, Math.round(h * f)), limited, capped: capped && !limited, already: false };
}
let upSeq = 0;
async function upEstimate() {
  const seq = ++upSeq, p = upPlan();
  $("upPlan").innerHTML = !p ? "" : p.already
    ? `⚠ This picture (${p.w}×${p.h}) is already at or above your screen size. <a href="#" id="upOff">Turn off the screen limit</a> to upscale anyway.`
    : `<b>${upScale}× → ${p.w}×${p.h}</b>${p.limited ? " (screen limit" : p.capped ? " (phone max" : ""}${(p.limited || p.capped) ? (p.scale < upScale ? `, runs ${p.scale}× then resizes)` : ", resized after upscaling)") : ""}`;
  if ($("upOff")) $("upOff").onclick = e => { e.preventDefault(); $("upScreen").checked = true; $("upScreen").click(); };
  $("upEst").innerHTML = "";
  if (!upDim) { $("upInfo").textContent = ""; return; }
  $("upInfo").textContent = `Picture ${upDim[0]}×${upDim[1]}`;
  if (!p || p.already) return;
  let c; try { c = await api("upscale/check", { width: upDim[0], height: upDim[1], scale: upScale }); } catch (e) { return; }
  if (seq !== upSeq) return;            // a newer scale/model/limit change already redrew this
  upChk = c;
  $("upEst").className = "small " + (c.ok ? "muted" : "warn");
  $("upEst").innerHTML = c.ok ? (c.full_on_desktop ? `Full size ${c.out_width}×${c.out_height} (${c.megapixels} MP) is saved on the desktop; about ${c.memory_gb} GB of the ${c.memory_limit_gb} GB limit${c.scale >= 8 ? " · 2 passes" : ""}.`
                                     : `⚠ ${esc(c.full_note)}.`) + (c.capped ? ` Phone gets ${c.final_width}×${c.final_height}.` : "")
                              : `⚠ ${esc(c.reason)}. Pick a smaller scale.`;
}
["upModel", "upDev", "upFmt"].forEach(id => $(id).addEventListener("change", upEstimate));
$("upGo").onclick = async () => {
  if (!upSrc) return notice("Pick a photo first.");
  const p = upPlan();
  if (p.already) return notice("This picture is already at or above your screen size. Turn off 'Limit to my screen' to upscale it anyway.");
  if (upChk && !upChk.ok) return notice(upChk.reason + ". Pick a smaller scale.");
  const sc = upScale, fin = p.w !== upDim[0] * sc || p.h !== upDim[1] * sc;
  $("upOut").innerHTML = "";
  const r = await runJob(async () => { const im = await shrink(upSrc, Math.max(...upDim), "image/png");
      const b = { type: "upscale", image: im.b64, scale: sc, model: $("upModel").value, device: $("upDev").value, format: $("upFmt").value };
      if (fin) { b.max_w = p.w; b.max_h = p.h; }
      return b; },
    $("upJob"), `Upscale ${sc}× → ${p.w}×${p.h}` + (p.limited ? " (screen)" : p.capped ? " (phone max)" : ""),
    { btn: $("upGo"), done: (r, s) => `Done in ${s} s on ${r.device || "?"} → ${r.width}×${r.height} ${r.format || ""}${r.image ? " (" + fmtMB(r.image.length * 0.75) + ")" : ""}${r.ran_scale && r.ran_scale !== sc ? ` · ran ${r.ran_scale}×` : ""}` });
  if (r) {
    const ext = r.format === "JPEG" ? "jpg" : "png";
    showResult($("upOut"), r, "npustudio-upscale", `· ${r.width}×${r.height} ${r.format || ""}`);
    $("upOut").querySelectorAll("button").forEach(b => { b.onclick = () => (b.classList.contains("sv") ? saveImg : shareImg)(r.image, `npustudio-upscale-${sc}x-${Date.now()}.${ext}`); });
  }
};

// ---------- connection ----------
const SV = typeof Shell?.getConnection === "function";    // shell v2+ can switch home/remote hosts
let LAT = null;
async function ping() {
  const t = performance.now(); await api("health"); LAT = Math.round(performance.now() - t); return LAT;
}
function hostKind(h) { return /^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\./.test(h) || h.endsWith(".ts.net") ? "Remote (Tailscale)" : "Home (Wi-Fi)"; }
function connPill(ok) {
  const p = $("conn"); p.className = "pill " + (ok ? "ok" : "bad");
  p.textContent = ok ? `🔗 ${hostKind(location.hostname).split(" ")[0]} · ${LAT} ms` : "🔗 Offline";
}
async function connPage() {
  let st = "offline"; try { await ping(); st = `online, ${LAT} ms`; } catch (e) {}
  const c = SV ? JSON.parse(Shell.getConnection()) : null;
  $("connBox").innerHTML = `<b>Connected to</b><span>${esc(location.hostname)}:${esc(location.port)}</span><b>Type</b><span>${hostKind(location.hostname)}</span>
    <b>Status</b><span>${st}</span><b>App</b><span>${SV ? "Android app v" + shellVersion : (Shell ? "Android app v" + shellVersion + " (update for home/remote switching)" : "browser")}</span>`;
  $("connShell").classList.toggle("hidden", !SV);
  if (c) { $("cHome").value = c.home || ""; $("cRemote").value = c.remote || ""; $("cMode").value = c.mode || "auto"; }
  $("connNote").textContent = SV ? "Auto tries the home address first and switches to the remote (Tailscale) address when home can't be reached. Tailscale must be on for remote."
    : (Shell ? "Install the new NpuStudio-Companion.apk to save a home and a remote address and switch between them." : "In a browser, open this page at the address you want to use.");
}
$("conn").onclick = () => { document.querySelectorAll("#tabs button").forEach(x => x.classList.remove("on")); document.querySelectorAll("main .page").forEach(p => p.classList.toggle("hidden", p.id !== "tab-conn")); clearInterval(monTimer); connPage(); };
$("cSave").onclick = () => { const r = Shell.setConnection($("cHome").value.trim(), $("cRemote").value.trim(), $("cMode").value); if (r) notice(r); };
$("cRepair").onclick = () => Shell.rePair();
$("cTest").onclick = () => { $("connNote").textContent = "Testing…"; const id = "c" + (++cbSeq);
  cbs[id] = d => { const r = JSON.parse(d || "{}"); $("connNote").textContent = Object.entries(r).map(([k, v]) => `${k}: ${v == null ? "unreachable" : v + " ms"}`).join("   ·   ") || "Nothing saved."; };
  Shell.probe(id); };

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
(function bgSetup() {          // shell v3: the phone's background link (Find my phone)
  if (!(Shell && Shell.getBackground)) return;
  $("bgBox").classList.remove("hidden");
  const show = () => { try { const b = JSON.parse(Shell.getBackground()); $("bgLink").checked = b.on;
    $("bgInfo").textContent = !b.on ? "Off: the PC can't ring this phone." : b.last_ok ? `Connected to the desktop ${Math.round((Date.now() - b.last_ok) / 1000)} s ago. Uses very little battery (one check-in about every 25 s).` : "Starting…"; } catch (e) {} };
  $("bgLink").onchange = () => { Shell.setBackground($("bgLink").checked); setTimeout(show, 1500); }; show(); setInterval(show, 15000);
})();
$("unpair").onclick = async () => { if (!(await ask("Forget the pairing key?"))) return; localStorage.removeItem("npuKey"); if (Shell && Shell.unpair) Shell.unpair(); else location.reload(); };

// ---------- start ----------
async function start() {
  shellUpdateCheck();
  if (!key()) { $("pair").classList.remove("hidden"); return; }
  try {
    await ping(); connPill(true); loadOptions().then(genPlan);
    $("pair").classList.add("hidden"); $("main").classList.remove("hidden"); $("tabs").classList.remove("hidden");
    try { const v = await (await fetch("version.json", { cache: "no-store" })).json(); if (Shell && v.min_shell > shellVersion) notice("Please update the NpuStudio Companion app (this UI needs shell v" + v.min_shell + ")."); } catch (e) {}
    if (window.__startTab) { const b = document.querySelector(`#tabs button[data-t="${window.__startTab}"]`); if (b) b.click(); }
    if (!Shell && window.Notification && Notification.permission === "default") Notification.requestPermission();
  } catch (e) {
    if (e.status === 403) { $("pairMsg").textContent = "That key was refused."; $("pair").classList.remove("hidden"); $("main").classList.add("hidden"); $("tabs").classList.add("hidden"); }
    else connPill(false);
  }
}
$("pairBtn").onclick = () => { localStorage.setItem("npuKey", $("pairKey").value.trim()); $("pairMsg").textContent = ""; start(); };
setInterval(async () => { if (!key()) return; try { await ping(); connPill(true); } catch (e) { connPill(false); if (SV && Shell.fallback) Shell.fallback(); } }, 10000);
start();
