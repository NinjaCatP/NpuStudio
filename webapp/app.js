// NpuStudio Companion web UI. Served by the desktop at /app; the Android shell (window.NpuShell) is optional.
"use strict";
const $ = id => document.getElementById(id);
const UI_VERSION = "1.9.7";
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
// Android app versionName per shell number (from android/app/build.gradle.kts); the shell only exposes its number
const SHELL_APP = { 3: "2.1", 4: "2.2", 5: "2.3", 6: "2.4", 7: "2.5" };
const SHELL_BLURB = {
  3: "adds brightness, Do Not Disturb, ring modes, calls/texts, maps, notes, read aloud and Find my phone",
  7: "moves Heat to the separate Heat Guard app. Pairing, Monitor, Chat, and the other tabs stay here"
};
function appVersion() { try { if (Shell && Shell.appVersion) return String(Shell.appVersion()); } catch (e) {} return shellVersion ? (SHELL_APP[shellVersion] || "") : ""; }
const MODELS = { "SM-S928": "Galaxy S24 Ultra", "SM-S926": "Galaxy S24+", "SM-S921": "Galaxy S24", "SM-S918": "Galaxy S23 Ultra", "SM-S938": "Galaxy S25 Ultra" };
function phoneModel() {
  const m = navigator.userAgent.match(/Android [\d.]+; ([^;)]+?)(?: Build|\))/); if (!m) return "";
  const k = Object.keys(MODELS).find(x => m[1].startsWith(x)); return k ? MODELS[k] : m[1].trim();
}
function devHeaders() {
  if (!/Android/.test(navigator.userAgent)) return {};
  let id = null, name = null;
  try { if (typeof Shell !== "undefined" && Shell && Shell.getBackground) { const b = JSON.parse(Shell.getBackground()); id = b.id; name = b.name; } } catch (e) {}
  if (!id) { id = localStorage.getItem("npuDevId") || ("w" + Math.random().toString(36).slice(2, 9)); localStorage.setItem("npuDevId", id); }
  const h = { "X-Device-Id": id }; if (name) h["X-Device-Name"] = name;
  try { if (typeof shellVersion !== "undefined" && shellVersion) h["X-Shell-Version"] = String(shellVersion); } catch (e) {}
  try { h["X-UI-Version"] = UI_VERSION; const a = appVersion(); if (a) h["X-App-Version"] = a; const m = phoneModel(); if (m) h["X-Phone-Model"] = m; } catch (e) {}
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
function ask(q, okOnly, yes, no) {
  return new Promise(res => {
    $("dlgQ").textContent = q; $("dlg").classList.remove("hidden");
    $("dlgNo").classList.toggle("hidden", !!okOnly); $("dlgYes").textContent = okOnly ? "OK" : (yes || "Yes"); $("dlgNo").textContent = no || "No";
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
// ---------- 1.8.7: "Save as" PNG / JPG / WebP (shared, remembered) ----------
const SAVEF = { png: ["image/png", "png"], jpg: ["image/jpeg", "jpg"], webp: ["image/webp", "webp"] };
const saveFmt = () => SAVEF[localStorage.getItem("saveFmt")] ? localStorage.getItem("saveFmt") : "png";
const saveQ = () => Math.max(30, Math.min(100, +localStorage.getItem("saveQ") || 92));
function saveFmtUI() {
  const f = saveFmt();
  return `<span class="saveAs" style="display:inline-flex;gap:4px;align-items:center;font-size:13px">Save as <select class="saveFmt">${["png", "jpg", "webp"].map(k => `<option value="${k}"${k === f ? " selected" : ""}>${k.toUpperCase()}</option>`).join("")}</select>
    <span class="saveQw${f === "png" ? " hidden" : ""}">quality <input class="saveQ" type="number" min="30" max="100" value="${saveQ()}" style="width:52px"></span></span>`;
}
document.addEventListener("change", e => {
  if (e.target.classList.contains("saveFmt")) localStorage.setItem("saveFmt", e.target.value);
  else if (e.target.classList.contains("saveQ")) localStorage.setItem("saveQ", String(Math.max(30, Math.min(100, +e.target.value || 92))));
  else return;
  document.querySelectorAll(".saveFmt").forEach(s => s.value = saveFmt());
  document.querySelectorAll(".saveQ").forEach(s => s.value = saveQ());
  document.querySelectorAll(".saveQw").forEach(s => s.classList.toggle("hidden", saveFmt() === "png"));
});
function b64Type(b64) { const h = atob(b64.slice(0, 24)); if (h.startsWith("\x89PNG")) return "png"; if (h.startsWith("\xff\xd8")) return "jpg"; if (h.slice(8, 12) === "WEBP") return "webp"; return "png"; }
async function convertB64(b64, fmt, q) {   // in-browser canvas re-encode so the bytes really are the chosen type
  const src = b64Type(b64); if (src === fmt && fmt === "png") return b64;
  const im = await loadImg(`data:image/${src === "jpg" ? "jpeg" : src};base64,` + b64);
  if (im.naturalWidth * im.naturalHeight > 60e6) throw new Error("too big to convert on the phone");
  const c = document.createElement("canvas"); c.width = im.naturalWidth; c.height = im.naturalHeight; const x = c.getContext("2d");
  if (fmt === "jpg") { x.fillStyle = "#fff"; x.fillRect(0, 0, c.width, c.height); }   // JPG has no transparency: white behind
  x.drawImage(im, 0, 0); const u = c.toDataURL(SAVEF[fmt][0], q / 100);
  if (!u.startsWith("data:" + SAVEF[fmt][0])) throw new Error(fmt.toUpperCase() + " isn't supported by this browser");
  return u.split(",")[1];
}
async function saveImg(b64, name, opts = {}) {
  let fmt = opts.fmt || saveFmt();
  if (opts.alpha && fmt === "jpg" && !opts.quiet) fmt = await ask("This picture has a transparent background. JPG can't keep transparency (it becomes white). Save as PNG instead?", false, "PNG", "JPG anyway") ? "png" : "jpg";
  let out = b64;
  try { out = await convertB64(b64, fmt, saveQ()); }
  catch (e) { fmt = b64Type(b64); if (!opts.quiet) galToast?.(`Saved as ${fmt.toUpperCase()} (${e.message})`, 4000); }
  name = String(name || "npustudio").replace(/\.(png|jpe?g|webp)$/i, "") + "." + SAVEF[fmt][1];
  if (Shell && Shell.saveImage) { Shell.saveImage(out, name); return name; }
  const a = document.createElement("a"); a.href = `data:${SAVEF[fmt][0]};base64,` + out; a.download = name; a.click(); return name;
}
async function shareImg(b64, name) {
  if (Shell && Shell.share) { Shell.share(b64, name); return; }
  try { const blob = await (await fetch("data:image/png;base64," + b64)).blob(); await navigator.share({ files: [new File([blob], name, { type: blob.type })] }); }
  catch (e) { saveImg(b64, name); }
}
function showResult(box, r, base, extra = "") {
  const name = `${base}-${new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19)}.png`;
  box.innerHTML = `<img src="data:image/png;base64,${r.image}"><div class="row"><button class="primary grow sv">💾 Save</button><button class="grow sh">📤 Share</button></div><div class="row">${saveFmtUI()}</div>
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
  pc: "Commands run on that PC: apps, volume, brightness, media, screenshot, lock, files, recent files, clipboard, system info, Wi-Fi/Bluetooth/battery saver/night light settings, empty the Recycle Bin, find my phone, open an app and click a button in it ('open Comfy Desktop and click Local'). Risky ones ask you here first."
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
      const latest = v.latest_shell || v.min_shell || 1;
      const name = SHELL_APP[latest] || latest;
      const blurb = SHELL_BLURB[latest] || "adds new phone controls";
      $("shellUpd").innerHTML = `📲 An Android app update (v${name}) ${blurb}. Install <b>NpuStudio-Companion.apk</b> over this one: open <a href="${u}" target="_blank">${u}</a> in Chrome on this phone, or copy it from H:\\downloads on the desktop. Your pairing is kept.`;
      $("shellUpd").classList.remove("hidden");
    }
  } catch (e) {}
}
$("chatSend").onclick = chatSend;
$("chatIn").onkeydown = e => { if (e.key === "Enter") chatSend(); };
$("chatNew").onclick = async () => { try { await api("chat/new", {}); } catch (e) {} $("chatLog").innerHTML = ""; };

// ---------- image gen ----------
let OPTS = null;
// newer web UI on the server than this loaded page (WebView cache): reload once, bypassing the cache
(async function () {
  try {
    const v = await (await fetch("version.json?t=" + Date.now(), { cache: "no-store" })).json();
    if (v.ui_version && v.ui_version !== UI_VERSION && sessionStorage.getItem("npuReloadFor") !== v.ui_version) {
      sessionStorage.setItem("npuReloadFor", v.ui_version);
      if (navigator.serviceWorker && navigator.serviceWorker.getRegistrations) { try { (await navigator.serviceWorker.getRegistrations()).forEach(r => r.unregister()); } catch (e) {} }
      if (window.caches && caches.keys) { try { (await caches.keys()).forEach(k => caches.delete(k)); } catch (e) {} }
      const u = new URL(location.href); u.searchParams.set("v", v.ui_version); location.replace(u.toString());
    }
  } catch (e) {}
})();
async function loadOptions() {
  try { OPTS = await api("options"); } catch (e) { return; }
  // only offer Device choices this server says it supports (an older server just gets Auto + what it lists)
  // Device boxes: options the server doesn't list are greyed "(update server)"; laptop chips "(offline)" when it isn't reachable
  const lapOn = OPTS.laptop_online !== false, oldSrv = !OPTS.server_code;
  // an old server process has no gen/edit lists either: a missing list then means "not supported", so every box agrees
  [["upDev", OPTS.upscale_devices], ["genDev", OPTS.gen_devices], ["editDev", OPTS.edit_devices], ["anaDev", OPTS.analyze_devices]].forEach(([id, list]) => {
    const e = $(id); if (!e) return;
    [...e.options].forEach(o => {
      if (!o.dataset.t) o.dataset.t = o.textContent;
      const known = list ? list.includes(o.value) : (!oldSrv || !o.value.startsWith("Laptop") && o.value !== "Both machines"), lap = o.value.startsWith("Laptop") || o.value === "Both machines";
      const notEdit = id === "editDev" && o.value === "Both machines";   // nothing sensible to split for Remove background / AI erase
      const off = o.value !== "Auto" && (notEdit || !known || (lap && !lapOn));
      o.disabled = off; o.textContent = o.dataset.t + (notEdit ? " (not for Edit - one picture runs on one chip)" : off ? (!known ? (oldSrv ? " (old server still running)" : " (update server)") : " (offline)") : "");
    });
    if (e.selectedOptions[0] && e.selectedOptions[0].disabled) e.value = "Auto";
  });
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
function chipShort(n) { const m = /^(Desktop|Laptop)\s+(.*)$/.exec(n || ""); if (!m) return n; const s = m[1][0], c = m[2]; const r = /RTX\s*(\d{4}\s*(Ti|SUPER)?)/i.exec(c); if (r) return r[1].replace(/\s+/g, "") + s; if (/UHD/i.test(c)) return "UHD " + s; if (/Arc/i.test(c)) return "Arc " + s; if (/NPU/i.test(c)) return "NPU " + s; if (/CPU/i.test(c)) return "CPU " + s; return c.split(" ")[0] + " " + s; }
const GT = { sel: null, got: {}, el: {} };   // 1.8.1: multi-chip Gen tabs
function gtReset() { GT.sel = null; GT.got = {}; GT.el = {}; const o = $("genOut"); o.innerHTML = '<div class="row gtabs" style="flex-wrap:wrap;gap:4px"></div><div class="gpanes"></div>'; o.style.display = ""; }
function gtSelect(n) { GT.sel = n; Object.entries(GT.el).forEach(([k, e]) => { e.pane.classList.toggle("hidden", k !== n); e.btn.classList.toggle("primary", k === n); }); }
function gtChip(n, x, w, h) {
  const o = $("genOut"); let e = GT.el[n];
  if (!e) {
    const btn = document.createElement("button"); btn.className = "small"; btn.onclick = () => gtSelect(n); o.querySelector(".gtabs").appendChild(btn);
    const pane = document.createElement("div"); pane.className = "hidden"; pane.innerHTML = '<p class="small st"></p><div class="pic"></div>'; o.querySelector(".gpanes").appendChild(pane);
    e = GT.el[n] = { btn, pane }; if (!GT.sel) gtSelect(n);
  }
  const ic = { done: "✅", running: "⏳", queued: "🕓", error: "⚠️" }[x.state] || "⛔";
  e.btn.textContent = `${ic} ${chipShort(n)}`; e.btn.title = n;
  const pct = x.n ? Math.round(100 * x.k / x.n) : 0;
  e.pane.querySelector(".st").innerHTML = `<b>${esc(n)}</b> · ${esc(x.state)}${x.state === "running" ? ` · ${pct}%` + (x.eta_s != null ? ` · ~${Math.round(x.eta_s)} s left` : "") : ""}${x.seed ? ` · seed ${x.seed}` : ""}${x.secs ? ` · ${x.secs} s` : ""}<br><span class="dim">${esc((x.msg || "").slice(0, 100))}</span>`;
}
function gtPic(n, p, w, h) {
  const e = GT.el[n]; if (!e || GT.got[n]) return; GT.got[n] = 1;
  const d = e.pane.querySelector(".pic"); showResult(d, p, "npustudio-gen-" + n.replace(/\W+/g, "-"), `· ${w}×${h} · seed ${p.seed}`); addSend(d, p.image, p.gen_meta);
  const im = d.querySelector("img"); if (im) im.onclick = () => { const wv = window.open(); if (wv) wv.document.write(`<img src="${im.src}" style="max-width:100%">`); };
  const r = document.createElement("div"); r.className = "row"; r.innerHTML = `<button class="grow">🎲 Use seed ${p.seed}</button>`; r.children[0].onclick = () => { $("genSeed").value = p.seed; notice("Seed set to " + p.seed); }; d.appendChild(r);
  if (!Object.keys(GT.got).some(k => k !== n) && GT.sel !== n && !GT.picked) { GT.picked = 1; gtSelect(n); }   // first finished auto-selects
}
$("genGo").onclick = async () => {
  const p = $("genPrompt").value.trim(); if (!p) return $("genPrompt").focus();
  const [w, h] = genTarget();
  if (!(w >= 64 && h >= 64 && w <= 16384 && h <= 16384)) return notice("Size must be between 64 and 16384 on each side.");
  const multi = ["Both machines", "All chips"].includes($("genDev").value); let jid = null;
  if (multi) { gtReset(); GT.picked = 0; }
  const r = await runJob({ type: "gen", prompt: p, width: w, height: h, steps: Math.min(50, +$("genSteps").value || 50), device: $("genDev").value, seed: +$("genSeed").value || 0, random_seeds: $("genRandEach").checked, skip_slow: $("genSkipSlow").checked, upscaler: $("genUp").value, autofix: $("genFix").checked }, $("genJob"), "Image Gen",
    { btn: $("genGo"), onUpdate: s => { if (!multi || !s.chips) return; jid = s.id;
        Object.entries(s.chips).forEach(([n, x]) => { gtChip(n, x, w, h);
          if (x.state === "done" && !GT.got[n] && x.i >= 0 && !GT["f" + n]) { GT["f" + n] = 1; api(`jobs/${jid}/pic/${x.i}`).then(pp => gtPic(n, { ...pp, seconds: x.secs }, w, h)).catch(() => { GT["f" + n] = 0; }); } }); },
      done: (r, s) => `Done in ${s} s · ${w}×${h}${r.device ? " on " + r.device : ""}` });
  if (r && multi && r.images) {
    if (r.chips) Object.entries(r.chips).forEach(([n, x]) => gtChip(n, x, w, h));
    r.images.forEach(pp => { if (!GT.el[pp.chip]) gtChip(pp.chip, { state: "done", seed: pp.seed }, w, h); gtPic(pp.chip, { ...pp, seconds: r.chips && r.chips[pp.chip] ? r.chips[pp.chip].secs : r.seconds }, w, h); });
    if (r.note) $("genOut").insertAdjacentHTML("beforeend", `<p class="small err">${esc(r.note)}</p>`);
  } else if (r) { $("genOut").style.display = ""; showResult($("genOut"), r, "npustudio-gen", `· ${w}×${h}`); addSend($("genOut"), r.image, r.gen_meta); if (r.note) $("genOut").insertAdjacentHTML("beforeend", `<p class="small err">${esc(r.note)}</p>`); }
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
  $("anaDesc").value = r.description || ""; $("anaPrompt").value = r.prompt || ""; $("anaSize").textContent = `Size: ${r.width}×${r.height}`; $("anaPose").textContent = "Pose: " + (r.pose || "-"); $("anaGo").textContent = "🔄 Re-analyze"; if (r.note) notice(r.note);
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
  if (!fromResult) ED.alpha = false;
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
// 1.8.8: Remove social media overlay - auto-mask onto the brush layer (editable), then LaMa fill on the CPU
function paintMaskB64(b64) {
  return loadImg("data:image/png;base64," + b64).then(m => {
    const cv = $("maskCv"), x = cv.getContext("2d"), t = document.createElement("canvas"); t.width = cv.width; t.height = cv.height;
    const tx = t.getContext("2d"); tx.drawImage(m, 0, 0, t.width, t.height);
    const d = tx.getImageData(0, 0, t.width, t.height), o = x.getImageData(0, 0, cv.width, cv.height); strokes.push(x.getImageData(0, 0, cv.width, cv.height));
    for (let i = 0; i < d.data.length; i += 4) if (d.data[i] > 127) { o.data[i] = 0; o.data[i + 1] = 220; o.data[i + 2] = 255; o.data[i + 3] = 255; }
    x.putImageData(o, 0, 0); cv.dataset.dirty = "1"; $("brOn").checked = true; $("brOn").onchange();
  });
}
async function ovFind() {
  if (!ED.cur) return notice("Pick a screenshot first.");
  const r = await runJob({ type: "overlay_find", image: curB64(), text: $("ovText").checked, icons: $("ovIcons").checked, bars: $("ovBars").checked, shade: $("ovShade").checked }, $("editJob"), "Find overlay",
    { done: r => r.found ? "Overlay found" + (r.image ? " (bars cropped / top shading removed)" : "") : "No overlay found" });
  if (!r) return false;
  if (r.image) { const url = "data:image/png;base64," + r.image; await new Promise(res => { pushEdit(url); const i = $("editIn"); const f = i.onload; i.onload = () => { f && f(); res(); }; }); }
  else clearMask();
  if (r.found) await paintMaskB64(r.mask);
  $("editOut").innerHTML = `<p class="muted small">${r.found ? "Cyan = what will be removed. Fix it with ➕ Add / ➖ Erase (Brush mask section), then tap ✨ Remove overlay." : "Nothing found - paint over the overlay with the brush, then tap ✨ Remove overlay."}</p>`;
  return true;
}
$("ovFind").onclick = ovFind;
$("ovGo").onclick = async () => {
  if (!ED.cur) return notice("Pick a screenshot first.");
  if (!$("maskCv").dataset.dirty && !await ovFind()) return;
  const m = maskB64(); if (!m) return notice("Nothing to remove - paint over the overlay first.");
  editJob({ type: "overlay_remove", image: curB64(), mask: m, hq: $("ovHQ").checked ? true : "auto" }, "Remove overlay");
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
  pushEdit(url); ED.alpha = body.type === "remove_bg" ? true : ED.alpha;   // 1.8.7: transparent result -> PNG by default
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
$("eraseGo").onclick = () => { const m = maskB64(); if (!m) return notice("Paint over what to erase first (tick Paint)."); editJob({ type: "erase", image: curB64(), mask: m, device: $("editDev").value }, "AI erase"); };
$("editBg").onclick = $("editBg2").onclick = () => editJob({ type: "remove_bg", image: curB64(), device: $("editDev").value === "Both machines" ? "Auto" : $("editDev").value }, "Remove background");   // 1.8.6: own section too
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
$("editSave").onclick = () => ED.cur && saveImg(curB64(), `npustudio-edit-${Date.now()}.png`, { alpha: !!ED.alpha });
if ($("editSaveAs")) $("editSaveAs").innerHTML = saveFmtUI();
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
const mb = v => typeof v === "number" ? (v / 1024).toFixed(1) + " GB" : na(v);   // 1.8.5: GB, one decimal
function chipLabel(mach, chip) {   // 1.8.3: 'NPU L', 'Arc L', '4070L (laptop RTX 4070)', '4070D (desktop RTX 4070)', 'UHD D (desktop Intel UHD 770)'
  const m = mach[0];
  if (chip === "NPU") return "NPU " + m;
  if (chip === "NVIDIA") return `4070${m} (${mach.toLowerCase()} RTX 4070)`;
  if (chip === "Intel Arc") return mach === "Desktop" ? "UHD D (desktop Intel UHD 770)" : "Arc " + m;
  return chip + " " + m;
}
const CHIP_COL = { "Desktop:NVIDIA": "#ff9f1a", "Desktop:Intel Arc": "#ff5a5a", "Laptop:NVIDIA": "#76b900", "Laptop:Intel Arc": "#87cefa", "Laptop:NPU": "#a970ff" };   // same as the PC apps
const CHIP_ORD = ["NVIDIA", "Intel Arc", "NPU"];
const CHIP_NAME = { "Desktop:NVIDIA": "NVIDIA GeForce RTX 4070", "Desktop:Intel Arc": "Intel UHD Graphics 770", "Laptop:NVIDIA": "NVIDIA GeForce RTX 4070 Laptop GPU", "Laptop:Intel Arc": "Intel Arc Graphics", "Laptop:NPU": "Intel AI Boost NPU" };
const MON_HIST = {};
function spark(k, v, col) {   // 1.8.4: last 40 readings of Use %
  const h = MON_HIST[k] = (MON_HIST[k] || []); if (typeof v === "number") { h.push(v); if (h.length > 40) h.shift(); }
  if (h.length < 2) return "";
  const pts = h.map((y, i) => `${(i * 200 / 39).toFixed(1)},${(30 - Math.min(100, y) * 0.28).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 200 30" preserveAspectRatio="none" style="width:100%;height:30px;display:block;margin-top:4px"><polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2"/></svg>`;
}
const MON_SEEN = new Set();
function chipRows(chips, mach, names) {   // 1.8.4: every chip card uses the same layout as the desktop RTX card
  const u = (v, s) => typeof v === "number" ? `${Math.round(v * 10) / 10}${s}` : "N/A";
  const g = v => (v / 1024).toFixed(1);
  const mem = c => { const d = typeof c.ded === "number", s = typeof c.shared === "number";   // 1.8.5: GB, one decimal
    if (d && c.ded_total) return `${g(c.ded)} / ${g(c.ded_total)} GB` + (s && c.shared >= 51 ? ` (+${g(c.shared)} GB shared)` : "");
    const p = []; if (d) p.push(`${g(c.ded)} GB dedicated`); if (s) p.push(`${g(c.shared)} GB shared`); return p.length ? p.join(" · ") : "N/A"; };
  return Object.entries(chips || {}).filter(([n, c]) => !n.startsWith("_") && c && typeof c === "object").filter(([n]) => { const id = mach + ":" + n; if (MON_SEEN.has(id)) return false; MON_SEEN.add(id); return true; })   // one card per stable chip id
    .sort((a, b) => CHIP_ORD.indexOf(a[0]) - CHIP_ORD.indexOf(b[0])).map(([n, c]) => {
    const k = mach + ":" + n, col = CHIP_COL[k] || "#888", nm = (names && names[n]) || CHIP_NAME[k] || n;
    return `<div style="border-left:4px solid ${col};background:#ffffff08;border-radius:8px;padding:6px 10px;margin:8px 0"><h3 style="color:${col};margin:2px 0 4px">${esc(chipLabel(mach, n))}</h3>
     <div class="kv"><b>Chip</b><span>${esc(nm)}</span><b>Use</b><span>${u(c.use, " %")}</span><b>Memory</b><span>${mem(c)}</span>
     <b>Power</b><span>${u(c.watts, " W")}</span><b>Temp</b><span>${u(c.temp, " °C")}</span><b>Jobs running</b><span>${typeof c.jobs === "number" ? c.jobs : "N/A"}</span></div>${spark(k, c.use, col)}</div>`; }).join("");
}
async function monitor() {
  try {
    const m = await api("monitor"), d = m.desktop; MON_SEEN.clear();
    let h = `<h2>🖥️ Desktop · ${esc(d.name || "")}</h2><div class="kv"><b>Jobs running</b><span>${d.busy}</span></div>`;
    h += chipRows(m.chips, "Desktop");
    const others = Object.entries(m.others || {}).sort((a, b) => (b[1]._t || 0) - (a[1]._t || 0)).slice(0, 1);   // one laptop (newest report)
    h += others.length ? others.map(([n, o]) => `<h2>💻 Laptop · ${esc(n)}</h2>${chipRows(o.chips, "Laptop")}`).join("") : `<h2>💻 Laptop</h2><p class="muted">Offline (NpuStudio isn't open on the laptop).</p>`;
    $("monBox").innerHTML = h;
  } catch (e) { $("monBox").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

// ---------- 1.9.3 Clear VRAM (unloads NpuStudio's own models; they reload on the next use) ----------
function vramCard(r) {
  const mach = r.id === "laptop" ? "Laptop" : "Desktop", nm = mach + (r.name ? " · " + r.name : "");
  if (r.code !== 200 || !r.ok) return `<p class="err">${esc(nm)}: ${esc(r.error || "failed")}</p>`;
  const rows = Object.entries(r.chips || {}).sort((a, b) => CHIP_ORD.indexOf(a[0]) - CHIP_ORD.indexOf(b[0])).map(([n, c]) => {
    const col = CHIP_COL[mach + ":" + n] || "#888";
    return `<b style="color:${col}">${esc(chipLabel(mach, n).split(" (")[0])}</b><span>${c.before_gb.toFixed(1)} → ${c.after_gb.toFixed(1)} GB${c.total_gb ? " of " + c.total_gb.toFixed(1) : ""} · freed ${c.freed_gb.toFixed(1)} GB</span>`; }).join("");
  const un = [...new Set(Object.values(r.unloaded || {}).flat())];
  return `<div style="border-left:4px solid ${CHIP_COL[mach + ":NVIDIA"]};background:#ffffff08;border-radius:8px;padding:6px 10px;margin:6px 0"><b>${esc(nm)}</b><div class="kv">${rows}</div><span class="small muted">${un.length ? "Unloaded: " + esc(un.join(", ")) : "Nothing was loaded"}</span></div>`;
}
document.querySelectorAll("#vramBar button[data-vram]").forEach(b => b.onclick = async () => {
  const id = b.dataset.vram, lbl = { desktop: "the desktop", laptop: "the laptop", both: "both PCs" }[id];
  if (!await ask(`Unload NpuStudio's models from GPU/NPU memory on ${lbl}? They load again on the next use.`, false, "🧹 Clear", "Cancel")) return;
  const box = $("vramMsg"), btns = document.querySelectorAll("#vramBar button"); btns.forEach(x => x.disabled = true); box.innerHTML = `Clearing ${lbl}…`;
  try {
    const r = await fetch("/api/v1/vram/clear", { method: "POST", headers: { "X-Key": key(), "Content-Type": "application/json", ...devHeaders() }, body: JSON.stringify({ id }) });
    let d = {}; try { d = await r.json(); } catch (e) {}
    const list = d.pcs || [{ ...d, code: d.code || r.status, id: d.id || id }];
    box.innerHTML = list.map(vramCard).join("");
  } catch (e) { box.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  btns.forEach(x => x.disabled = false);
});

// ---------- desktop status ----------
async function desk() {
  pcsShow();
  try {
    const [i, hl] = await Promise.all([api("info"), api("health")]);
    $("deskBox").innerHTML = `<div class="kv"><b>PC</b><span>${esc(i.name)}</span><b>Home address</b><span>${esc(i.lan)}:${i.port}</span>
      <b>Tailscale</b><span>${i.tailscale && i.tailscale.length ? esc(i.tailscale.join(", ")) : "not set up"}</span>
      <b>GPU</b><span>${esc(hl.gpu || "N/A")}</span><b>Jobs running</b><span>${i.busy}</span>
      <b>Web UI</b><span>${UI_VERSION} (server files ${esc((i.ui || {}).ui_version || "?")}, running server code ${esc(i.server_code || "OLD (pre-1.7.7) - restart the server")}${i.pid ? ", pid " + i.pid : ""})</span><b>Android shell</b><span>${shellVersion || "browser"}</span></div>
      ${(i.jobs || []).length ? "<h3>Jobs</h3>" + i.jobs.map(j => `<div class="row"><span class="grow">${esc(j.type)} · ${esc(j.state)}</span>${j.state === "running" || j.state === "confirm" ? `<button class="ghost danger" data-id="${j.id}">Cancel</button>` : ""}</div>`).join("") : ""}`;
    $("deskBox").querySelectorAll("button[data-id]").forEach(b => b.onclick = async () => { await api(`jobs/${b.dataset.id}/cancel`, {}); desk(); });
  } catch (e) { $("deskBox").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}
$("deskRe").onclick = () => { desk(); pcsShow(); };
// ---------- restart the NpuStudio PC apps ----------
async function pcsShow(msg) {
  const box = $("pcsBox"); if (!box) return;
  try {
    const r = await api("pcs");
    let phones = []; try { phones = ((await api("devices")).devices || []).filter(d => d.kind === "phone"); } catch (e) {}
    const me = devHeaders()["X-Device-Id"];
    const meRow = { id: "phone:" + me, name: phoneModel() || "This phone", online: true, ui: UI_VERSION, appver: appVersion(), app: shellVersion || null };
    phones = [meRow].concat(phones.filter(p => p.id !== "phone:" + me && p.id !== "phone:phone" && p.online));
    const phoneHtml = phones.map(p => `<div class="row"><span class="grow"><b>${esc(p.model || p.name)}</b> (phone)${p === meRow ? " · this device" : ""}<br><span class="small muted">${p.online ? "🟢" : "⚪"} web UI ${esc(p.ui || "?")}${p.app ? ` · Android app ${esc(p.appver || "")}${p.appver ? " " : ""}(shell ${esc(String(p.app))})` : p === meRow ? " · browser" : ""}</span></span>` +
      (p === meRow ? `<button class="ghost" id="phReload">↻ Reload page</button>` : "") + `</div>`).join("");
    box.innerHTML = (r.pcs || []).map(p => `<div class="row"><span class="grow"><b>${esc(p.name)}</b> ${p.id === "laptop" ? "(laptop)" : "(desktop)"}<br><span class="small muted">${p.online ? "🟢 app open" : p.online === null ? "🟡 needs one manual restart" : "⚪ app closed"} · v${esc(p.ver || "?")}${p.busy && p.busy.length ? " · busy: " + esc(p.busy.join(", ")) : ""}${p.note && !p.online ? " · " + esc(p.note) : ""}</span></span>
      <button class="ghost" data-pc="${p.id}" ${p.online ? "" : "disabled"}>↻ Restart</button></div>`).join("") + phoneHtml + (msg ? `<p class="small">${esc(msg)}</p>` : "");
    const rl = $("phReload"); if (rl) rl.onclick = () => { const u = new URL(location.href); u.searchParams.set("v", Date.now()); location.replace(u.toString()); };
    box.querySelectorAll("button[data-pc]").forEach(b => b.onclick = () => pcRestart(b.dataset.pc, (r.pcs || []).find(p => p.id === b.dataset.pc)));
  } catch (e) { box.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}
async function pcRestart(id, before) {
  const name = before ? before.name : id;
  if (!await ask(`Restart the NpuStudio app on ${name}${id === "desktop" ? " and its phone server" : ""}? (Not the PC.)`, false, "↻ Restart", "Cancel")) return;
  let r;
  try { r = await api("pcs/restart", { id }); }
  catch (e) {
    if (e.status !== 409) return ask("Restart failed: " + e.message, true);
    let d = {}; try { d = await (await fetch("/api/v1/pcs/restart", { method: "POST", headers: { "X-Key": key(), "Content-Type": "application/json" }, body: JSON.stringify({ id, dry_run: true }) })).json(); } catch (x) {}
    if (!d.busy) return ask(e.message, true);
    if (!await ask(`${name} is busy: ${d.busy.join(", ")}. Restarting stops it.`, false, "Restart anyway", "Cancel")) return;
    try { r = await api("pcs/restart", { id, force: true }); } catch (e2) { return ask("Restart failed: " + e2.message, true); }
  }
  const t0 = (r && r.started_before) || (before && before.started), end = Date.now() + 120000;
  $("pcsBox").insertAdjacentHTML("beforeend", `<p class="small" id="pcsWait">Restarting ${esc(name)}…</p>`);
  while (Date.now() < end) {
    await sleep(2500);
    try {
      const p = ((await api("pcs")).pcs || []).find(x => x.id === id);
      if (p && p.online && p.started && p.started !== t0) return pcsShow(`✓ ${name} restarted (v${p.ver || "?"})`);
      const w = $("pcsWait"); if (w) w.textContent = `Restarting ${name}… waiting for the app`;
    } catch (e) { const w = $("pcsWait"); if (w) w.textContent = "Reconnecting…"; }
  }
  pcsShow(`${name} hasn't come back after 2 minutes - check the PC.`);
}
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

// remember the last-used Gen steps on this phone (default 50)
(function () { const e = $("genSteps"); if (!e) return; const v = localStorage.getItem("npuGenSteps"); if (v) e.value = v; e.addEventListener("change", () => localStorage.setItem("npuGenSteps", e.value)); })();
(function () { const e = $('upDev'); if (!e) return; const v = localStorage.getItem('npuUpDev'); if (v && [...e.options].some(o => o.value === v)) e.value = v; e.addEventListener('change', () => localStorage.setItem('npuUpDev', e.value)); })();
(function () { const e = $('genDev'); if (!e) return; const v = localStorage.getItem('npuGenDev'); if (v && [...e.options].some(o => o.value === v)) e.value = v; e.addEventListener('change', () => localStorage.setItem('npuGenDev', e.value)); })();
(function () { const e = $('editDev'); if (!e) return; const v = localStorage.getItem('npuEditDev'); if (v && [...e.options].some(o => o.value === v)) e.value = v; e.addEventListener('change', () => localStorage.setItem('npuEditDev', e.value)); })();

(function () { const e = $('genDev'), m = $('genMulti'); if (!e || !m) return; const f = () => { const multi = e.value === "Both machines" || e.value === "All chips"; m.classList.toggle("hidden", !multi); $('genSkipSlow').parentElement.classList.toggle("hidden", e.value !== "All chips"); }; e.addEventListener('change', f); f(); })();
// ---------- 1.8.4 Gallery: every picture saved on the desktop / laptop ----------
const GAL = { src: "desktop", kind: "All", page: 0, total: 0, items: [], done: false, loading: false, sel: new Set(), cur: -1, q: [], act: 0, lastDate: null };
(function galCss() {
  const s = document.createElement("style"); s.textContent = `
  #gal{position:fixed;inset:0;z-index:50;background:#111;display:flex;flex-direction:column}
  #gal .gh{padding:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;border-bottom:1px solid #333}
  #gal .chips button{padding:4px 10px;font-size:13px}
  #gal .chips button.on,#gal .srcs button.on{background:#0068b5;color:#fff}
  #galGrid{flex:1;overflow-y:auto;padding:6px}
  #galGrid .gd{color:#aaa;font-size:13px;margin:10px 2px 4px;grid-column:1/-1}
  #galGrid .gg{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:4px}
  #galGrid .gt{position:relative;aspect-ratio:1;background:#222;border-radius:6px;overflow:hidden}
  #galGrid .gt img{width:100%;height:100%;object-fit:cover;display:block}
  #galGrid .gt .k{position:absolute;left:3px;bottom:3px;font-size:10px;background:#000a;color:#ddd;padding:1px 4px;border-radius:4px}
  #galGrid .gt.sel{outline:3px solid #2f9bff}
  #galGrid .gt.sel::after{content:"✓";position:absolute;right:4px;top:2px;color:#2f9bff;font-weight:bold;font-size:18px}
  #galBar{padding:8px;display:flex;gap:6px;border-top:1px solid #333}
  #gv{position:fixed;inset:0;z-index:60;background:#000;display:flex;flex-direction:column}
  #gvBox{flex:1;overflow:hidden;position:relative;touch-action:none}
  #gvImg{position:absolute;left:50%;top:50%;max-width:100%;max-height:100%;transform-origin:center;transform:translate(-50%,-50%)}
  #gvInfo{color:#ccc;font-size:12px;padding:6px 8px;max-height:22vh;overflow-y:auto}
  #gvBtns{display:flex;flex-wrap:wrap;gap:4px;padding:6px}
  #gvBtns button{flex:1 1 30%;font-size:13px;padding:6px 4px}
  #dlg{z-index:100 !important}
  #galToast{position:fixed;left:50%;bottom:70px;transform:translateX(-50%);z-index:90;background:#333;color:#fff;padding:6px 12px;border-radius:14px;font-size:13px}`;
  document.head.appendChild(s);
})();
function galToast(t, ms) { let e = $("galToast"); if (!e) { e = document.createElement("div"); e.id = "galToast"; document.body.appendChild(e); } e.textContent = t; e.classList.remove("hidden"); clearTimeout(e._t); e._t = setTimeout(() => e.classList.add("hidden"), ms || 2500); }
async function apiBlob(path) {
  const r = await fetch("/api/v1/" + path, { headers: { "X-Key": key(), ...devHeaders() }, cache: "no-store" });
  if (!r.ok) { let d = {}; try { d = await r.json(); } catch (e) {} throw new Error(d.error || ("HTTP " + r.status)); }
  return r.blob();
}
const blobB64 = b => new Promise((res, rej) => { const f = new FileReader(); f.onload = () => res(String(f.result).split(",")[1]); f.onerror = rej; f.readAsDataURL(b); });
const blobUrlData = b => new Promise((res, rej) => { const f = new FileReader(); f.onload = () => res(String(f.result)); f.onerror = rej; f.readAsDataURL(b); });
const gq = (sub, it) => `gallery/${sub}?src=${GAL.src}&id=${encodeURIComponent(it.id)}`;
function galOpen() {
  let g = $("gal");
  if (!g) {
    g = document.createElement("div"); g.id = "gal"; document.body.appendChild(g);
    g.innerHTML = `<div class="gh"><button id="galClose">✕</button><b>Gallery</b><span class="srcs"><button data-s="desktop" class="on">🖥️ Desktop</button> <button data-s="laptop">💻 Laptop</button></span>
      <span class="muted small" id="galCount"></span></div><div class="gh chips" id="galKinds"></div><div id="galGrid"></div>
      <div id="galBar" class="hidden"><span id="galSelN" class="grow"></span>${saveFmtUI()}<button id="galSelSave">💾 Save</button><button id="galSelDel">🗑️ Delete</button><button id="galSelX">Cancel</button></div>`;
    $("galClose").onclick = () => { g.classList.add("hidden"); galSelClear(); };
    g.querySelectorAll(".srcs button").forEach(b => b.onclick = () => { GAL.src = b.dataset.s; g.querySelectorAll(".srcs button").forEach(x => x.classList.toggle("on", x === b)); galReset(); });
    $("galSelX").onclick = galSelClear;
    $("galSelSave").onclick = async () => { const its = GAL.items.filter(i => GAL.sel.has(i.id)); for (let k = 0; k < its.length; k++) { galToast(`Saving ${k + 1} of ${its.length}…`, 60000); await galSave(its[k], true); } notice(`Saved ${its.length} picture(s) to the phone.`); galSelClear(); };
    $("galSelDel").onclick = async () => { const ids = [...GAL.sel]; if (!ids.length) return; if (!await ask(`Move ${ids.length} picture(s) to the ${GAL.src === "laptop" ? "laptop's" : "desktop's"} Recycle Bin?`)) return; await galDelete(ids); galSelClear(); };
    $("galGrid").addEventListener("scroll", () => { const e = $("galGrid"); if (e.scrollTop + e.clientHeight > e.scrollHeight - 600) galMore(); });
  }
  g.classList.remove("hidden"); galReset();
}
function galSelClear() { GAL.sel.clear(); document.querySelectorAll("#galGrid .gt.sel").forEach(e => e.classList.remove("sel")); galBar(); }
function galBar() { const n = GAL.sel.size; $("galBar").classList.toggle("hidden", !n); $("galSelN").textContent = `${n} selected`; }
function galReset() {
  GAL.page = 0; GAL.items = []; GAL.done = false; GAL.lastDate = null; GAL.q = []; GAL.sel.clear(); galBar();
  $("galGrid").innerHTML = ""; $("galCount").textContent = "loading…"; galMore(true);
}
async function galMore(first) {
  if (GAL.loading || GAL.done) return; GAL.loading = true;
  try {
    const r = await api(`gallery/list?src=${GAL.src}&kind=${encodeURIComponent(GAL.kind)}&page=${GAL.page}&n=60${first ? "&refresh=1" : ""}`);
    if (first) $("galKinds").innerHTML = r.kinds.map(k => `<button data-k="${esc(k)}" class="${k === GAL.kind ? "on" : ""}">${esc(k)}</button>`).join("");
    $("galKinds").querySelectorAll("button").forEach(b => b.onclick = () => { GAL.kind = b.dataset.k; galReset(); });
    GAL.total = r.total; $("galCount").textContent = `${r.total} picture${r.total === 1 ? "" : "s"}`;
    if (!r.total && first) $("galGrid").innerHTML = `<p class="muted">No pictures here yet.</p>`;
    const grid = $("galGrid");
    r.items.forEach(it => {
      const idx = GAL.items.push(it) - 1;
      if (it.date !== GAL.lastDate) { GAL.lastDate = it.date; grid.insertAdjacentHTML("beforeend", `<div class="gd">${esc(it.date)}</div><div class="gg"></div>`); }
      const t = document.createElement("div"); t.className = "gt"; t.dataset.i = idx; t.innerHTML = `<span class="k">${esc(it.kind)}</span>`;
      grid.lastElementChild.appendChild(t); galThumbObs.observe(t); galPress(t, idx);
    });
    GAL.page++; GAL.done = GAL.items.length >= r.total;
  } catch (e) { $("galCount").textContent = ""; $("galGrid").insertAdjacentHTML("beforeend", `<p class="err">${esc(e.message)}${GAL.src === "laptop" ? " (is NpuStudio open on the laptop?)" : ""}</p>`); GAL.done = true; }
  finally { GAL.loading = false; }
}
const galThumbObs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { galThumbObs.unobserve(e.target); GAL.q.push(e.target); galPump(); } }), { root: null, rootMargin: "400px" });
function galPump() {   // at most 4 thumbnails in flight
  while (GAL.act < 4 && GAL.q.length) {
    const t = GAL.q.shift(), it = GAL.items[+t.dataset.i]; if (!it) continue; GAL.act++;
    apiBlob(gq("thumb", it)).then(b => { const im = document.createElement("img"); im.src = URL.createObjectURL(b); im.alt = it.name; t.prepend(im); })
      .catch(() => t.insertAdjacentHTML("afterbegin", `<span class="small muted" style="padding:4px;display:block">⚠️</span>`)).finally(() => { GAL.act--; galPump(); });
  }
}
function galPress(t, idx) {   // tap = open; long-press = select (then taps toggle)
  let tm = null, long = false;
  t.addEventListener("pointerdown", () => { long = false; tm = setTimeout(() => { long = true; galToggle(t, idx); }, 500); });
  ["pointerup", "pointerleave", "pointercancel"].forEach(ev => t.addEventListener(ev, () => clearTimeout(tm)));
  t.addEventListener("contextmenu", e => e.preventDefault());
  t.addEventListener("click", () => { if (long) return; if (GAL.sel.size) galToggle(t, idx); else galView(idx); });
}
function galToggle(t, idx) { const id = GAL.items[idx].id; if (GAL.sel.has(id)) GAL.sel.delete(id); else GAL.sel.add(id); t.classList.toggle("sel", GAL.sel.has(id)); galBar(); }
async function galSave(it, quiet) {
  try {
    const b = await apiBlob(gq("full", it));
    const nm = await saveImg(await blobB64(b), it.name, { quiet });   // 1.8.7: honours Save as (keeps the original type if it can't convert)
    if (!quiet && !(Shell && Shell.saveImage)) notice("Saved " + nm);
  } catch (e) { notice("Couldn't save: " + e.message); }
}
async function galDelete(ids) {
  try {
    const r = await api("gallery/delete", { ids, src: GAL.src });
    const n = (r.deleted || []).length, bad = Object.keys(r.errors || {}).length;
    notice(`Moved ${n} to the Recycle Bin` + (bad ? `; ${bad} failed: ${Object.values(r.errors)[0]}` : ""));
    galReset();
  } catch (e) { notice("Delete failed: " + e.message); }
}
function galView(idx) {
  let v = $("gv");
  if (!v) {
    v = document.createElement("div"); v.id = "gv"; document.body.appendChild(v);
    v.innerHTML = `<div class="row" style="padding:6px"><button id="gvX">✕</button><span id="gvN" class="grow muted small"></span><button id="gvP">‹</button><button id="gvNx">›</button></div>
      <div id="gvBox"><img id="gvImg"></div><div id="gvInfo"></div>
      <div id="gvBtns"><button data-a="save">💾 Save to phone</button><button data-a="share">📤 Share</button><button data-a="edit">✏️ Send to Edit</button>
      <button data-a="up">🔍 Send to Upscale</button><button data-a="ana">🧐 Analyze</button><button data-a="gen">🎲 Use prompt + seed</button><button data-a="del">🗑️ Delete</button></div><div style="padding:0 8px 6px">${saveFmtUI()}</div>`;
    $("gvX").onclick = () => v.classList.add("hidden");
    $("gvP").onclick = () => galShow(GAL.cur - 1); $("gvNx").onclick = () => galShow(GAL.cur + 1);
    v.querySelectorAll("#gvBtns button").forEach(b => b.onclick = () => galAct(b.dataset.a));
    galGestures($("gvBox"), $("gvImg"));
  }
  v.classList.remove("hidden"); galShow(idx);
}
const GV = { s: 1, x: 0, y: 0, info: null, url: null };
function gvApply() { $("gvImg").style.transform = `translate(calc(-50% + ${GV.x}px), calc(-50% + ${GV.y}px)) scale(${GV.s})`; }
async function galShow(idx) {
  if (idx < 0 || idx >= GAL.items.length) { if (idx >= GAL.items.length && !GAL.done) { await galMore(); if (idx < GAL.items.length) return galShow(idx); } return; }
  GAL.cur = idx; const it = GAL.items[idx]; GV.s = 1; GV.x = GV.y = 0; gvApply(); GV.info = null;
  $("gvN").textContent = `${idx + 1} / ${GAL.total} · ${GAL.src === "laptop" ? "Laptop" : "Desktop"} · ${it.kind}`;
  const im = $("gvImg"); im.removeAttribute("src"); $("gvInfo").textContent = "loading…";
  try { const b = await apiBlob(gq("view", it)); if (GAL.cur !== idx) return; if (GV.url) URL.revokeObjectURL(GV.url); GV.url = URL.createObjectURL(b); im.src = GV.url; } catch (e) { $("gvInfo").textContent = e.message; }
  try {
    const d = await api(gq("info", it)); if (GAL.cur !== idx) return; GV.info = d;
    $("gvInfo").innerHTML = [d.device && `<b>Chip</b> ${esc(d.device)}`, d.seed != null && `<b>Seed</b> ${d.seed}`, d.width && `<b>Size</b> ${d.width}×${d.height}`,
      `<b>Date</b> ${esc(d.modified)}`, `<b>File</b> ${esc(it.folder)}/${esc(d.name)} (${(d.bytes / 1048576).toFixed(1)} MB)`, d.prompt && `<b>Prompt</b> ${esc(d.prompt)}`].filter(Boolean).join(" · ");
  } catch (e) {}
}
function galGestures(box, im) {
  const pts = new Map(); let st = null, sx = 0, tap = 0;
  box.addEventListener("pointerdown", e => { box.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); sx = e.clientX;
    if (pts.size === 2) { const [a, b] = [...pts.values()]; st = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), s: GV.s }; } else st = { px: e.clientX, py: e.clientY, x: GV.x, y: GV.y }; });
  box.addEventListener("pointermove", e => { if (!pts.has(e.pointerId)) return; pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2 && st && st.d) { const [a, b] = [...pts.values()]; GV.s = Math.min(8, Math.max(1, st.s * Math.hypot(a[0] - b[0], a[1] - b[1]) / st.d)); if (GV.s === 1) GV.x = GV.y = 0; gvApply(); }
    else if (pts.size === 1 && GV.s > 1 && st && st.px != null) { GV.x = st.x + e.clientX - st.px; GV.y = st.y + e.clientY - st.py; gvApply(); } });
  const up = e => { if (!pts.has(e.pointerId)) return; pts.delete(e.pointerId);
    if (!pts.size) { const dx = e.clientX - sx;
      if (GV.s === 1 && Math.abs(dx) > 60) galShow(GAL.cur + (dx < 0 ? 1 : -1));
      else if (Math.abs(dx) < 8) { const n = Date.now(); if (n - tap < 300) { GV.s = GV.s > 1 ? 1 : 2.5; GV.x = GV.y = 0; gvApply(); } tap = n; }
      st = null; } };
  box.addEventListener("pointerup", up); box.addEventListener("pointercancel", up);
  box.addEventListener("wheel", e => { e.preventDefault(); GV.s = Math.min(8, Math.max(1, GV.s * (e.deltaY < 0 ? 1.15 : 0.87))); if (GV.s === 1) GV.x = GV.y = 0; gvApply(); }, { passive: false });
}
async function galAct(a) {
  const it = GAL.items[GAL.cur]; if (!it) return;
  const pic = async () => { const b = await apiBlob(gq(it.bytes < 15e6 ? "full" : "view", it)); return blobUrlData(b); };
  try {
    if (a === "save") return galSave(it);
    if (a === "share") { const b = await apiBlob(gq(it.bytes < 15e6 ? "full" : "view", it)); return shareImg(await blobB64(b), it.name); }
    if (a === "del") { if (await ask(`Move ${it.name} to the ${GAL.src === "laptop" ? "laptop's" : "desktop's"} Recycle Bin?`)) { await galDelete([it.id]); $("gv").classList.add("hidden"); } return; }
    if (a === "gen") { const d = GV.info || {}; if (!d.prompt && d.seed == null) return notice("No prompt or seed saved with this picture.");
      if (d.prompt) $("genPrompt").value = d.prompt; if (d.seed != null) $("genSeed").value = d.seed; $("gv").classList.add("hidden"); $("gal").classList.add("hidden"); tabTo("gen"); return notice("Prompt and seed copied into Gen."); }
    galToast("Loading picture…"); const d = await pic(); $("gv").classList.add("hidden"); $("gal").classList.add("hidden");
    if (a === "edit") { await editLoad(d, true); tabTo("edit"); }
    if (a === "up") { await upLoad(d); tabTo("up"); }
    if (a === "ana") { await anaLoad(d); tabTo("ana"); }
  } catch (e) { notice(e.message); }
}
if ($("galOpenBtn")) $("galOpenBtn").onclick = galOpen;
