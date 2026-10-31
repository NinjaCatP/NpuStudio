"""NpuStudio desktop worker: lets the laptop send heavy jobs (Image Gen, upscale, AI Change, background removal, erase)
to this PC over the home network. LAN only: binds to this PC's home-network address, refuses non-private callers,
and every request needs the pairing key from models/remote_server.json. Start with NpuStudio-Server.bat."""
import sys as _s, os as _o
for _n in ("stdout", "stderr"):
    if getattr(_s, _n) is None: setattr(_s, _n, open(_o.devnull, "w", encoding="utf-8"))
import os, urllib.request, sys, json, time, uuid, secrets, socket, logging, threading, ipaddress, traceback, inspect
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
os.environ["NPU_SERVER"] = "1"
APP = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(APP, "models"); sys.path.insert(0, APP)
logging.basicConfig(filename=os.path.join(APP, "error.log"), level=logging.INFO, format="%(asctime)s %(levelname)s [server] %(message)s")
import wire
CFG = os.path.join(MD, "remote_server.json"); PORT = 47860; VERSION = 1

def config():
    os.makedirs(MD, exist_ok=True)
    try: c = json.load(open(CFG))
    except Exception: c = {}
    if not c.get("key"): c["key"] = secrets.token_urlsafe(24)
    c.setdefault("port", PORT); json.dump(c, open(CFG, "w"), indent=1); return c

TS_NET = ipaddress.ip_network("100.64.0.0/10")   # Tailscale addresses
def allowed_peer(ip):
    a = ipaddress.ip_address(ip); return a.is_private or a in TS_NET or a.is_loopback

def tailscale_ips():
    out = []
    try:
        for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            if ipaddress.ip_address(i[4][0]) in TS_NET and i[4][0] not in out: out.append(i[4][0])
    except Exception: pass
    try:
        import subprocess
        r = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5, creationflags=0x08000000)
        for l in r.stdout.split():
            if l not in out and ipaddress.ip_address(l) in TS_NET: out.append(l)
    except Exception: pass
    return out

def lan_ip():
    ips = [i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("192.168.0.1", 9)); ips.insert(0, s.getsockname()[0]); s.close()
    except Exception: pass
    for pref in ("192.168.", "10.", "172."):
        for ip in ips:
            if ip.startswith(pref) and ipaddress.ip_address(ip).is_private: return ip
    raise RuntimeError("no home-network address found")

# ---------------- jobs ----------------
_up = {}; _up_lock = threading.Lock(); _ai = None; _ai_lock = threading.Lock(); JOBS = {}; _busy = [0]
def upscaler(name, pref=None):
    import npustudio
    pref = None if pref in (None, "", "Auto") else pref
    with _up_lock:
        if (name, pref) not in _up: _up[(name, pref)] = npustudio.Upscaler(name, pref)
        return _up[(name, pref)]

# Phone/API limits for an upscale result (the laptop app's 18 GB memory budget applies too)
API_MAX_SIDE = 16384          # WebViews / most image viewers can't decode a bigger side
API_MAX_MP = 160              # ~160 MP: still a sane download over Wi-Fi as JPEG
UP_DEVICES = ["Auto", "NVIDIA", "Intel Arc", "CPU", "All"]
def upscale_check(w, h, scale):
    import npustudio
    ow, oh = w * scale, h * scale; need = npustudio.need_bytes(w, h, scale)
    r = {"width": w, "height": h, "scale": scale, "out_width": ow, "out_height": oh, "megapixels": round(ow * oh / 1e6, 1),
         "memory_gb": round(need / 2**30, 2), "memory_limit_gb": npustudio.MAX_BYTES / 2**30,
         "max_side": API_MAX_SIDE, "max_megapixels": API_MAX_MP, "ok": True, "reason": ""}
    if scale not in (2, 4, 8, 16): r.update(ok=False, reason="scale must be 2, 4, 8 or 16")
    f = min(1.0, API_MAX_SIDE / max(ow, oh), (API_MAX_MP * 1e6 / (ow * oh)) ** 0.5)   # input size that would fit
    r["fit_width"], r["fit_height"] = max(1, int(w * f)), max(1, int(h * f))
    # Too big for a phone: upscale only as far as needed, then resize the result down to the phone max (not the source).
    fw, fh, rs = phone_cap(w, h, scale)
    r.update(final_width=fw, final_height=fh, run_scale=rs, capped=(fw, fh) != (ow, oh), full_on_desktop=need <= npustudio.MAX_BYTES)
    if r["ok"] and not r["full_on_desktop"]:
        r["full_note"] = f"full {ow}x{oh} needs about {need/2**30:.1f} GB, over the {npustudio.MAX_BYTES/2**30:.0f} GB memory limit - the desktop keeps the {rs}x result"
        if npustudio.need_bytes(w, h, rs) > npustudio.MAX_BYTES: r.update(ok=False, reason=f"needs about {npustudio.need_bytes(w, h, rs)/2**30:.1f} GB, over the {npustudio.MAX_BYTES/2**30:.0f} GB limit")
    if r["capped"]: r["note"] = f"{ow}x{oh} is over the phone max ({API_MAX_SIDE}px side, {API_MAX_MP} MP): runs {rs}x, then resizes to {fw}x{fh}"
    return r
def im_size(im, s): return (im.size[0] * s, im.size[1] * s)
def phone_cap(w, h, scale, mw=0, mh=0):
    """-> (final w, final h, model scale to run). Caps at 16384 px side / 160 MP (and an optional max_w x max_h)."""
    f = min(float(scale), API_MAX_SIDE / max(w, h), (API_MAX_MP * 1e6 / (w * h)) ** 0.5)
    if mw > 0 and mh > 0: f = min(f, mw / w, mh / h)
    rs = next((x for x in (2, 4, 8, 16) if x >= f - 1e-9 and x <= scale), scale)
    fw, fh = (w * scale, h * scale) if f >= scale else (max(1, min(API_MAX_SIDE, round(w * f))), max(1, min(API_MAX_SIDE, round(h * f))))
    return fw, fh, rs
def ai():
    global _ai
    with _ai_lock:
        if _ai is None:
            import editor; _ai = editor.AI()
        return _ai

def run(fn, args, kw, progress, status):
    import cudaeng
    if fn in ("ip2p", "inpaint", "inpaint_cn", "inpaint_pose", "gen"):
        f = getattr(cudaeng, fn); ps = inspect.signature(f).parameters
        if "progress" in ps: kw["progress"] = progress
        if "status" in ps: kw["status"] = status
        return f(*args, **kw)
    if fn == "upscale":          # whole picture: (model name, image, scale); kw device = chip on this machine (Laptop NPU...)
        name, im, scale = args; u = upscaler(name, kw.get("device")); status(f"Upscaling on {socket.gethostname()} {u.device}...")
        return u.upscale_to(im, int(scale), progress)
    if fn == "tile":             # one 4x tile for split ("Both machines") upscales
        name, tile = args; u = upscaler(name); return u.workers[0][1](tile)
    if fn == "chat_ask":         # the other PC's chat uses this PC's LLM (e.g. Phi-3.5 on the laptop NPU); actions stay on the caller
        text, target = args[0], (args[1] if len(args) > 1 else "pc")
        import npustudio
        eng = chat_engine(status)
        if kw.get("new") or getattr(eng, "target", "pc") != target: eng.want_target = target; eng.new_chat()
        buf = []
        def tok(t): buf.append(t); status("\x00" + "".join(buf))
        n, dt = eng.ask(text, tok)
        return {"text": "".join(buf), "tokens": n, "seconds": dt, "device": eng.device}
    if fn == "chat_new":
        if _chat["eng"]: _chat["eng"].new_chat()
        return True
    if fn == "remove_bg": return ai().remove_bg(args[0], status)
    if fn == "erase": return ai().erase(args[0], args[1], status)
    if fn == "change": return ai().change(*args, progress=progress, status=status, **kw)
    raise ValueError("unknown job " + fn)

def gpu_info():
    d = {"name": socket.gethostname(), "version": VERSION, "busy": _busy[0]}
    try:
        import pynvml; pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
        n = pynvml.nvmlDeviceGetName(h); d["gpu"] = n.decode() if isinstance(n, bytes) else n
        d["use"] = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        m = pynvml.nvmlDeviceGetMemoryInfo(h); d["vram_used"] = m.used >> 20; d["vram_total"] = m.total >> 20
        d["watts"] = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1)
        d["temp"] = pynvml.nvmlDeviceGetTemperature(h, 0)
    except Exception: d.setdefault("gpu", "N/A")
    return d

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _ok(self):
        if not allowed_peer(self.client_address[0]): return False
        k = self.headers.get("X-Key") or self.headers.get("Authorization", "").replace("Bearer ", "", 1)
        return secrets.compare_digest(k.strip(), KEY)
    def _send(self, code, body, ctype="application/octet-stream"):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode(); ctype = "application/json"
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path.split("?")[0].rstrip("/") == "/pair": return pair_page(self)
        if self.path.split("?")[0].rstrip("/") in ("/app", "") or self.path.startswith("/app/"):
            return serve_app(self)   # the mobile web UI holds no secrets; its API calls carry the key
        if not self._ok(): return self._send(403, {"error": "bad key"})
        try: note_client(self)
        except Exception: pass
        p = self.path.split("?")[0].strip("/").split("/")
        if p[:2] == ["api", "v1"]: return api_get(self, p[2:])
        if p[0] == "health": return self._send(200, gpu_info())
        if p[0] == "job" and len(p) >= 2 and p[1] in JOBS:
            j = JOBS[p[1]]
            if len(p) == 3 and p[2] == "result":
                if j["state"] == "running": return self._send(409, {"error": "not done"})
                JOBS.pop(p[1], None)
                if j["state"] == "error": return self._send(500, {"error": j["err"]})
                return self._send(200, j["res"])
            return self._send(200, {k: j[k] for k in ("state", "k", "n", "msg")})
        self._send(404, {"error": "not found"})
    def do_POST(self):
        if not self._ok(): return self._send(403, {"error": "bad key"})
        try: note_client(self)
        except Exception: pass
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        p = self.path.split("?")[0].strip("/").split("/")
        if p[:2] == ["api", "v1"]: return api_post(self, p[2:], data)
        req = wire.unpack(data)
        fn, args, kw = req["fn"], list(req.get("args", [])), dict(req.get("kw", {}))
        if self.path.strip("/") == "call":     # quick synchronous call (tiles)
            try: return self._send(200, wire.pack(run(fn, args, kw, None, lambda m: None)))
            except Exception as e: logging.exception("call %s", fn); return self._send(500, {"error": str(e)})
        jid = uuid.uuid4().hex; j = JOBS[jid] = {"state": "running", "k": 0, "n": 0, "msg": "", "res": None, "err": None}
        def prog(k, n): j["k"], j["n"] = k, n
        def stat(m): j["msg"] = str(m)
        def work():
            _busy[0] += 1; t = time.time()
            try: j["res"] = wire.pack(run(fn, args, kw, prog, stat)); j["state"] = "done"; logging.info("job %s done %.1fs", fn, time.time() - t)
            except Exception as e: logging.exception("job %s", fn); j["err"] = f"{type(e).__name__}: {e}"; j["state"] = "error"
            finally: _busy[0] -= 1
        threading.Thread(target=work, daemon=True).start(); self._send(200, {"id": jid})

# ================= pairing QR (only from this PC itself) =================
def pair_page(h):
    if h.client_address[0] not in ("127.0.0.1", "::1"):
        return h._send(403, {"error": "open the pairing page on the desktop itself: http://127.0.0.1:%s/pair" % config()["port"]})
    import html, urllib.parse
    c = config(); hosts = [lan_ip()] + tailscale_ips(); port = c["port"]
    cards = []; ts = tailscale_ips()
    combos = ([("Home + Tailscale (recommended: switches automatically)", {"host": lan_ip(), "remote": ts[0]})] if ts else []) + \
             [("Tailscale (anywhere)" if h.startswith("100.") else "Home Wi-Fi / LAN", {"host": h}) for h in hosts]
    for label, q in combos:
        host = q["host"] + (" + " + q["remote"] if "remote" in q else "")
        uri = "npustudio://pair?" + urllib.parse.urlencode({**q, "port": port, "key": c["key"]})
        try:
            import segno; svg = segno.make(uri, error="m").svg_inline(scale=6)
        except Exception as e: svg = "<p>(QR needs the segno package: %s)</p>" % html.escape(str(e))
        cards.append(f"<div class=c><h2>{label}</h2>{svg}<p><b>Address:</b> {host} port {port}</p></div>")
    body = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width"><title>NpuStudio pairing</title>
<style>body{{font-family:Segoe UI,sans-serif;background:#14161c;color:#e8e8ee;text-align:center}}.c{{display:inline-block;background:#fff;color:#111;border-radius:12px;padding:16px;margin:12px}}code{{background:#222;padding:4px 8px;border-radius:6px}}</style>
<h1>Pair the NpuStudio Companion app</h1><p>In the phone app, tap <b>Scan QR</b> and point it at one of these codes.</p>{''.join(cards)}
<p>Or type the address above and this key: <code>{html.escape(c['key'])}</code></p><p>Keep this key private. To change it, delete models\\remote_server.json and restart the server.</p>"""
    h.send_response(200); h.send_header("Content-Type", "text/html; charset=utf-8"); h.send_header("Cache-Control", "no-store")
    b = body.encode(); h.send_header("Content-Length", str(len(b))); h.end_headers(); h.wfile.write(b)

# ================= mobile web UI (/app) =================
WEB = os.path.join(APP, "webapp")
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
        ".png": "image/png", ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json", ".ico": "image/x-icon", ".apk": "application/vnd.android.package-archive"}
def serve_app(h):
    if not allowed_peer(h.client_address[0]): return h._send(403, {"error": "not allowed"})
    rel = h.path.split("?")[0][len("/app"):].lstrip("/") or "index.html"
    f = os.path.realpath(os.path.join(WEB, rel))
    if not f.startswith(os.path.realpath(WEB) + os.sep) or not os.path.isfile(f): return h._send(404, {"error": "not found"})
    body = open(f, "rb").read(); h.send_response(200)
    h.send_header("Content-Type", MIME.get(os.path.splitext(f)[1].lower(), "application/octet-stream"))
    h.send_header("Cache-Control", "no-cache"); h.send_header("Content-Length", str(len(body))); h.end_headers(); h.wfile.write(body)

# ================= JSON API v1 (phone app / anything that speaks HTTP+JSON). See API.md =================
import base64, io
def b64img(s):
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[-1]))); im.load(); return im
def imgb64(im, fmt="PNG"):
    b = io.BytesIO(); im.save(b, fmt, **({"quality": 92} if fmt == "JPEG" else {"compress_level": 1})); return base64.b64encode(b.getvalue()).decode()

_chat = {"eng": None}; _chat_lock = threading.Lock(); _chg = {"c": None}; _mon = {"c": None}
def chat_engine(status):
    with _chat_lock:
        if _chat["eng"] is None:
            import npustudio; _chat["eng"] = npustudio.ChatEngine(status)
        return _chat["eng"]

class Cancelled(Exception): pass
def save_out(im, kind, stat=None, tag=""):
    """Full-size copy into the desktop outputs folder. Never fails the job."""
    try:
        import outputs
        if stat and im.size[0] * im.size[1] > 24e6: stat(f"saving the full {im.size[0]}x{im.size[1]} on the desktop...")
        return {"saved_path": outputs.save(im, kind, tag), "full_width": im.size[0], "full_height": im.size[1]}
    except Cancelled: raise
    except Exception as e: logging.exception("save output"); return {"save_error": str(e)}
def api_job(kind, a, j):
    """Runs one JSON job. j = job dict (progress k/n, msg, text, confirm...). Returns result dict."""
    def prog(k, n):
        if j.get("cancel"): raise Cancelled("cancelled")
        j["k"], j["n"] = k, n
    def stat(m):
        if j.get("cancel"): raise Cancelled("cancelled")
        j["msg"] = str(m)
    if kind == "gen":
        import imagegen, cudaeng
        tw, th = int(a.get("width", 768)), int(a.get("height", 768)); (w, h), up = imagegen.plan_for(tw, th)
        im = cudaeng.gen(a["prompt"], w, h, int(a.get("steps", 4)), int(a.get("seed", 42)), prog, stat)
        upn = "Fast (Intel SR-1032)" if str(a.get("upscaler", "sharp")).lower() == "fast" else "Sharp (Real-ESRGAN x4plus)"
        if up > 1: stat(f"upscaling {up}x ({upn.split(' (')[0]})..."); im = upscaler(upn).upscale_to(im, up, prog)
        im = imagegen.fit_crop(im, tw, th); return {"image": imgb64(im), "device": "NVIDIA", **save_out(im, "gen", stat, a.get("prompt", ""))}
    if kind == "upscale":
        name = "Sharp (Real-ESRGAN x4plus)" if a.get("model", "sharp") == "sharp" else "Fast (Intel SR-1032)"
        src = b64img(a["image"]).convert("RGB"); sc = int(a.get("scale", 4))
        chk = upscale_check(*src.size, sc)
        if not chk["ok"]: raise ValueError(f"Upscale {sc}x of {src.size[0]}x{src.size[1]}: {chk['reason']}. Pick a smaller scale.")
        mw, mh = int(a.get("max_w") or 0), int(a.get("max_h") or 0)
        fw, fh, rs = phone_cap(*src.size, sc, mw, mh)
        import npustudio; need = npustudio.need_bytes(*src.size, sc); full = need <= npustudio.MAX_BYTES; why = ""
        if full: rs = sc               # always make the full size for the desktop outputs folder, then cap what goes to the phone
        else: why = (f"full {sc}x ({src.size[0] * sc}x{src.size[1] * sc}) needs {need / 2**30:.1f} GB, over the "
                     f"{npustudio.MAX_BYTES / 2**30:.0f} GB memory limit; saved the {rs}x result instead")
        dev = a.get("device") or "Auto"
        if dev not in UP_DEVICES: raise ValueError("device must be one of " + ", ".join(UP_DEVICES))
        u = upscaler(name, dev); stat(f"Upscaling {rs}x on {u.device}" + (" (2 passes)" if rs >= 8 else "") + (f", then resizing to {fw}x{fh}" if (fw, fh) != im_size(src, rs) else "") + "...")
        im = u.upscale_to(src, rs, prog); del src
        sv = save_out(im, "upscale", stat, f"{rs}x {name.split(' (')[0]}")
        if why: sv["full_note"] = why
        if im.size != (fw, fh):        # phone max / "Limit to my screen": send a smaller result
            from PIL import Image; stat(f"resizing to {fw}x{fh}...")
            im = im.resize((fw, fh), Image.LANCZOS)
        fmt = a.get("format", "AUTO").upper()
        if fmt == "AUTO": fmt = "PNG" if im.size[0] * im.size[1] <= 24e6 else "JPEG"     # big PNGs are huge to send
        stat("encoding result..."); return {"image": imgb64(im, fmt), "format": fmt, "width": im.size[0], "height": im.size[1], "device": u.device,
                "ran_scale": rs, **sv}
    if kind in ("remove_bg", "erase"):
        im = b64img(a["image"])
        r, d = ai().remove_bg(im, stat) if kind == "remove_bg" else ai().erase(im, b64img(a["mask"]).convert("L").resize(im.size), stat)
        return {"image": imgb64(r), "device": str(d), **save_out(r, "edit", stat, kind.replace("_", "-"))}
    if kind == "change":     # AI Change: free-text instruction, optional mask (white = area to change)
        import aichange
        if _chg["c"] is None: _chg["c"] = aichange.Changer()
        c = _chg["c"]; im = b64img(a["image"]).convert("RGB"); instr = a["instruction"]
        stat("analyzing picture..."); ana = c.analyze(im, stat)
        mask = b64img(a["mask"]).convert("L").resize(im.size) if a.get("mask") else None
        if mask is None:
            try:
                tgt = aichange.parse(instr); mask = aichange.Seg().mask(im, aichange.targets_for(tgt[0] if isinstance(tgt, tuple) else tgt, ana)) if tgt else None
            except Exception: logging.exception("api auto-mask"); mask = None
        grow = max(-30, min(40, int(a.get("grow", 8))))
        if mask is not None and grow:
            from PIL import ImageFilter
            k = abs(grow) * 2 + 1; mask = mask.filter(ImageFilter.MaxFilter(k) if grow > 0 else ImageFilter.MinFilter(k))
        method = a.get("method", "Auto")
        if method not in ("Auto", "Redraw", "Recolor"): raise ValueError("method must be Auto, Redraw or Recolor")
        if mask is not None and mask.getbbox():
            st_ = float(a.get("strength", 1.0)); stat("changing the masked area...")
            r = c.edit_masked(ai(), im, mask, instr, ana, strength=max(0.5, min(1.0, st_)), progress=prog, status=stat, method=method)
        else:
            if not aichange.is_global(instr) and not a.get("whole"):
                raise ValueError("Couldn't find what to change. Name the thing (e.g. 'change outfit to a red dress'), paint it with the brush, "
                                 "or tick 'Whole picture' for a whole-picture change. Nothing was changed.")
            st_ = float(a.get("strength", 0.4)); r = c.edit_whole(im, instr, ana, strength=max(0.2, min(0.6, st_)), progress=prog, status=stat)
        return {"image": imgb64(r[0]), **save_out(r[0], "aichange", stat, instr), "device": ", ".join(r[1]) if isinstance(r[1], (tuple, list)) else str(r[1]), "method": r[2], "prompt": r[3], "analysis": ana}
    if kind == "chat":       # allowlisted PC actions run on THIS PC; a CONFIRM action waits for POST .../confirm
        eng = chat_engine(stat); out = []
        def on_text(t): out.append(t); j["text"] = "".join(out)
        def confirm(q):
            j["confirm"] = q; j["state"] = "confirm"; ev = j["ev"] = threading.Event()
            ok = ev.wait(120); j["state"] = "running"; j["confirm"] = None
            return bool(ok and j.get("answer"))
        dv = str(a.get("device", "pc")).lower()
        import actions
        if dv == "laptop" or dv.startswith("phone:"):     # another device in the mesh: understand here, run there
            if dv == "laptop":
                def runner(x):
                    if x["name"] in actions.CONFIRM and not confirm(f"Do you want me to {actions.describe(x)} on the laptop?"): return f"Cancelled: {actions.describe(x)}."
                    try: return laptop_action(x)
                    except Exception as e: return f"Couldn't reach the laptop: {e}"
                src, act, res, n, dt = eng.handle(a["message"], on_text, confirm=confirm, target="pc", runner=runner)
            else:
                pid = dv[6:]
                def runner(x):
                    if x["name"] in ("save_image", "share_image", "time"): return "That only works on the phone you're chatting from."
                    if x["name"] in actions.PHONE_CONFIRM and not confirm(f"Do you want me to {actions.describe_phone(x)} on {(_phones.get(pid) or {}).get('name', 'the phone')}?"):
                        return f"Cancelled: {actions.describe_phone(x)}."
                    st_ = phone_status(pid); _, r = phone_queue(x["name"], x.get("args"), pid, wait=20 if st_["connected"] else 0)
                    return r or ("Sent to the phone." if st_["connected"] else "The phone is offline; it will run this when it reconnects.")
                src, act, res, n, dt = eng.handle(a["message"], on_text, confirm=confirm, target="phone", runner=runner)
            return {"text": j.get("text", ""), "action": actions.label(act) if act else None, "action_result": res, "target": dv, "device": eng.device}
        target = "phone" if dv == "phone" else "pc"
        src, act, res, n, dt = eng.handle(a["message"], on_text, confirm=confirm, target=target)
        if target == "phone":   # never run on this PC: hand the allowlisted action back to the phone app
            import actions
            pa = act if act and act.get("name") in actions.PHONE else None
            return {"text": j.get("text", ""), "phone_action": pa, "needs_confirm": bool(pa and pa["name"] in actions.PHONE_CONFIRM),
                    "confirm_text": actions.describe_phone(pa) if pa else None, "target": "phone", "device": eng.device}
        return {"text": j.get("text", ""), "action": actions.label(act) if act else None, "action_result": res, "target": "pc", "device": eng.device}
    raise ValueError("unknown job type " + kind)

REPORTS = {}
def monitor():
    d = {"desktop": gpu_info(), "others": {k: v for k, v in REPORTS.items() if time.time() - v["_t"] < 10}}
    try:
        if _mon["c"] is None:
            import chipmon; _mon["c"] = chipmon.ChipCounter()
        d["chips"] = _mon["c"].read()
    except Exception as e: d["chips"] = {}; d["chips_error"] = str(e)
    return d

# ---- device mesh: phones (Companion v2.1 background link long-polls), the laptop (reports every 3 s), this desktop ----
_phones = {}            # id -> {"name", "seen", "via", "q": [cmd], "res": {cid: text}}
_pcv = threading.Condition()
PHONES_F = os.path.join(MD, "phones.json"); _psave = [0.0]
def _ph(pid, name=None):
    p = _phones.setdefault(pid, {"name": name or "Phone", "seen": 0.0, "via": None, "q": []})
    if name: p["name"] = name
    return p
def _phones_save(force=False):
    if not force and time.time() - _psave[0] < 30: return
    _psave[0] = time.time()
    try: json.dump({k: {x: p.get(x) for x in ("name", "seen", "ver", "how")} for k, p in _phones.items() if k != "phone"}, open(PHONES_F, "w"), indent=1)
    except Exception as e: logging.debug("phones save: %s", e)
def _phones_load():
    try:
        for k, v in json.load(open(PHONES_F)).items():
            p = _ph(k, v.get("name")); p["seen"] = float(v.get("seen") or 0); p["ver"] = v.get("ver"); p["how"] = v.get("how")
    except Exception: pass
def note_client(h):
    """Any authorised request from the phone app/web UI registers that phone (not only the v2.1 background service)."""
    if SHARE: return
    ua = h.headers.get("User-Agent", ""); pid = (h.headers.get("X-Device-Id") or "")[:64]
    if "Android" not in ua and not pid: return
    if not pid:
        import hashlib; pid = "ua-" + hashlib.sha1((ua + h.client_address[0]).encode()).hexdigest()[:8]
    import re; m = re.search(r"Android [\d.]+; ([^;)]+?)(?: Build|\))", ua)
    name = (h.headers.get("X-Device-Name") or (m.group(1).strip() if m else "Android phone"))[:40]
    with _pcv:
        new = pid not in _phones; p = _ph(pid, name); p["seen"] = time.time()
        p["via"] = "remote" if h.client_address[0].startswith("100.") else "home"
        v = h.headers.get("X-Shell-Version")
        if v: p["ver"] = v
        p.setdefault("how", "web")
    _phones_save(new)
def phone_status(pid=None):
    ps = [p for k, p in _phones.items() if pid in (None, k)]
    p = max(ps, key=lambda x: x["seen"]) if ps else None
    ago = time.time() - p["seen"] if p and p["seen"] else None
    return {"connected": ago is not None and ago < 75, "last_seen_s": round(ago, 1) if ago is not None else None,
            "via": p and p["via"], "pending": len(p["q"]) if p else 0, "phones": len(_phones)}
def phone_poll(h, q):
    wait = max(0, min(30, int((q.get("wait") or ["25"])[0] or 25)))
    pid = (q.get("id") or ["phone"])[0][:64]; name = (q.get("name") or [None])[0]
    end = time.time() + wait
    with _pcv:
        p = _ph(pid, name and name[:40]); p["via"] = "remote" if h.client_address[0].startswith("100.") else "home"; p["how"] = "service"
        while True:
            p["seen"] = time.time()
            if p["q"] or time.time() >= end: break
            _pcv.wait(min(5, end - time.time()))
        cmds, p["q"] = p["q"], []
    p["seen"] = time.time(); _phones_save()
    return h._send(200, {"commands": cmds})
_phone_res = {}
def phone_queue(name, args=None, pid=None, wait=0):
    """Queue an allowlisted command for one phone (pid) or all phones; optionally wait for its result text."""
    cid = uuid.uuid4().hex[:12]; cmd = {"cid": cid, "name": name, "args": args or {}, "t": time.time()}
    with _pcv:
        if not _phones: _ph("phone")
        for k, p in _phones.items():
            if pid in (None, k): p["q"] = [c for c in p["q"] if not (name in ("find_phone", "stop_sound") and c["name"] == name)] + [cmd]
        _pcv.notify_all()
        end = time.time() + wait
        while wait and cid not in _phone_res and time.time() < end: _pcv.wait(min(2, end - time.time()))
    return cid, _phone_res.pop(cid, None)
def devices():
    """Everything the chat 'Control' picker can target."""
    out = [{"id": "desktop", "name": socket.gethostname() + " (desktop)", "kind": "pc", "online": True}]
    try:
        lp = json.load(open(os.path.join(MD, "laptop.json")))
        rep = REPORTS.get(lp.get("name", ""), {}); ago = time.time() - rep.get("_t", 0)
        out.append({"id": "laptop", "name": lp.get("name") or "Laptop", "kind": "pc", "online": ago < 15 and bool(rep.get("serve")), "last_seen_s": round(ago, 1) if rep else None})
    except Exception: pass
    for k, p in sorted(_phones.items()):
        if k == "phone" and len(_phones) > 1 and not p["seen"]: continue
        if p.get("how") == "web":   # same phone seen by its background link (or a newer id) -> one entry
            twin = [q for q in _phones.values() if q is not p and (q.get("how") == "service" and q["name"].endswith(p["name"]) or q["name"] == p["name"] and q["seen"] >= p["seen"])]
            if twin:
                for q in twin:
                    if p.get("ver") and not q.get("ver"): q["ver"] = p["ver"]
                    q["seen"] = max(q["seen"], p["seen"])
                continue
        ago = time.time() - p["seen"] if p["seen"] else None
        out.append({"id": "phone:" + k, "name": p["name"], "kind": "phone", "online": ago is not None and ago < 75, "last_seen_s": round(ago, 1) if ago is not None else None,
                    "app": p.get("ver"), "link": p.get("how") == "service"})
    return out
def laptop_action(a):
    """Run an (already confirmed) PC action on the laptop's share server."""
    import remote; c = remote.cfg()
    if not c.get("host"): raise RuntimeError("the laptop isn't linked")
    r = urllib.request.Request(f"http://{c['host']}:{c.get('port', 47860)}/api/v1/action", data=json.dumps({"action": a, "confirmed": True}).encode(),
                               headers={"X-Key": c["key"], "Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(r, timeout=30)).get("result", "")

def api_get(h, p):
    if p == ["health"]: return h._send(200, gpu_info())
    if p == ["phone", "poll"]:
        import urllib.parse; return phone_poll(h, urllib.parse.parse_qs(h.path.partition("?")[2]))
    if p == ["phone", "status"]: return h._send(200, phone_status())
    if p == ["devices"]: return h._send(200, {"devices": devices()})
    if p == ["info"]:
        c = config(); v = {}
        try: v = json.load(open(os.path.join(WEB, "version.json")))
        except Exception: pass
        return h._send(200, {"name": socket.gethostname(), "port": c["port"], "lan": lan_ip(), "tailscale": tailscale_ips(),
                             "ui": v, "busy": _busy[0], "phone": phone_status(), "jobs": [{"id": k, "type": x.get("type"), "state": x["state"]} for k, x in JOBS.items() if x.get("type")]})
    if p == ["monitor"]: return h._send(200, monitor())
    if p == ["options"]:     # what the phone UI should offer (mirrors the laptop app)
        import npustudio, imagegen
        return h._send(200, {"upscale_scales": [2, 4, 8, 16], "upscale_models": [{"id": "sharp", "name": "Sharp (Real-ESRGAN x4plus)"}, {"id": "fast", "name": "Fast (Intel SR-1032)"}],
                             "upscale_devices": UP_DEVICES, "memory_limit_gb": npustudio.MAX_BYTES / 2**30, "max_side": API_MAX_SIDE, "max_megapixels": API_MAX_MP,
                             "gen_sizes": [p.split(" ")[0] for p in imagegen.PRESETS if "x" in p], "change_methods": ["Auto", "Redraw", "Recolor"]})
    if len(p) >= 2 and p[0] == "jobs":
        j = JOBS.get(p[1])
        if not j: return h._send(404, {"error": "no such job"})
        if len(p) == 3 and p[2] == "result":
            if j["state"] in ("running", "confirm"): return h._send(409, {"error": "not done"})
            JOBS.pop(p[1], None)
            if j["state"] == "cancelled": return h._send(410, {"error": "cancelled"})
            return h._send(500 if j["state"] == "error" else 200, {"error": j["err"]} if j["state"] == "error" else j["res"])
        return h._send(200, {"id": p[1], "type": j.get("type"), "state": j["state"], "progress": j["k"], "total": j["n"],
                             "message": j["msg"], "text": j.get("text", ""), "confirm": j.get("confirm")})
    h._send(404, {"error": "not found"})

def api_post(h, p, data):
    try: a = json.loads(data or b"{}")
    except Exception: return h._send(400, {"error": "body must be JSON"})
    if p == ["phone", "command"]:
        import phonelink
        if a.get("name") not in phonelink.CMDS: return h._send(400, {"error": "allowed: " + ", ".join(sorted(phonelink.CMDS))})
        pid = str(a["device"])[6:] if str(a.get("device", "")).startswith("phone:") else None
        phone_queue(a["name"], {}, pid); logging.info("phone command queued: %s", a["name"])
        return h._send(200, {"queued": a["name"], "status": phone_status(pid)})
    if p == ["phone", "result"]:          # the phone's background service reports what a queued command did
        with _pcv: _phone_res[str(a.get("cid"))] = str(a.get("text", ""))[:2000]; _pcv.notify_all()
        return h._send(200, {"ok": True})
    if p == ["action"]:                   # another PC asks THIS PC to run an allowlisted action (confirm was asked on the chatting device)
        import actions
        act = a.get("action") or {}
        if act.get("name") not in actions.ALL: return h._send(400, {"error": "not an allowed action"})
        ok = bool(a.get("confirmed"))
        try: res = actions.execute({"name": act["name"], "args": act.get("args") or {}}, confirm=lambda m: ok)
        except Exception as e: logging.exception("remote action"); return h._send(500, {"error": str(e)})
        logging.info("remote action %s -> %s", act["name"], res); return h._send(200, {"result": res})
    if p == ["chat", "new"]:
        if _chat["eng"]: _chat["eng"].new_chat()
        return h._send(200, {"ok": True})
    if len(p) == 3 and p[0] == "jobs" and p[2] == "cancel":
        j = JOBS.get(p[1])
        if not j: return h._send(404, {"error": "no such job"})
        j["cancel"] = True
        if j["state"] == "confirm": j["answer"] = False; j["ev"].set()
        return h._send(200, {"ok": True})
    if p == ["upscale", "check"]:
        try: return h._send(200, upscale_check(int(a["width"]), int(a["height"]), int(a["scale"])))
        except Exception as e: return h._send(400, {"error": str(e)})
    if p == ["report"]:      # the laptop app pushes its Chip Monitor numbers here so the phone can show them
        a["_t"] = time.time(); a["_ip"] = h.client_address[0]; REPORTS[str(a.get("name", "laptop"))[:40]] = a; return h._send(200, {"ok": True})
    if len(p) == 3 and p[0] == "jobs" and p[2] == "confirm":
        j = JOBS.get(p[1])
        if not j or j["state"] != "confirm": return h._send(409, {"error": "job is not waiting for a confirm"})
        j["answer"] = bool(a.get("yes")); j["ev"].set(); return h._send(200, {"ok": True})
    if p == ["jobs"]:
        kind = a.get("type"); jid = uuid.uuid4().hex
        j = JOBS[jid] = {"type": kind, "state": "running", "k": 0, "n": 0, "msg": "", "res": None, "err": None}
        def work():
            _busy[0] += 1; t = time.time()
            try: j["res"] = api_job(kind, a, j); j["res"]["seconds"] = round(time.time() - t, 2); j["state"] = "done"
            except Cancelled: j["err"] = "cancelled"; j["state"] = "cancelled"
            except Exception as e:
                if j.get("cancel"): j["err"] = "cancelled"; j["state"] = "cancelled"
                else: logging.exception("api job %s", kind); j["err"] = f"{type(e).__name__}: {e}"; j["state"] = "error"
            finally: _busy[0] -= 1
        threading.Thread(target=work, daemon=True).start(); return h._send(200, {"id": jid})
    h._send(404, {"error": "not found"})

STATUS = os.path.join(MD, "server_status.json")
def _status(**kw):
    try: json.dump({"t": time.time(), "pid": os.getpid(), **kw}, open(STATUS + (".share" if SHARE else ""), "w"))
    except Exception: pass

_srv = {}
def bind_all(port, fixed=None):
    """Listen on 127.0.0.1 + the home-network address + any Tailscale address. Called again every 30 s, so a Tailscale
    install (or a new Wi-Fi address) is picked up without a restart. Never binds 0.0.0.0."""
    want = []
    try: want.append(fixed or lan_ip())
    except Exception: pass
    want += tailscale_ips() + ["127.0.0.1"]
    errs = []
    for ip in want:
        if ip in _srv: continue
        try:
            srv = ThreadingHTTPServer((ip, port), H); _srv[ip] = srv
            threading.Thread(target=srv.serve_forever, daemon=True).start(); logging.info("listening on %s:%s", ip, port)
        except OSError as e: errs.append(f"{ip}: {e}"); logging.warning("can't listen on %s:%s: %s", ip, port, e)
    for ip in [i for i in _srv if i not in want]:   # address went away
        try: _srv.pop(ip).shutdown()
        except Exception: pass
    conflict = any("10048" in e or "in use" in e.lower() for e in errs)
    _status(state=("conflict" if conflict else "error") if errs and not _srv else "running", ips=list(_srv), port=port, error="; ".join(errs))
    return list(_srv)

def _parent_watch(pid):
    """--share child: quit when the NpuStudio app that started it closes."""
    import ctypes
    k = ctypes.windll.kernel32; h = k.OpenProcess(0x00100000, False, pid)   # SYNCHRONIZE
    if not h: os._exit(0)
    k.WaitForSingleObject(h, 0xFFFFFFFF); os._exit(0)

SHARE = "--share" in sys.argv
if not SHARE: _phones_load()
if __name__ == "__main__":
    if SHARE:   # laptop: share this PC's chips with the paired desktop, using the same pairing key (models/remote.json)
        logging.getLogger().handlers[0].setFormatter(logging.Formatter("%(asctime)s %(levelname)s [share] %(message)s"))
        rc = json.load(open(os.path.join(MD, "remote.json"))); c = {"key": rc["key"], "port": int(rc.get("share_port", PORT))}
        if "--parent" in sys.argv: threading.Thread(target=_parent_watch, args=(int(sys.argv[sys.argv.index("--parent") + 1]),), daemon=True).start()
    else: c = config()
    KEY = c["key"]; _status(state="starting", port=int(c["port"]))
    ips = bind_all(int(c["port"]), None if SHARE else c.get("bind"))
    for _ in range(16):          # just restarted: the old instance may still be freeing the port
        if _srv: break
        time.sleep(0.5); ips = bind_all(int(c["port"]), None if SHARE else c.get("bind"))
    def rebind():
        while True:
            time.sleep(30)
            try: bind_all(int(c["port"]), None if SHARE else c.get("bind"))
            except Exception: logging.exception("rebind")
    threading.Thread(target=rebind, daemon=True).start()
    if SHARE:
        while True: time.sleep(3600)
    ip = ", ".join(ips)
    logging.info("NpuStudio server listening on %s port %s (LAN + Tailscale only)", ip, c["port"])
    print(f"NpuStudio desktop worker on {ip} port {c['port']}\nPairing key: {c['key']}\nOn the laptop: Device > Desktop PC, then enter this address and key once.\nPhone pairing QR: run Show-PairingQR.bat (opens http://127.0.0.1:{c['port']}/pair).\nClose this window to stop.", flush=True)
    try:
        while True: time.sleep(3600)
    except KeyboardInterrupt: pass
    finally: _status(state="stopped")
