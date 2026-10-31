"""NpuStudio desktop worker: lets the laptop send heavy jobs (Image Gen, upscale, AI Change, background removal, erase)
to this PC over the home network. LAN only: binds to this PC's home-network address, refuses non-private callers,
and every request needs the pairing key from models/remote_server.json. Start with NpuStudio-Server.bat."""
import os, sys, json, time, uuid, secrets, socket, logging, threading, ipaddress, traceback, inspect
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
def upscaler(name):
    import npustudio
    with _up_lock:
        if name not in _up: _up[name] = npustudio.Upscaler(name)
        return _up[name]
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
    if fn == "upscale":          # whole picture: (model name, image, scale)
        name, im, scale = args; u = upscaler(name); status(f"Upscaling on desktop {u.device}...")
        return u.upscale_to(im, int(scale), progress)
    if fn == "tile":             # one 4x tile for split ("Both machines") upscales
        name, tile = args; u = upscaler(name); return u.workers[0][1](tile)
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
        if self.path.split("?")[0].rstrip("/") in ("/app", "") or self.path.startswith("/app/"):
            return serve_app(self)   # the mobile web UI holds no secrets; its API calls carry the key
        if not self._ok(): return self._send(403, {"error": "bad key"})
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

# ================= mobile web UI (/app) =================
WEB = os.path.join(APP, "webapp")
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
        ".png": "image/png", ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json", ".ico": "image/x-icon"}
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

def api_job(kind, a, j):
    """Runs one JSON job. j = job dict (progress k/n, msg, text, confirm...). Returns result dict."""
    def prog(k, n): j["k"], j["n"] = k, n
    def stat(m): j["msg"] = str(m)
    if kind == "gen":
        import imagegen, cudaeng
        tw, th = int(a.get("width", 768)), int(a.get("height", 768)); (w, h), up = imagegen.plan_for(tw, th)
        im = cudaeng.gen(a["prompt"], w, h, int(a.get("steps", 4)), int(a.get("seed", 42)), prog, stat)
        if up > 1: stat("upscaling..."); im = upscaler("Sharp (Real-ESRGAN x4plus)").upscale_to(im, up, prog)
        return {"image": imgb64(imagegen.fit_crop(im, tw, th)), "device": "NVIDIA"}
    if kind == "upscale":
        name = "Sharp (Real-ESRGAN x4plus)" if a.get("model", "sharp") == "sharp" else "Fast (Intel SR-1032)"
        u = upscaler(name); im = u.upscale_to(b64img(a["image"]).convert("RGB"), int(a.get("scale", 4)), prog)
        return {"image": imgb64(im, a.get("format", "PNG")), "device": u.device}
    if kind in ("remove_bg", "erase"):
        im = b64img(a["image"])
        r, d = ai().remove_bg(im, stat) if kind == "remove_bg" else ai().erase(im, b64img(a["mask"]).convert("L"), stat)
        return {"image": imgb64(r), "device": str(d)}
    if kind == "change":     # AI Change: free-text instruction, optional mask (white = area to change)
        import aichange
        if _chg["c"] is None: _chg["c"] = aichange.Changer()
        c = _chg["c"]; im = b64img(a["image"]).convert("RGB"); instr = a["instruction"]
        stat("analyzing picture..."); ana = c.analyze(im, stat)
        mask = b64img(a["mask"]).convert("L") if a.get("mask") else None
        if mask is None:
            try:
                tgt = aichange.parse(instr); mask = aichange.Seg().mask(im, aichange.targets_for(tgt[0] if isinstance(tgt, tuple) else tgt, ana)) if tgt else None
            except Exception: logging.exception("api auto-mask"); mask = None
        if mask is not None and mask.getbbox():
            r = c.edit_masked(ai(), im, mask, instr, ana, progress=prog, status=stat, method=a.get("method", "Auto"))
        else: r = c.edit_whole(im, instr, ana, progress=prog, status=stat)
        return {"image": imgb64(r[0]), "device": str(r[1]), "method": r[2], "prompt": r[3], "analysis": ana}
    if kind == "chat":       # allowlisted PC actions run on THIS PC; a CONFIRM action waits for POST .../confirm
        eng = chat_engine(stat); out = []
        def on_text(t): out.append(t); j["text"] = "".join(out)
        def confirm(q):
            j["confirm"] = q; j["state"] = "confirm"; ev = j["ev"] = threading.Event()
            ok = ev.wait(120); j["state"] = "running"; j["confirm"] = None
            return bool(ok and j.get("answer"))
        src, act, res, n, dt = eng.handle(a["message"], on_text, confirm=confirm)
        return {"text": j.get("text", ""), "action": str(act) if act else None, "action_result": res, "device": eng.device}
    raise ValueError("unknown job type " + kind)

def monitor():
    d = {"desktop": gpu_info()}
    try:
        if _mon["c"] is None:
            import chipmon; _mon["c"] = chipmon.ChipCounter()
        d["chips"] = _mon["c"].read()
    except Exception as e: d["chips"] = {}; d["chips_error"] = str(e)
    return d

def api_get(h, p):
    if p == ["health"]: return h._send(200, gpu_info())
    if p == ["monitor"]: return h._send(200, monitor())
    if len(p) >= 2 and p[0] == "jobs":
        j = JOBS.get(p[1])
        if not j: return h._send(404, {"error": "no such job"})
        if len(p) == 3 and p[2] == "result":
            if j["state"] in ("running", "confirm"): return h._send(409, {"error": "not done"})
            JOBS.pop(p[1], None)
            return h._send(500 if j["state"] == "error" else 200, {"error": j["err"]} if j["state"] == "error" else j["res"])
        return h._send(200, {"id": p[1], "type": j.get("type"), "state": j["state"], "progress": j["k"], "total": j["n"],
                             "message": j["msg"], "text": j.get("text", ""), "confirm": j.get("confirm")})
    h._send(404, {"error": "not found"})

def api_post(h, p, data):
    try: a = json.loads(data or b"{}")
    except Exception: return h._send(400, {"error": "body must be JSON"})
    if p == ["chat", "new"]:
        if _chat["eng"]: _chat["eng"].new_chat()
        return h._send(200, {"ok": True})
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
            except Exception as e: logging.exception("api job %s", kind); j["err"] = f"{type(e).__name__}: {e}"; j["state"] = "error"
            finally: _busy[0] -= 1
        threading.Thread(target=work, daemon=True).start(); return h._send(200, {"id": jid})
    h._send(404, {"error": "not found"})

if __name__ == "__main__":
    c = config(); KEY = c["key"]; ips = [c.get("bind") or lan_ip()] + tailscale_ips()
    for ip in ips:
        srv = ThreadingHTTPServer((ip, int(c["port"])), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    ip = ", ".join(ips)
    logging.info("NpuStudio server listening on %s port %s (LAN + Tailscale only)", ip, c["port"])
    print(f"NpuStudio desktop worker on {ip} port {c['port']}\nPairing key: {c['key']}\nOn the laptop: Device > Desktop PC, then enter this address and key once.\nClose this window to stop.", flush=True)
    try:
        while True: time.sleep(3600)
    except KeyboardInterrupt: pass
