"""NpuStudio desktop link (laptop side). Device = "Desktop PC" sends heavy jobs to the desktop worker;
"Both machines" splits upscale tiles between this laptop and the desktop and sends single AI jobs to the desktop
(its 12 GB RTX 4070 is faster). If the desktop is offline everything runs locally and the status line says so.
Config: models/remote.json {"host": "192.168.x.x", "port": 47860, "key": "..."} (same key as the desktop's models/remote_server.json)."""
import os, json, time, logging, threading, urllib.request, urllib.error
APP = os.path.dirname(os.path.abspath(__file__)); CFG = os.path.join(APP, "models", "remote.json")
SRV = os.path.join(APP, "models", "remote_server.json"); LAPTOP = os.path.join(APP, "models", "laptop.json")
SHARE = os.path.join(APP, "models", "share.json")
REMOTE = ("Desktop PC", "Both machines", "Laptop NPU", "Laptop Intel Arc", "Laptop RTX 4070")
LAPTOP_DEV = {"Laptop NPU": "NPU", "Laptop Intel Arc": "Intel Arc", "Laptop RTX 4070": "NVIDIA"}
_h = {"t": 0, "ok": False, "info": {}}; _note = {"msg": ""}

def role():
    """'laptop' = paired to a desktop (models/remote.json); 'desktop' = runs NpuStudio-Server (models/remote_server.json)."""
    try:
        if json.load(open(CFG)).get("host"): return "laptop"
    except Exception: pass
    return "desktop" if os.path.exists(SRV) else "laptop"
def peer(): return "Desktop PC" if role() == "laptop" else "Laptop"

_disc = {"t": 0}
def _discover():
    """Desktop side: the laptop reports to this PC's server every 3 s (and says it shares its chips), so we learn its
    address from there - no extra pairing. The last address is kept in models/laptop.json."""
    if time.time() - _disc["t"] < 10: return
    _disc["t"] = time.time()
    try:
        s = json.load(open(SRV)); port = s.get("port", 47860)
        r = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/monitor", headers={"X-Key": s["key"]})
        o = json.load(urllib.request.urlopen(r, timeout=1.5)).get("others", {})
        for name, v in o.items():
            sv = v.get("serve")
            if sv and v.get("_ip"):
                c = {"host": v["_ip"], "port": int(sv.get("port", 47860)), "name": name}
                try: old = json.load(open(LAPTOP))
                except Exception: old = {}
                if {k: old.get(k) for k in c} != c: json.dump(c, open(LAPTOP, "w"), indent=1); _h["t"] = 0
                return
    except Exception as e: logging.debug("discover: %s", e)

def cfg():
    if role() == "laptop":
        try: return json.load(open(CFG))
        except Exception: return {}
    _discover()
    try: c = json.load(open(LAPTOP)); c["key"] = json.load(open(SRV))["key"]; return c   # the laptop uses the desktop's key
    except Exception: return {}
def configured(): c = cfg(); return bool(c.get("host") and c.get("key"))
def sharing():
    """Laptop setting 'Share this laptop's chips' (default on)."""
    try: return bool(json.load(open(SHARE)).get("share", True))
    except Exception: return True
def set_sharing(on): json.dump({"share": bool(on)}, open(SHARE, "w"))
def remote_device():
    """Chip to use on the other PC (Laptop NPU -> NPU); None = its Auto."""
    try:
        import devices; return LAPTOP_DEV.get(devices.get_raw())
    except Exception: return None
def save(host, key, port=47860):
    os.makedirs(os.path.dirname(CFG), exist_ok=True); json.dump({"host": host.strip(), "port": int(port), "key": key.strip()}, open(CFG, "w"), indent=1); _h["t"] = 0
def _url(p): c = cfg(); return f"http://{c['host']}:{c.get('port', 47860)}/{p}"
def _req(p, data=None, timeout=10):
    r = urllib.request.Request(_url(p), data=data, headers={"X-Key": cfg().get("key", "")}, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(r, timeout=timeout) as f: return f.read(), f.headers.get("Content-Type", "")

def health(force=False, max_age=5):
    """Quick check (cached a few seconds). Returns info dict or None."""
    if os.environ.get("NPU_SERVER") or not configured(): return None
    if force or time.time() - _h["t"] > max_age:
        try: _h["info"] = json.loads(_req("health", timeout=1.5)[0]); _h["ok"] = True
        except Exception: _h["ok"] = False; _h["info"] = {}
        _h["t"] = time.time()
    return _h["info"] if _h["ok"] else None
def online(): return health() is not None

def mode():
    import devices
    m = devices.get_raw(); return m if m in REMOTE else None
def active():
    """True when this job should go to the desktop. Sets a note when we fall back to local."""
    m = mode()
    if not m: return False
    if online(): _note["msg"] = ""; return True
    _note["msg"] = f"{peer()} offline - running on this PC"; logging.warning(_note["msg"]); return False
def note(): return _note["msg"]

def call_sync(fn, *args, **kw):
    import wire
    data, _ = _req("call", wire.pack({"fn": fn, "args": list(args), "kw": kw}), timeout=120); return wire.unpack(data)

def call(fn, *args, progress=None, status=None, on_text=None, **kw):
    """Run a job on the other PC, polling progress; returns the result. on_text gets streamed chat text."""
    import wire
    P = peer()
    try: jid = json.loads(_req("job", wire.pack({"fn": fn, "args": list(args), "kw": kw}), timeout=60)[0])["id"]
    except urllib.error.HTTPError as e: raise RuntimeError(f"{P} refused the job: {e.read()[:200]!r}")
    if status: status(f"Sent to {P} ({fn})...")
    last = None
    while True:
        time.sleep(0.3)
        st = json.loads(_req(f"job/{jid}", timeout=10)[0])
        if progress and st["n"] and (st["k"], st["n"]) != last: last = (st["k"], st["n"]); progress(st["k"], st["n"])
        if st["msg"].startswith("\x00"):
            if on_text: on_text(st["msg"][1:])
        elif status and st["msg"]: status(f"{P}: " + st["msg"])
        if st["state"] != "running": break
    try: data, _ = _req(f"job/{jid}/result", timeout=120)
    except urllib.error.HTTPError as e: raise RuntimeError(f"{P}: " + json.loads(e.read()).get("error", "failed"))
    return wire.unpack(data)

_tl = threading.local()
def is_local(): return bool(getattr(_tl, "local", False))
def both():
    """True in 'Both machines' mode with the desktop online (AI Change then splits its candidate versions)."""
    try: return mode() == "Both machines" and online()
    except Exception: return False
def run_split(make, n):
    """Run make(k) for k in 0..n-1: even k on this laptop, odd k on the desktop, in parallel. A job that fails remotely is
    redone locally. Inside make(), is_local() tells which side it is running on (pass progress callbacks only locally)."""
    res = [None] * n
    def worker(ks, local):
        _tl.local = local
        for k in ks:
            t = time.time()
            try: res[k] = make(k); logging.info("split job %d on %s %.1fs", k, "laptop" if local else "desktop", time.time() - t)
            except Exception: logging.exception("split job %d (%s)", k, "laptop" if local else "desktop")
        _tl.local = False
    th = threading.Thread(target=worker, args=(range(1, n, 2), False), daemon=True); th.start()
    worker(range(0, n, 2), True); th.join()
    miss = [k for k in range(n) if res[k] is None]
    if miss: worker(miss, True)
    return res

_patched = [False]
def start_reporter(name="Laptop"):
    """Push this PC's Chip Monitor numbers to the desktop every 3 s so the phone app's Monitor tab can show the laptop."""
    if os.environ.get("NPU_SERVER") or _rep[0] or role() != "laptop": return
    _rep[0] = True
    def loop():
        cc = None
        while True:
            time.sleep(3)
            if not configured() or health() is None: continue
            try:
                if cc is None:
                    import chipmon; cc = chipmon.ChipCounter()
                rep = {"name": name, "chips": cc.read()}
                if sharing(): rep["serve"] = {"port": int(cfg().get("share_port", 47860))}
                body = json.dumps(rep, default=str).encode()
                r = urllib.request.Request(_url("api/v1/report"), data=body, method="POST",
                                           headers={"X-Key": cfg().get("key", ""), "Content-Type": "application/json"})
                urllib.request.urlopen(r, timeout=3).read()
            except Exception as e: logging.debug("report: %s", e)
    threading.Thread(target=loop, daemon=True, name="remote-report").start()
_rep = [False]

def _params(fn):
    import inspect
    try: return [p.name for p in inspect.signature(fn).parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except Exception: return []

def install():
    """Route the CUDA diffusion engines (AI Change redraw/pose/recolor, Image Gen) to the desktop while a remote Device is
    picked. Wraps cudaeng's functions at run time (cudaeng.py itself is unchanged); falls back to local when offline."""
    if _patched[0] or os.environ.get("NPU_SERVER"): return
    try: import cudaeng
    except Exception: return
    for name in ("ip2p", "inpaint", "inpaint_cn", "inpaint_pose", "gen"):
        orig = getattr(cudaeng, name, None)
        if orig is None: continue
        def wrap(*a, _o=orig, _n=name, _sig=_params(orig), progress=None, status=None, **kw):
            a = list(a)     # progress/status passed positionally (e.g. imagegen -> gen(prompt, w, h, steps, seed, progress)) -> keywords
            for k in ("status", "progress"):
                i = _sig.index(k) if k in _sig else -1
                if 0 <= i < len(a):
                    v = a.pop(i)
                    if k == "progress" and progress is None: progress = v
                    if k == "status" and status is None: status = v
            if not is_local() and active():
                t = time.time(); r = call(_n, *a, progress=progress, status=status, **kw)
                logging.info("remote %s %.1fs", _n, time.time() - t); return r
            return _o(*a, progress=progress, status=status, **kw)
        setattr(cudaeng, name, wrap)
    oa = cudaeng.available; cudaeng.available = lambda: active() or oa()
    _patched[0] = True
    start_reporter()

# ---------------- this PC's own worker server (Server button) ----------------
_child = {"p": None}
def local_key():
    try: return json.load(open(CFG))["key"] if role() == "laptop" else json.load(open(SRV))["key"]
    except Exception: return ""
def local_status():
    """'running' | 'starting' | 'conflict' | 'error' | 'stopped' for this PC's worker server."""
    try:
        r = urllib.request.Request("http://127.0.0.1:47860/health", headers={"X-Key": local_key()})
        urllib.request.urlopen(r, timeout=1).read(); return "running"
    except urllib.error.HTTPError: return "conflict"      # something answers on the port but rejects our key
    except Exception: pass
    st = {}
    try: st = json.load(open(os.path.join(APP, "models", "server_status.json" + (".share" if role() == "laptop" else ""))))
    except Exception: pass
    p = _child["p"]
    if st.get("state") in ("conflict", "error") and time.time() - st.get("t", 0) < 120: return st["state"]
    if p is not None and p.poll() is None: return "starting"
    if p is not None and p.poll() is not None: return "error"
    return "stopped"
def local_error():
    try: return json.load(open(os.path.join(APP, "models", "server_status.json" + (".share" if role() == "laptop" else "")))).get("error", "")
    except Exception: return ""

def start_share():
    """Laptop: run worker_server.py --share as a background child (quits with the app). Same pairing key, LAN + Tailscale."""
    import subprocess, sys
    if role() != "laptop" or not configured() or not sharing(): return False
    if _child["p"] is not None and _child["p"].poll() is None: return True
    if local_status() == "running": return True
    py = sys.executable
    if py.lower().endswith("python.exe") and os.path.exists(py[:-10] + "pythonw.exe"): py = py[:-10] + "pythonw.exe"
    _child["p"] = subprocess.Popen([py, os.path.join(APP, "worker_server.py"), "--share", "--parent", str(os.getpid())],
                                   cwd=APP, creationflags=0x08000000, close_fds=True)
    logging.info("share server started pid %s", _child["p"].pid); return True
def stop_share():
    p = _child["p"]
    if p is not None and p.poll() is None: p.terminate()
    _child["p"] = None
def start_desktop_server():
    """Desktop: open NpuStudio-Server.bat in its own window (same as double-clicking it)."""
    import subprocess
    subprocess.Popen(["cmd", "/c", "start", "NpuStudio Server", "/D", APP, "NpuStudio-Server.bat"], creationflags=0x08000000)
