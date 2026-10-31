"""NpuStudio desktop link (laptop side). Device = "Desktop PC" sends heavy jobs to the desktop worker;
"Both machines" splits upscale tiles between this laptop and the desktop and sends single AI jobs to the desktop
(its 12 GB RTX 4070 is faster). If the desktop is offline everything runs locally and the status line says so.
Config: models/remote.json {"host": "192.168.x.x", "port": 47860, "key": "..."} (same key as the desktop's models/remote_server.json)."""
import os, json, time, logging, threading, urllib.request, urllib.error
APP = os.path.dirname(os.path.abspath(__file__)); CFG = os.path.join(APP, "models", "remote.json")
REMOTE = ("Desktop PC", "Both machines")
_h = {"t": 0, "ok": False, "info": {}}; _note = {"msg": ""}

def cfg():
    try: return json.load(open(CFG))
    except Exception: return {}
def configured(): c = cfg(); return bool(c.get("host") and c.get("key"))
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
    _note["msg"] = "Desktop PC offline - running on this laptop"; logging.warning(_note["msg"]); return False
def note(): return _note["msg"]

def call_sync(fn, *args, **kw):
    import wire
    data, _ = _req("call", wire.pack({"fn": fn, "args": list(args), "kw": kw}), timeout=120); return wire.unpack(data)

def call(fn, *args, progress=None, status=None, **kw):
    """Run a job on the desktop, polling progress; returns the result."""
    import wire
    try: jid = json.loads(_req("job", wire.pack({"fn": fn, "args": list(args), "kw": kw}), timeout=60)[0])["id"]
    except urllib.error.HTTPError as e: raise RuntimeError(f"Desktop PC refused the job: {e.read()[:200]!r}")
    if status: status(f"Sent to Desktop PC ({fn})...")
    last = None
    while True:
        time.sleep(0.3)
        st = json.loads(_req(f"job/{jid}", timeout=10)[0])
        if progress and st["n"] and (st["k"], st["n"]) != last: last = (st["k"], st["n"]); progress(st["k"], st["n"])
        if status and st["msg"]: status("Desktop PC: " + st["msg"])
        if st["state"] != "running": break
    try: data, _ = _req(f"job/{jid}/result", timeout=120)
    except urllib.error.HTTPError as e: raise RuntimeError("Desktop PC: " + json.loads(e.read()).get("error", "failed"))
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
    if os.environ.get("NPU_SERVER") or _rep[0]: return
    _rep[0] = True
    def loop():
        cc = None
        while True:
            time.sleep(3)
            if not configured() or health() is None: continue
            try:
                if cc is None:
                    import chipmon; cc = chipmon.ChipCounter()
                body = json.dumps({"name": name, "chips": cc.read()}, default=str).encode()
                r = urllib.request.Request(_url("api/v1/report"), data=body, method="POST",
                                           headers={"X-Key": cfg().get("key", ""), "Content-Type": "application/json"})
                urllib.request.urlopen(r, timeout=3).read()
            except Exception as e: logging.debug("report: %s", e)
    threading.Thread(target=loop, daemon=True, name="remote-report").start()
_rep = [False]

def install():
    """Route the CUDA diffusion engines (AI Change redraw/pose/recolor, Image Gen) to the desktop while a remote Device is
    picked. Wraps cudaeng's functions at run time (cudaeng.py itself is unchanged); falls back to local when offline."""
    if _patched[0] or os.environ.get("NPU_SERVER"): return
    try: import cudaeng
    except Exception: return
    for name in ("ip2p", "inpaint", "inpaint_cn", "inpaint_pose", "gen"):
        orig = getattr(cudaeng, name, None)
        if orig is None: continue
        def wrap(*a, _o=orig, _n=name, progress=None, status=None, **kw):
            if not is_local() and active():
                t = time.time(); r = call(_n, *a, progress=progress, status=status, **kw)
                logging.info("remote %s %.1fs", _n, time.time() - t); return r
            return _o(*a, progress=progress, status=status, **kw)
        setattr(cudaeng, name, wrap)
    oa = cudaeng.available; cudaeng.available = lambda: active() or oa()
    _patched[0] = True
    start_reporter()
