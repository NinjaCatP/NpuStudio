import os, sys, time, threading, logging, traceback, re, queue, json
# pythonw has no console: sys.stdout/stderr are None and tqdm/print crash ("'NoneType' object has no attribute 'write'")
for _n in ("stdout", "stderr"):
    if getattr(sys, _n) is None: setattr(sys, _n, open(os.devnull, "w", encoding="utf-8"))
import numpy as np
from PIL import Image, ImageTk, ImageFilter
APP = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=os.path.join(APP, "error.log"), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
MODEL = os.path.join(APP, "models", "single-image-super-resolution-1032.xml")
LH, LW, HH, HW = 270, 480, 1080, 1920
SCALE = float(os.environ.get("NPU_SCALE", "255"))
BGR = os.environ.get("NPU_BGR", "1") == "1"

MODELS = {
    "Fast (Intel SR-1032)": ("fast", "single-image-super-resolution-1032.xml"),
    "Sharp (Real-ESRGAN x4plus)": ("sharp", "realesrgan-x4plus-256.xml"),
}
MAX_BYTES = 18 * 2**30  # memory budget for the upscaled image (Jason: 18 GB)
Image.MAX_IMAGE_PIXELS = None

def avail_ram():
    try:
        import ctypes
        class M(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                       ("tp", "ap", "tf", "af", "tv", "av", "ae")]
        m = M(); m.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); return m.ap
    except Exception: return 8 << 30

def plan(scale):
    """List of per-pass output factors (each pass runs the 4x model)."""
    return {2: [2], 4: [4], 8: [4, 2], 16: [4, 4]}[scale]

def need_bytes(w, h, scale):
    # PIL keeps RGB at 4 bytes/px; tiles are pasted straight into one preallocated image (no full-size copies).
    o = w * h * scale * scale * 4
    prev = {2: 0, 4: 0, 8: o / 4, 16: o / 16}[scale]  # previous pass held during the 2nd pass
    return int(o + prev + 200 * 2**20)  # + working overhead

class Upscaler:
    def __init__(self, name="Fast (Intel SR-1032)", pref=None):
        import openvino as ov, devices
        self.kind, fn = MODELS[name]; self.name = name
        self.core = ov.Core()
        try: self.core.set_property({"CACHE_DIR": os.path.join(APP, "models", "cache")})
        except Exception: pass
        self.err = []; self.workers = []  # (label, infer_fn)
        import remote; self.remote = None; rm = remote.mode()
        if rm and remote.active():
            if rm != "Both machines":   # whole picture goes to the other PC (Desktop PC, or Laptop NPU/Arc/RTX on the desktop)
                self.remote = "whole"; self.device = rm; self.th = self.tw = 256; self.m = 16
                self.lock = threading.Lock(); logging.info("Upscaler %s on %s", name, rm); return
            self.remote = "split"
        model = self.core.read_model(os.path.join(APP, "models", fn))
        feat = "upscale_" + self.kind; labels = devices.plan(feat, pref)
        if not labels: raise RuntimeError(devices.unsupported("Upscale", pref))
        many = (pref or devices.get()) == "All"
        for lab in labels:
            try:
                if lab == "NVIDIA" and self.kind == "sharp":
                    import cudaeng
                    if not cudaeng.available(): raise RuntimeError("CUDA not available")
                    cudaeng.esr_model(); self.workers.append((lab, cudaeng.esr_tile))
                else:
                    cm = self.core.compile_model(model, devices.ov(lab)); req = cm.create_infer_request()
                    self.workers.append((lab, (lambda t, r=req, o=cm.outputs[0]: self._ov_infer(r, o, t))))
                if not many: break
            except Exception as e:
                self.err.append(f"{lab}: {e}"); logging.warning("upscaler %s on %s failed: %s", name, lab, e)
        self.share = 0.6   # Both machines: part of the picture the desktop does (learned from each run's timings)
        if not self.workers: raise RuntimeError("No device worked: " + "; ".join(self.err))
        self.device = "+".join(l for l, _ in self.workers) + ("+Desktop PC" if self.remote == "split" else "")
        if self.kind == "fast": self.th, self.tw, self.m = LH, LW, 8
        else: self.th = self.tw = 256; self.m = 16
        self.lock = threading.Lock(); logging.info("Upscaler %s on %s", name, self.device)

    def _prep(self, a):
        if BGR: a = a[..., ::-1]
        a = a.astype(np.float32)
        if SCALE == 1: a = a / 255.0
        return np.ascontiguousarray(a.transpose(2, 0, 1)[None])

    def _ov_infer(self, req, outp, tile):
        if self.kind == "fast":
            bic = np.asarray(Image.fromarray(tile).resize((HW, HH), Image.BICUBIC))
            r = req.infer({0: self._prep(tile), 1: self._prep(bic)})[outp][0].transpose(1, 2, 0)
            if BGR: r = r[..., ::-1]
        else:
            x = np.ascontiguousarray(tile.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
            r = req.infer({0: x})[outp][0].transpose(1, 2, 0)
        return np.clip(r * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def grid(self, W, H):
        PH, PW = max(H, self.th), max(W, self.tw); st = 2 * self.m
        ys = list(range(0, PH - self.th + 1, self.th - st)); xs = list(range(0, PW - self.tw + 1, self.tw - st))
        if ys[-1] + self.th < PH: ys.append(PH - self.th)
        if xs[-1] + self.tw < PW: xs.append(PW - self.tw)
        return PH, PW, ys, xs

    def ntiles(self, W, H):
        _, _, ys, xs = self.grid(W, H); return len(ys) * len(xs)

    def upscale(self, img, progress=None, f=4):
        """One pass: run the 4x model on overlapping tiles; f=2 downsizes each tile (Lanczos) as it goes."""
        img = img.convert("RGB"); W, H = img.size; src = np.asarray(img)
        PH, PW, ys, xs = self.grid(W, H)
        pad = np.pad(src, ((0, PH - H), (0, PW - W), (0, 0)), mode="edge") if (PH, PW) != (H, W) else src
        OW, OH = W * f, H * f
        out = Image.new("RGB", (OW, OH)); del src
        n = len(ys) * len(xs); m = self.m
        jobs = queue.Queue()
        for y0 in ys:
            for x0 in xs: jobs.put((y0, x0))
        plock = threading.Lock(); cnt = {"k": 0, "err": None}; self.tiles_by = {l: 0 for l, _ in self.workers}
        def work(lab, infer):
            while cnt["err"] is None:
                try: y0, x0 = jobs.get_nowait()
                except queue.Empty: return
                try:
                    r = infer(np.ascontiguousarray(pad[y0:y0 + self.th, x0:x0 + self.tw]))
                    if f != 4: r = np.asarray(Image.fromarray(r).resize((self.tw * f, self.th * f), Image.LANCZOS))
                    t = 0 if y0 == 0 else m; l = 0 if x0 == 0 else m
                    ty, tx = (y0 + t) * f, (x0 + l) * f
                    ch, cw = min((y0 + self.th) * f, OH) - ty, min((x0 + self.tw) * f, OW) - tx
                    piece = Image.fromarray(np.ascontiguousarray(r[t * f:t * f + ch, l * f:l * f + cw])) if ch > 0 and cw > 0 else None
                    with plock:
                        if piece is not None: out.paste(piece, (tx, ty))
                        cnt["k"] += 1; self.tiles_by[lab] += 1; k = cnt["k"]
                    if progress: progress(k, n)
                except Exception as e:
                    cnt["err"] = e
        if len(self.workers) == 1: work(*self.workers[0])
        else:
            ths = [threading.Thread(target=work, args=w, daemon=True) for w in self.workers]
            for t in ths: t.start()
            for t in ths: t.join()
        if cnt["err"] is not None: raise cnt["err"]
        del pad
        return out

    def upscale_to(self, img, scale, progress=None):
        if self.remote == "whole":
            import remote
            if remote.online(): return remote.call("upscale", self.name, img.convert("RGB"), int(scale), progress=progress, device=remote.remote_device())
            logging.warning("%s offline - upscaling on this PC", remote.peer()); remote._note["msg"] = f"{remote.peer()} offline - running on this PC"
            if getattr(self, "_loc", None) is None: self._loc = Upscaler(self.name)   # remote is offline, so this builds on local chips
            return self._loc.upscale_to(img, scale, progress)
        if self.remote == "split":
            import remote
            if remote.online() and img.size[1] >= 64: return self._split(img, scale, progress)
        with self.lock: return self._upscale_to(img, scale, progress)

    def _split(self, img, scale, progress=None):
        """Both machines: the desktop upscales the top part (one request), this laptop the bottom part, at the same time.
        Parts overlap by 16 px and are cut in the middle of the overlap, so there is no seam."""
        import remote
        img = img.convert("RGB"); W, H = img.size; ov = 16
        cut = max(ov, min(H - ov, int(H * self.share)))
        top = img.crop((0, 0, W, min(H, cut + ov))); bot = img.crop((0, max(0, cut - ov), W, H))
        res = {}; pr = {"r": (0, 1), "l": (0, 1)}; t0 = time.time()
        def rep_():
            if progress:
                (a, b), (c, d) = pr["r"], pr["l"]; progress(int(1000 * (a / b * self.share + c / d * (1 - self.share))), 1000)
        def rem():
            try: res["r"] = remote.call("upscale", self.name, top, int(scale), progress=lambda k, n: (pr.__setitem__("r", (k, n)), rep_()))
            except Exception as e: res["re"] = e
            res["rt"] = time.time() - t0
        th = threading.Thread(target=rem, daemon=True); th.start()
        with self.lock: lo = self._upscale_to(bot, scale, lambda k, n: (pr.__setitem__("l", (k, n)), rep_()))
        lt = time.time() - t0; th.join()
        if "re" in res:   # desktop failed mid-job: do its part here
            logging.warning("Both machines: desktop part failed (%s), finishing locally", res["re"])
            with self.lock: res["r"] = self._upscale_to(top, scale); res["rt"] = 1e9
        out = Image.new("RGB", (W * scale, H * scale))
        out.paste(res["r"].crop((0, 0, W * scale, cut * scale)), (0, 0))
        out.paste(lo.crop((0, (cut - max(0, cut - ov)) * scale, W * scale, lo.size[1])), (0, cut * scale))
        # learn the split: each side's speed = its rows / its time
        rs, ls = cut / max(res["rt"], 1e-3), (H - cut) / max(lt, 1e-3); self.share = min(0.9, max(0.1, 0.5 * self.share + 0.5 * rs / (rs + ls)))
        self.tiles_by = {"Desktop PC rows": cut, "local rows": H - cut, "desktop s": round(res["rt"], 2), "local s": round(lt, 2)}
        logging.info("Both machines upscale %s: desktop %d rows %.1fs, local %d rows %.1fs, next share %.2f", img.size, cut, res["rt"], H - cut, lt, self.share)
        return out

    def _upscale_to(self, img, scale, progress=None):
        passes = plan(scale); W, H = img.size; sizes = []; w, h = W, H
        for f in passes: sizes.append(self.ntiles(w, h)); w, h = w * f, h * f
        tot = sum(sizes); done = 0; cur = img
        for i, f in enumerate(passes):
            base = done
            prev = cur; cur = None
            cur = self.upscale(prev, (lambda k, n, b=base: progress(b + k, tot)) if progress else None, f)
            del prev
            done += sizes[i]
        return cur

def denoise(img):
    return img.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.SMOOTH)

class NpuCounter:
    def __init__(self):
        import win32pdh
        self.p = win32pdh
        self.luid = self._find_luid()
        self.q = None; self.cs = []; self.mem = []; self.built = 0
        self._build()

    def _build(self):
        p = self.p
        if self.q:
            try: p.CloseQuery(self.q)
            except Exception: pass
        self.q = p.OpenQuery(); self.cs = []; self.mem = []
        if self.luid:
            try:
                p.EnumObjectItems(None, None, "GPU Engine", p.PERF_DETAIL_WIZARD, 1)
            except Exception: pass
            for path in p.ExpandCounterPath(r"\GPU Engine(*%s*engtype_Compute)\Utilization Percentage" % self.luid):
                try: self.cs.append(p.AddCounter(self.q, path))
                except Exception: pass
            for nm in ("Dedicated Usage", "Shared Usage"):
                for path in p.ExpandCounterPath(r"\GPU Adapter Memory(*%s*)\%s" % (self.luid, nm)):
                    try: self.mem.append(p.AddCounter(self.q, path))
                    except Exception: pass
        p.CollectQueryData(self.q)
        self.built = time.time(); self.fresh = True

    def _find_luid(self):
        _, inst = self.p.EnumObjectItems(None, None, "GPU Engine", self.p.PERF_DETAIL_WIZARD)
        types = {}
        for i in inst:
            m = re.search(r"(luid_0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+).*engtype_(\w*)", i)
            if m: types.setdefault(m.group(1), set()).add(m.group(2))
        for l, t in types.items():
            if t == {"Compute"}: logging.info("NPU luid %s", l); return l
        logging.warning("No compute-only adapter: %s", types); return None

    def read(self):
        # processes come and go, so refresh the per-process counter list every 3 s
        if time.time() - self.built > 3:
            self._build(); time.sleep(0.1)
        self.p.CollectQueryData(self.q); u = 0.0; m = 0.0
        for c in self.cs:
            try: u += self.p.GetFormattedCounterValue(c, self.p.PDH_FMT_DOUBLE)[1]
            except Exception: pass
        for c in self.mem:
            try: m += self.p.GetFormattedCounterValue(c, self.p.PDH_FMT_DOUBLE)[1]
            except Exception: pass
        return min(u, 100.0), m / 1048576

def headless_test(path, outpath):
    c = NpuCounter(); up = Upscaler(); peak = [0.0]; stop = [False]
    def mon():
        while not stop[0]:
            peak[0] = max(peak[0], c.read()[0]); time.sleep(0.25)
    th = threading.Thread(target=mon, daemon=True); th.start()
    img = Image.open(path); t = time.time()
    time.sleep(3.5)
    for _ in range(6): res = up.upscale(img)
    dt = time.time() - t; stop[0] = True; th.join()
    res.save(outpath)
    ref = np.asarray(img.convert("RGB").resize(res.size, Image.BICUBIC)).astype(float)
    diff = np.abs(ref - np.asarray(res).astype(float)).mean()
    print(f"device={up.device} luid={c.luid} peak_npu={peak[0]:.1f}% time3x={dt:.1f}s mean_diff_vs_bicubic={diff:.2f} out={res.size}")

CHAT_REPO = "OpenVINO/Phi-3.5-mini-instruct-int4-gq-ov"
CHAT_DIR = os.path.join(APP, "models", CHAT_REPO.split("/")[1])
CHAT_SIZE = 2.08e9

def dir_size(d):
    t = 0
    for r, _, fs in os.walk(d):
        for f in fs:
            try: t += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return t

def chat_download(progress=None):
    if os.path.exists(os.path.join(CHAT_DIR, "done.txt")): return
    from huggingface_hub import snapshot_download
    os.makedirs(CHAT_DIR, exist_ok=True); stop = [False]
    def mon():
        while not stop[0]:
            if progress: progress(min(dir_size(CHAT_DIR) / CHAT_SIZE, 0.99))
            time.sleep(1)
    threading.Thread(target=mon, daemon=True).start()
    try: snapshot_download(CHAT_REPO, local_dir=CHAT_DIR)
    finally: stop[0] = True
    open(os.path.join(CHAT_DIR, "done.txt"), "w").write("ok")

def pair_info():
    """Desktop address + key for pairing the Android app (the phone always talks to the desktop server)."""
    import urllib.request, urllib.parse, json
    srv = os.path.join(APP, "models", "remote_server.json"); cli = os.path.join(APP, "models", "remote.json")
    if os.path.exists(srv):
        c = json.load(open(srv)); host, port, key = "127.0.0.1", c.get("port", 47860), c["key"]
    elif os.path.exists(cli):
        c = json.load(open(cli)); host, port, key = c["host"], c.get("port", 47860), c["key"]
    else: raise RuntimeError("This PC isn't the desktop server and isn't linked to one (Settings > Desktop link).")
    r = urllib.request.Request(f"http://{host}:{port}/api/v1/info", headers={"X-Key": key})
    try: info = json.load(urllib.request.urlopen(r, timeout=5))
    except Exception as e: raise RuntimeError(f"Can't reach the desktop server at {host}:{port} ({e}). Start NpuStudio-Server on the desktop.")
    lan = info.get("lan") or host; ts = info.get("tailscale") or []
    q = {"host": lan, "port": port, "key": key}
    if ts: q["remote"] = ts[0]
    return "npustudio://pair?" + urllib.parse.urlencode(q), lan, ts, port

def open_pair_window(root):
    import tkinter as tk
    from tkinter import ttk, messagebox
    try:
        uri, lan, ts, port = pair_info()
        import segno, io
        from PIL import Image, ImageTk
        b = io.BytesIO(); segno.make(uri, error="m").save(b, kind="png", scale=7, border=3); b.seek(0)
        ph = ImageTk.PhotoImage(Image.open(b))
    except Exception as e:
        messagebox.showerror("Pair phone", str(e), parent=root); return
    w = tk.Toplevel(root); w.title("NpuStudio - Pair phone"); w.configure(bg="white")
    l = tk.Label(w, image=ph, bg="white"); l.image = ph; l.pack(padx=12, pady=(12, 4))
    txt = f"Home: {lan}:{port}" + (f"\nTailscale: {ts[0]}:{port} (switches automatically when away)" if ts else "\nTailscale: not detected on the desktop (home Wi-Fi only)")
    tk.Label(w, text="In the NpuStudio phone app tap Scan QR (or Connection > Re-pair).\n" + txt, bg="white", fg="#111", justify="center").pack(padx=12)
    tk.Label(w, text="This code contains the pairing key - don't share it.", bg="white", fg="#b00").pack(pady=(4, 6))
    ttk.Button(w, text="Close", command=w.destroy).pack(pady=(0, 12))

class ChatEngine:
    def __init__(self, status=None, lazy=False):
        self.status = status; self.err = []; self.pipe = None; self.device = None; self.load_time = 0.0; self.stop = False
        if lazy:     # desktop using the laptop's NPU: no local model until a fallback needs it
            self.device = "Laptop"; self.target = "pc"; self._rnew = True; return
        self._load_local()

    def _load_local(self):
        status = self.status
        import openvino as ov, openvino_genai as og
        self.og = og; self.err = []
        avail = ov.Core().available_devices
        for d in ["NPU", "GPU", "CPU"]:
            if d not in avail: continue
            if status: status(f"Loading model on {d} (first time on NPU can take a few minutes)...")
            try:
                cfg = {"CACHE_DIR": os.path.join(APP, "npucache")}
                if d == "NPU": cfg.update(MAX_PROMPT_LEN=2048, MIN_RESPONSE_LEN=512)
                t = time.time(); self.pipe = og.LLMPipeline(CHAT_DIR, d, **cfg)
                self.device = d; self.load_time = time.time() - t; break
            except Exception as e:
                self.err.append(f"{d}: {e}"); logging.warning("LLM on %s failed: %s", d, e)
        if not self.pipe: raise RuntimeError("No device worked: " + "; ".join(self.err))
        logging.info("Chat on %s (load %.1fs)", self.device, self.load_time)
        self.stop = False; self.new_chat()

    def new_chat(self):
        if self.pipe is None:
            self.target = getattr(self, "want_target", "pc"); self._rnew = True; return
        try: self.pipe.finish_chat()
        except Exception: pass
        import actions
        self.target = getattr(self, "want_target", "pc"); self._rnew = True
        self.pipe.start_chat(actions.PHONE_PROMPT if self.target == "phone" else actions.TOOLS_PROMPT)

    def handle(self, q, on_text, confirm=None, timer_cb=None, on_action=None, target="pc"):
        """Router first, then the model. Returns (source, action, result, tokens, seconds).
        target="phone": actions are NOT run here; the result is ("phone", action) for the phone app to run."""
        import actions
        phone = target == "phone"
        if getattr(self, "target", "pc") != target:   # switch the system prompt (new conversation for the other device)
            self.want_target = target; self.new_chat()
        run = (lambda a: ("phone", a)) if phone else (lambda a: actions.execute(a, confirm, timer_cb))
        a = actions.route_phone(q) if phone else actions.route(q)
        if a:
            if on_action: on_action(a)
            return "router", a, run(a), 0, 0.0
            if on_action: on_action(a)
            return "router", a, actions.execute(a, confirm, timer_cb), 0, 0.0
        buf = []; state = {"mode": None}
        def tok(sub):
            buf.append(sub)
            if state["mode"] is None:
                head = "".join(buf).lstrip()
                if head.startswith(("ACTION", "{")) or (len(head) < 7 and "ACTION:".startswith(head)):
                    if head.startswith(("ACTION:", "{")): state["mode"] = "action"
                    return
                state["mode"] = "text"; on_text("".join(buf))
            elif state["mode"] == "text": on_text(sub)
        n, dt = self.ask(q, tok)
        full = "".join(buf)
        a = actions.parse_action(full, actions.PHONE if phone else None)
        if a and state["mode"] != "text":  # only act when the reply *is* an action, not an answer that rambles into one
            if on_action: on_action(a)
            return "model", a, run(a), n, dt
        if state["mode"] != "text":
            on_text(full if not full.lstrip().startswith(("ACTION", "{")) else "Sorry, I'm not sure how to do that. Try wording it like \"open notepad\" or \"volume to 30\".")
        return "model", None, None, n, dt

    def _use_peer(self):
        """Desktop with Device = Laptop NPU/Arc/RTX: chat runs on the laptop's LLM (its NPU), actions still run here."""
        try:
            import remote, devices
            return chat_peer(devices.get_raw())
        except Exception: return False

    def ask(self, text, on_token, max_new=400):
        if self._use_peer():
            import remote
            seen = [""]
            def on_text(full):
                if full.startswith(seen[0]) and len(full) > len(seen[0]): on_token(full[len(seen[0]):]); seen[0] = full
            t = time.time()
            try:
                r = remote.call("chat_ask", text, getattr(self, "target", "pc"), on_text=on_text, new=getattr(self, "_rnew", True))
                self._rnew = False; on_text(r["text"]); self.last_device = "Laptop " + str(r.get("device"))
                return r.get("tokens", 0), time.time() - t
            except Exception as e:
                logging.warning("laptop chat failed (%s) - answering here", e)
                if seen[0]: return 0, time.time() - t
        if self.pipe is None:
            if self.status: self.status("Laptop offline - loading the chat model on this PC...")
            chat_download(); self._load_local()
        self.last_device = self.device
        self.stop = False; n = [0]
        def cb(sub):
            n[0] += 1; on_token(sub)
            return self.og.StreamingStatus.STOP if self.stop else self.og.StreamingStatus.RUNNING
        cfg = self.og.GenerationConfig(); cfg.max_new_tokens = max_new; cfg.do_sample = False; cfg.repetition_penalty = float(os.environ.get("NPU_REP", "1.0")); cfg.stop_strings = {"<|end|>", "<|user|>", "<|endoftext|>"}
        t = time.time(); self.pipe.generate(text, cfg, cb)
        return n[0], time.time() - t

def chat_peer(raw=None):
    """Desktop: chat runs on the laptop's LLM (its NPU) when Device is Auto or a Laptop chip and the laptop is online."""
    try:
        import remote, devices
        raw = raw or devices.get_raw()
        return remote.role() == "desktop" and (raw in remote.LAPTOP_DEV or raw in ("Auto", None, "")) and remote.online()
    except Exception: return False

def chat_test(q):
    c = NpuCounter(); peak = [0.0]; stop = [False]
    def mon():
        while not stop[0]:
            peak[0] = max(peak[0], c.read()[0]); time.sleep(0.25)
    chat_download(lambda p: None)
    eng = ChatEngine(print)
    print(f"device={eng.device} load={eng.load_time:.1f}s", flush=True)
    th = threading.Thread(target=mon, daemon=True); th.start(); time.sleep(3.5)
    for i, p in enumerate([q, "Now say that again in five words or fewer.", "NEW", "What is 12 times 7?"]):
        if p == "NEW": eng.new_chat(); print("--- New chat ---"); continue
        out = []; n, dt = eng.ask(p, out.append, 200)
        print(f"Q: {p}\nA: {''.join(out).strip()}\n[{n} tokens, {n/dt:.1f} tok/s]", flush=True)
    stop[0] = True; th.join()
    print(f"peak_npu={peak[0]:.1f}% errors={eng.err}")

def action_test():
    import actions
    phrases = ["open notepad", "close notepad", "set the volume to 30", "turn on dark mode", "brightness 50",
               "remind me to stretch in 10 minutes", "what time is it", "how much battery do I have", "open wifi settings",
               "shut down the computer", "restart my pc", "lock the pc", "search for pizza near me", "open my downloads",
               "What is the capital of France?", "Could you pull up the calculator for me?", "make my screen a bit darker",
               "I need to change my bluetooth settings", "Get Microsoft Edge going for me", "my eyes hurt, the screen is way too bright",
               "I'd like to change my desktop wallpaper", "Tell me a fun fact about octopuses."]
    print("== ROUTER ==")
    for p in phrases: print(f"{p!r:45} -> {actions.route(p)}")
    chat_download(lambda p: None); eng = ChatEngine(print)
    print(f"device={eng.device} load={eng.load_time:.1f}s\n== MODEL (for phrases the router missed) ==", flush=True)
    for p in phrases:
        if actions.route(p): continue
        eng.new_chat(); out = []; t = time.time()
        orig = actions.execute
        actions.execute = lambda a, c=None, tc=None: f"(not executed in test: {a})"
        try: src, a, res, n, dt = eng.handle(p, out.append)
        finally: actions.execute = orig
        print(f"{p!r} -> action={a} text={''.join(out).strip()[:150]!r} [{n} tok, {dt:.1f}s]", flush=True)
    print("== LIVE (harmless) ==")
    v, m = actions.get_volume(); print("volume now", v, "muted", m)
    print(actions.execute({"name": "volume", "args": {"level": v}})); actions.set_volume(mute=m)
    print("volume after", actions.get_volume())
    try:
        b = actions.get_brightness(); print("brightness now", b); print(actions.execute({"name": "brightness", "args": {"level": b}})); print("brightness after", actions.get_brightness())
    except Exception as e: print("brightness error", e)
    print("dark mode now", actions.get_dark())
    print(actions.execute({"name": "open_app", "args": {"app": "notepad"}})); time.sleep(4)
    print("confirm declined ->", actions.execute({"name": "close_app", "args": {"app": "notepad"}}, confirm=lambda m: False))
    print("confirm accepted ->", actions.execute({"name": "close_app", "args": {"app": "notepad"}}, confirm=lambda m: (print("  confirm asked:", m), True)[1]))
    print(actions.execute({"name": "settings", "args": {"page": "display"}})); time.sleep(4)
    print("close settings ->", actions.execute({"name": "close_app", "args": {"app": "SystemSettings"}}, confirm=lambda m: True))
    print(actions.execute({"name": "battery", "args": {}})); print(actions.execute({"name": "time", "args": {}}))
    print(actions.execute({"name": "timer", "args": {"minutes": 0.1, "message": "test"}}, timer_cb=lambda s, m: print("  timer cb", s, m)))
    print("apps indexed:", len(actions.app_index()), "spotify ->", actions.find_app("spotify"), "calculator ->", actions.find_app("calculator"))

def build_chat_tab(root, nb, demo=None):
    import tkinter as tk
    from tkinter import ttk
    f = ttk.Frame(nb); nb.add(f, text="Chat")
    top = ttk.Frame(f); top.pack(fill="x", padx=6, pady=6)
    import queue
    uiq = queue.Queue()
    status = tk.StringVar(master=root, value="Preparing...")
    devlbl = tk.Label(top, text="Device: not loaded", fg="#0068b5", font=("Segoe UI", 10, "bold"))
    devlbl.pack(side="right", padx=6)
    class _D:
        def set(self, t): devlbl.configure(text=t)
    dev = _D()
    def ui(fn, *a): uiq.put((fn, a))
    def drain():
        try:
            while True:
                fn, a = uiq.get_nowait(); fn(*a)
        except queue.Empty: pass
        root.after(40, drain)
    drain()
    st = {"eng": None, "busy": False}
    pb = ttk.Progressbar(f, maximum=100)
    txt = tk.Text(f, wrap="word", font=("Segoe UI", 11), state="disabled", padx=8, pady=6, height=6)
    txt.tag_configure("you", foreground="#0068b5", font=("Segoe UI", 11, "bold"))
    txt.tag_configure("ai", foreground="#107c10", font=("Segoe UI", 11, "bold"))
    txt.pack(fill="both", expand=True, padx=6)
    bot = ttk.Frame(f); bot.pack(fill="x", padx=6, pady=6)
    ent = ttk.Entry(bot, font=("Segoe UI", 11)); ent.pack(side="left", fill="x", expand=True)
    ttk.Label(f, text="Things I can do: open apps or websites, volume, brightness, dark mode, Windows settings, screenshots, timers, battery, time, lock. Closing apps, sleep, restart and shut down ask you first.", foreground="#666", wraplength=1500, justify="left").pack(anchor="w", padx=6, before=bot)
    ttk.Label(f, textvariable=status).pack(anchor="w", padx=6, pady=(0, 4))
    txt.tag_configure("act", foreground="#8a4b00", font=("Segoe UI", 11, "bold"))
    from tkinter import messagebox
    def confirm(msg):
        ev = threading.Event(); r = [False]
        def ask(): r[0] = messagebox.askyesno("NpuStudio", msg, parent=root); ev.set()
        ui(ask); ev.wait(); return r[0]
    def timer_cb(secs, msg):
        def ring():
            root.bell(); top = tk.Toplevel(root); top.title("NpuStudio reminder"); top.attributes("-topmost", True)
            tk.Label(top, text="\u23f0 " + msg, font=("Segoe UI", 16), padx=30, pady=20).pack()
            ttk.Button(top, text="OK", command=top.destroy).pack(pady=(0, 15))
        ui(lambda: root.after(int(secs * 1000), ring))
    def put(s, tag=None):
        txt.configure(state="normal"); txt.insert("end", s, tag); txt.see("end"); txt.configure(state="disabled")
    def send(_=None, q=None):
        q = q or ent.get().strip()
        if not q or not st["eng"] or st["busy"]: return
        ent.delete(0, "end"); st["busy"] = True; import busy; busy.mark("Chat", True)
        put("You: ", "you"); put(q + "\n\n"); put("AI: ", "ai"); status.set("Thinking...")
        def work():
            try:
                src, a, res, n, dt = st["eng"].handle(q, lambda s: ui(put, s), confirm, timer_cb)
                if a is not None: ui(put, "\u2714 " + res, "act")
                if n: ui(status.set, f"{n} tokens in {dt:.1f}s ({n/max(dt,1e-3):.1f} tokens/s) on {getattr(st['eng'], 'last_device', None) or st['eng'].device}" + (f" - action picked by the AI" if a else ""))
                else: ui(status.set, "Done (understood instantly, no AI needed).")
            except Exception as e:
                logging.exception("chat"); ui(status.set, f"Error: {e}")
            ui(put, "\n\n"); st["busy"] = False; import busy; busy.mark("Chat", False)
        threading.Thread(target=work, daemon=True).start()
    def stopgen():
        if st["eng"]: st["eng"].stop = True
    def newchat():
        if st["eng"] and not st["busy"]:
            st["eng"].new_chat(); txt.configure(state="normal"); txt.delete("1.0", "end"); txt.configure(state="disabled"); status.set("New chat started.")
    ttk.Button(bot, text="Send", command=send).pack(side="left", padx=4)
    ttk.Button(bot, text="Stop", command=stopgen).pack(side="left")
    ttk.Button(top, text="New chat", command=newchat).pack(side="left")
    ent.bind("<Return>", send)
    def load():
        try:
            if chat_peer():
                st["eng"] = ChatEngine(lambda m: ui(status.set, m), lazy=True)
                ui(dev.set, "Running on: Laptop NPU"); ui(status.set, "Ready - chat runs on the laptop's NPU. Type a message and press Enter.")
                if demo: ui(lambda: root.after(300, lambda: send(q=demo)))
                return
            if not os.path.exists(os.path.join(CHAT_DIR, "done.txt")):
                ui(lambda: pb.pack(fill="x", padx=6, before=txt))
                def prog(p): ui(lambda: (pb.configure(value=100 * p), status.set(f"Downloading chat model (about 2.2 GB, one time only): {p*100:.0f}%")))
                chat_download(prog); ui(pb.pack_forget)
            ui(dev.set, "Device: loading model...")
            eng = ChatEngine(lambda m: ui(status.set, m))
            st["eng"] = eng
            ui(dev.set, f"Running on: {eng.device}")
            ui(status.set, f"Ready (model loaded in {eng.load_time:.0f}s). Type a message and press Enter.")
            if demo: ui(lambda: root.after(300, lambda: send(q=demo)))
        except Exception as e:
            logging.exception("chat load"); ui(status.set, f"Chat error: {e}"); ui(dev.set, "Device: error")
    threading.Thread(target=load, daemon=True).start()

def _server_pids():
    """Desktop: the standalone NpuStudio-Server processes (venv launcher + its python child)."""
    import subprocess
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | ? { $_.CommandLine -match 'worker_server\\.py' -and $_.CommandLine -notmatch '--share' } | % { $_.ProcessId }"],
                             capture_output=True, text=True, timeout=20, creationflags=0x08000000).stdout
        return [int(x) for x in out.split() if x.isdigit()]
    except Exception: return []

def gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    try: from tkinterdnd2 import TkinterDnD, DND_FILES
    except Exception: TkinterDnD = None
    try:
        import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception: pass
    root = None
    if TkinterDnD:
        try: root = TkinterDnD.Tk()
        except Exception: logging.exception("dnd init"); root = None
    if root is None: root = tk.Tk()
    root.report_callback_exception = lambda et, ev, tb: logging.error("UI callback error", exc_info=(et, ev, tb))
    root.title("NpuStudio")
    import remote; remote.install()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{int(sw*0.6)}x{int(sh*0.65)}+{int(sw*0.2)}+{int(sh*0.15)}")
    topbar = ttk.Frame(root); topbar.pack(fill="x", side="top", padx=6, pady=(4, 0))
    nb = ttk.Notebook(root); nb.pack(fill="both", expand=True)
    args = sys.argv[1:]
    demo = args[args.index("--chat-demo") + 1] if "--chat-demo" in args else None
    if demo: args = [a for a in args if a != demo]
    build_chat_tab(root, nb, demo)
    f = ttk.Frame(nb); nb.add(f, text="Upscale")
    bar = ttk.Frame(f); bar.pack(fill="x", padx=6, pady=6)
    st = {"img": None, "out": None, "up": None, "busy": False, "ups": {}}
    sizev = tk.StringVar(value="4x"); modelv = tk.StringVar(value=list(MODELS)[0]); est = tk.StringVar(value="")
    dn = tk.BooleanVar(value=False)
    dev = tk.StringVar(value="Device: loading model...")
    status = tk.StringVar(value="Open an image (or drag one in) to start.")
    pv = ttk.Frame(f); pv.pack(fill="both", expand=True)
    la = ttk.Label(pv, text="Before", anchor="center"); la.pack(side="left", fill="both", expand=True)
    lb = ttk.Label(pv, text="After", anchor="center"); lb.pack(side="left", fill="both", expand=True)
    from zoomview import ZoomViewer
    zv = ZoomViewer(root)
    def open_zoom(_=None):
        if st["img"] is not None: zv.open(st["img"], st["out"])
    for w_ in (la, lb):
        w_.configure(cursor="hand2"); w_.bind("<Button-1>", open_zoom)
    ttk.Label(f, text="Tip: click a picture to open the zoom compare window.", foreground="#666").pack(anchor="w", padx=6)
    ttk.Label(f, textvariable=est, foreground="#0068b5").pack(anchor="w", padx=6)
    pb = ttk.Progressbar(f, maximum=100); pb.pack(fill="x", padx=6)
    ttk.Label(f, textvariable=status).pack(anchor="w", padx=6, pady=4)
    def show(lbl, im):
        r = min(530 / im.size[0], 520 / im.size[1], 1.0)
        t = im.resize((max(1, int(im.size[0] * r)), max(1, int(im.size[1] * r))), Image.BILINEAR, reducing_gap=3.0); ph = ImageTk.PhotoImage(t); lbl.configure(image=ph, text=""); lbl.image = ph
    def load(p):
        try:
            st["img"] = Image.open(p).convert("RGB"); st["out"] = None; show(la, st["img"])
            lb.configure(image="", text="After"); status.set(f"{os.path.basename(p)}  {st['img'].size[0]}x{st['img'].size[1]}"); upd_est()
        except Exception as e: logging.exception("open"); messagebox.showerror("NpuStudio", str(e))
    def openf():
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if p: load(p)
    def cur_scale(): return int(sizev.get().rstrip("x"))
    def est_text(sc):
        w, h = st["img"].size; mp = w * h * sc * sc / 1e6
        return f"{w*sc}x{h*sc} ({mp:,.1f} MP, ~{need_bytes(w, h, sc)/2**30:.2f} GB of {MAX_BYTES/2**30:.0f} GB limit)"
    def allowed(sc):
        w, h = st["img"].size
        return need_bytes(w, h, sc) <= MAX_BYTES
    def upd_est(_=None):
        if st["img"] is None: est.set(""); return
        sc = cur_scale(); est.set("Output: " + est_text(sc) + ("" if allowed(sc) else "  - too big!"))
    def run():
        if not st["img"] or not st["up"]: return
        if st["busy"]: return
        sc = cur_scale()
        if not allowed(sc):
            ok = [x for x in (16, 8, 4, 2) if x < sc and allowed(x)]
            msg = f"{sc}x would make {est_text(sc)} - over the limit."
            if not ok: messagebox.showwarning("NpuStudio", msg + "\nThis image is too big to upscale further."); return
            if not messagebox.askyesno("NpuStudio", msg + f"\n\nUse {ok[0]}x instead ({est_text(ok[0])})?"): return
            sc = ok[0]; sizev.set(f"{sc}x"); upd_est()
        nb_ = need_bytes(*st["img"].size, sc); fr = avail_ram()
        if nb_ > fr and not messagebox.askyesno("NpuStudio", f"This needs about {nb_/2**30:.1f} GB but only {fr/2**30:.1f} GB of RAM is free right now.\nWindows may page to disk and get slow. Continue anyway?"): return
        if zv.win is not None: zv._close()
        st["out"] = None; lb.configure(image="", text="After"); lb.image = None
        up = st["up"]; st["busy"] = True; import busy; busy.mark("Upscale", True)
        def work():
            try:
                t = time.time()
                def prog(k, n): root.after(0, lambda: pb.configure(value=100 * k / n))
                st["out"] = None
                out = up.upscale_to(st["img"], sc, prog)
                if dn.get(): out = denoise(out)
                st["out"] = out
                root.after(0, lambda: status.set(f"Saving {out.size[0]}x{out.size[1]} to the output folder..."))
                import outputs; outputs.autosave(out, "upscale", f"{sc}x-{up.name.split(' (')[0]}")
                root.after(0, lambda: (show(lb, out), zv.win is not None and zv.open(st["img"], out), status.set(f"Done {sc}x with {up.name.split(' (')[0]} on {up.device} in {time.time()-t:.1f}s -> {out.size[0]}x{out.size[1]}")))
                logging.info("upscale %sx %s %s %.1fs", sc, up.name, out.size, time.time() - t)
            except MemoryError:
                logging.exception("upscale"); root.after(0, lambda: status.set("Out of memory - pick a smaller size."))
            except Exception as e:
                logging.exception("upscale"); root.after(0, lambda: status.set(f"Error: {e}"))
            finally: st["busy"] = False; busy.mark("Upscale", False)
        status.set(f"Upscaling {sc}x..."); threading.Thread(target=work, daemon=True).start()
    def save():
        if not st["out"]: return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not p: return
        out = st["out"]; status.set("Saving...")
        def work():
            try:
                big = out.size[0] * out.size[1] > 50e6
                if p.lower().endswith((".jpg", ".jpeg")):
                    if max(out.size) > 65500: raise ValueError("JPEG max is 65500 px per side - save as PNG.")
                    out.save(p, quality=95, subsampling=0)
                else: out.save(p, compress_level=1 if big else 6)
                root.after(0, status.set, "Saved " + p)
            except Exception as e:
                logging.exception("save"); root.after(0, status.set, f"Save error: {e}")
        threading.Thread(target=work, daemon=True).start()
    ttk.Button(bar, text="Open image", command=openf).pack(side="left")
    ttk.Label(bar, text="Size").pack(side="left", padx=(8, 2))
    cb1 = ttk.Combobox(bar, textvariable=sizev, values=["2x", "4x", "8x", "16x"], width=4, state="readonly"); cb1.pack(side="left")
    cb1.bind("<<ComboboxSelected>>", upd_est)
    ttk.Label(bar, text="Model").pack(side="left", padx=(8, 2))
    cb2 = ttk.Combobox(bar, textvariable=modelv, values=list(MODELS), width=26, state="readonly"); cb2.pack(side="left")
    cb2.bind("<<ComboboxSelected>>", lambda e: switch_model())
    ttk.Button(bar, text="Upscale", command=run).pack(side="left", padx=4)
    ttk.Checkbutton(bar, text="Denoise", variable=dn).pack(side="left", padx=4)
    ttk.Button(bar, text="Save", command=save).pack(side="left", padx=4)
    ttk.Label(bar, textvariable=dev, foreground="#0068b5").pack(side="right")
    def on_drop(ev):
        try:
            files = root.tk.splitlist(ev.data)
            if files: root.after(10, load, files[0])
        except Exception: logging.exception("drop")
        return ev.action
    if TkinterDnD and hasattr(root, "drop_target_register"):
        try:
            for w in (f, pv, la, lb):
                w.drop_target_register(DND_FILES); w.dnd_bind("<<Drop>>", on_drop)
        except Exception: logging.exception("dnd bind")
    def loadmodel(name):
        try:
            if name not in st["ups"]: st["ups"][name] = Upscaler(name)
            u = st["ups"][name]
            def done():
                if modelv.get() == name: st["up"] = u; dev.set(f"Running on: {u.device}")
            root.after(0, done)
        except Exception as e:
            logging.exception("model"); root.after(0, dev.set, f"Model error: {e}")
    def switch_model():
        name = modelv.get(); st["up"] = st["ups"].get(name)
        if st["up"]: dev.set(f"Running on: {st['up'].device}"); return
        dev.set("Device: loading model (first time can take ~1 min)...")
        threading.Thread(target=loadmodel, args=(name,), daemon=True).start()
    threading.Thread(target=loadmodel, args=(modelv.get(),), daemon=True).start()
    mon = {"win": None, "job": None, "cnt": None}
    def open_monitor():
        if mon["win"] is not None and mon["win"].winfo_exists():
            w = mon["win"]; w.deiconify(); w.lift(); w.focus_force(); return
        w = tk.Toplevel(root); w.title("NpuStudio - Chip Monitor"); mon["win"] = w; w.resizable(True, True)
        w.geometry(f"{int(sw*0.42)}x{int(sh*0.7)}+{int(sw*0.29)}+{int(sh*0.12)}")
        COL = {"NPU": ("#a970ff", "#2e1f4a"), "Intel Arc": ("#87cefa", "#1d3a4d"), "NVIDIA": ("#76b900", "#253a00"),
               "Desktop PC": ("#ff9f1a", "#4a2e05"), "Laptop NPU": ("#a970ff", "#2e1f4a"), "Laptop Intel Arc": ("#87cefa", "#1d3a4d"),
               "Laptop NVIDIA": ("#76b900", "#253a00")}
        top = tk.StringVar(master=w, value="Starting..."); ttk.Label(w, textvariable=top, foreground="#555").pack(anchor="w", padx=10, pady=(6, 0))
        if mon["cnt"] is None:
            try:
                import chipmon; mon["cnt"] = chipmon.ChipCounter()
            except Exception as e: logging.exception("counter"); top.set(f"Counter error: {e}")
        cnt = mon["cnt"]; panes = {}
        chips = [c for c in ("NPU", "Intel Arc", "NVIDIA") if cnt and c in cnt.luids]
        lap = remote.role() == "desktop"
        if lap: chips += ["Laptop NPU", "Laptop Intel Arc", "Laptop NVIDIA"]   # the laptop's chips (when NpuStudio is open there)
        elif remote.configured(): chips.append("Desktop PC")
        if not chips: top.set("No NPU/GPU found in performance counters")
        # scrollable column of equal-height panels (fits the window; scrolls when it's made small)
        outer = tk.Frame(w, bg="#111"); outer.pack(fill="both", expand=True)
        scv = tk.Canvas(outer, bg="#111", highlightthickness=0); sb = ttk.Scrollbar(outer, orient="vertical", command=scv.yview)
        scv.configure(yscrollcommand=sb.set); sb.pack(side="right", fill="y"); scv.pack(side="left", fill="both", expand=True)
        body = tk.Frame(scv, bg="#111"); bid = scv.create_window(0, 0, window=body, anchor="nw")
        PH = 110   # min panel height incl. text
        def relayout(e=None):
            W, H = scv.winfo_width(), scv.winfo_height(); n = max(len(panes), 1)
            ph = max(PH, H // n); scv.itemconfigure(bid, width=W, height=ph * n); scv.configure(scrollregion=(0, 0, W, ph * n))
        scv.bind("<Configure>", relayout)
        w.bind("<MouseWheel>", lambda e: scv.yview_scroll(int(-e.delta / 120), "units") if scv.winfo_height() < body.winfo_height() else None)
        for c in chips:
            f = tk.Frame(body, bg="#111"); f.pack(fill="both", expand=True, padx=10, pady=4)
            v = tk.StringVar(master=w, value=c); tk.Label(f, textvariable=v, fg=COL[c][0], bg="#111", anchor="w", justify="left", font=("Segoe UI", 11, "bold")).pack(fill="x")
            cv = tk.Canvas(f, bg="#111", highlightthickness=0, height=60); cv.pack(fill="both", expand=True)
            panes[c] = (v, cv, [0.0] * 120)
        dstate = {"i": None, "busy": False}
        def dq():   # other PC's numbers in the background (never blocks the UI)
            try:
                if lap:
                    dstate["i"] = json.loads(remote._req("api/v1/monitor", timeout=2)[0]).get("chips") if remote.configured() else None
                else: dstate["i"] = remote.health(max_age=0.9)
            except Exception: dstate["i"] = None
            finally: dstate["busy"] = False
        def draw(c, cv, hist):
            W, H = max(cv.winfo_width(), 2), max(cv.winfo_height(), 2); cv.delete("all")
            for i in range(1, 4): cv.create_line(0, H * i / 4, W, H * i / 4, fill="#333")
            pts = []
            for i, x in enumerate(hist): pts += [W * i / (len(hist) - 1), H - H * x / 100]
            cv.create_polygon([0, H] + pts + [W, H], fill=COL[c][1], outline=""); cv.create_line(pts, fill=COL[c][0], width=2)
        if cnt: top.set("Updates every second. Power from real counters only: " + (", ".join(cnt.energy_instances()) or "no Energy Meter counters"))
        def tick():
            if not w.winfo_exists(): return
            try: data = cnt.read() if cnt else {}
            except Exception: logging.exception("tick"); data = {}
            if cnt and data.get("_package") is not None:
                top.set(f"CPU package power: {data['_package']:.1f} W (whole processor package, RAPL)   |   updates every second")
            if ("Desktop PC" in panes or lap) and not dstate["busy"]:
                dstate["busy"] = True; threading.Thread(target=dq, daemon=True).start()
            for c, (v, cv, hist) in panes.items():
                if c.startswith("Laptop "):
                    d = (dstate["i"] or {}).get(c[7:])
                    if not d:
                        hist.append(0.0); del hist[0]; v.set(f"{c}: " + ("offline (open NpuStudio on the laptop)" if dstate["i"] is None else "not reported") + "\n ")
                    else:
                        hist.append(d.get("use") or 0.0); del hist[0]
                        pw = f"{d['watts']:.1f} W ({d['wnote']})" if d.get("watts") is not None else "N/A"
                        tmp = f"   temp {d['temp']}\u00b0C" if d.get("temp") is not None else "   temp N/A"
                        v.set(f"{c}:  {d.get('use', 0):5.1f}%   memory {d.get('ded', 0):,.0f} MB dedicated / {d.get('shared', 0):,.0f} MB shared\npower {pw}{tmp}")
                    draw(c, cv, hist); continue
                if c == "Desktop PC":
                    i = dstate["i"]
                    if not i:
                        hist.append(0.0); del hist[0]; v.set("Desktop PC: offline\n ")
                    else:
                        DEG = "{}\u00b0C"; g = lambda k, f="{}": f.format(i[k]) if i.get(k) not in (None, "N/A") else "N/A"
                        try: u = float(i.get("use") or 0)
                        except Exception: u = 0.0
                        hist.append(u); del hist[0]
                        v.set(f"Desktop PC ({i.get('gpu', 'N/A')}):  {g('use', '{}%')}   memory {g('vram_used', '{:,} MB')} / {g('vram_total', '{:,} MB')}\n"
                              f"power {g('watts', '{} W')}   temp {g('temp', DEG)}   jobs running {i.get('busy', 0)}")
                    draw(c, cv, hist); continue
                d = data.get(c)
                if not d: continue
                hist.append(d["use"]); del hist[0]
                pw = f"{d['watts']:.1f} W ({d['wnote']})" if d["watts"] is not None else "N/A"
                tmp = f"   temp {d['temp']}\u00b0C" if d.get("temp") is not None else "   temp N/A"
                v.set(f"{c}:  {d['use']:5.1f}%   memory {d['ded']:,.0f} MB dedicated / {d['shared']:,.0f} MB shared\npower {pw}{tmp}")
                draw(c, cv, hist)
            mon["job"] = root.after(1000, tick)
        def close():
            if mon["job"]: root.after_cancel(mon["job"]); mon["job"] = None
            w.destroy(); mon["win"] = None
        w.protocol("WM_DELETE_WINDOW", close)
        tick()
    from imagegen import build_gen_tab
    def to_upscale(im):
        st["img"] = im.copy(); st["out"] = None; show(la, st["img"]); lb.configure(image="", text="After"); lb.image = None
        status.set(f"From Image Gen  {im.size[0]}x{im.size[1]}"); upd_est(); nb.select(1)
    up_lock = threading.Lock()
    def get_up(name):
        with up_lock:
            if name not in st["ups"]: st["ups"][name] = Upscaler(name)
            return st["ups"][name]
    from editor import build_edit_tab
    edit_box = {}
    def to_edit(im):
        if edit_box.get("api"): edit_box["api"]["load"](im.copy()); nb.select(edit_box["api"]["tab"])
    gtab, gen_go, gen_prompt, gen_size = build_gen_tab(root, nb, to_upscale, get_up, list(MODELS), to_edit)
    def edit_upscale(im, f): return get_up(list(MODELS)[1]).upscale_to(im, 2 if f <= 2 else 4)
    try: edit_box["api"] = build_edit_tab(root, nb, to_upscale, DND_FILES if TkinterDnD else None, upscale=edit_upscale)
    except Exception: logging.exception("edit tab")
    ttk.Button(bar, text="Send to Edit", command=lambda: (st["out"] or st["img"]) is not None and to_edit(st["out"] or st["img"])).pack(side="left", padx=4)
    ttk.Button(topbar, text="Chip Monitor", command=open_monitor).pack(side="right")
    def do_restart(force=False, server_too=False):
        """Restart button: confirm if a job is running, stop the in-app share server, start a fresh copy, exit."""
        import busy, subprocess, urllib.request
        jobs = busy.running()
        try:
            k = remote.local_key()
            if k and remote.local_status() == "running":
                r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:47860/api/v1/info", headers={"X-Key": k}), timeout=2)
                if json.loads(r.read()).get("busy"): jobs.append("A server job (from the phone or the other PC)")
        except Exception: pass
        if jobs and not force and not messagebox.askyesno("Restart NpuStudio", f"{', '.join(jobs)} is still running - restart anyway?", parent=root): return
        logging.info("Restart button (jobs running: %s)", jobs)
        remote.stop_share()
        if server_too and remote.role() != "laptop":
            for pid in _server_pids(): subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=0x08000000)
        for _ in range(20):            # let the share server free port 47860
            if remote.local_status() in ("stopped", "error") or remote.role() != "laptop": break
            time.sleep(0.25)
        py = os.path.join(APP, "venv", "Scripts", "pythonw.exe")
        if not os.path.exists(py): py = sys.executable
        args = [a for a in sys.argv[1:] if a != "--autorestart"]
        subprocess.Popen([py, os.path.join(APP, "npustudio.py")] + args, cwd=APP, close_fds=True,
                         creationflags=0x00000008 | 0x00000200)     # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        if server_too and remote.role() != "laptop": remote.start_desktop_server()
        try: root.destroy()
        finally: os._exit(0)
    rstm = tk.Menubutton(topbar, text="\u21bb Restart", relief="raised", padx=8)
    rm_ = tk.Menu(rstm, tearoff=0); rstm["menu"] = rm_
    rm_.add_command(label="Restart NpuStudio", command=do_restart)
    if remote.role() != "laptop": rm_.add_command(label="Restart NpuStudio + NpuStudio-Server", command=lambda: do_restart(server_too=True))
    rstm.pack(side="right", padx=4)
    import outputs
    def pick_outputs():
        from tkinter import filedialog
        d = filedialog.askdirectory(parent=root, title="Folder for full-size results (phone jobs)", initialdir=outputs.folder())
        if d: outputs.set_folder(d); messagebox.showinfo("Output folder", f"Full-size results will be saved in:\n{outputs.folder()}", parent=root)
    outm = tk.Menubutton(topbar, text="\U0001F4C1 Output", relief="raised", padx=8)
    om_ = tk.Menu(outm, tearoff=0); outm["menu"] = om_
    om_.add_command(label="Open output folder", command=outputs.open_folder)
    om_.add_command(label="Change output folder...", command=pick_outputs)
    om_.add_command(label="Reset to default (NpuStudio\\outputs)", command=lambda: outputs.set_folder(""))
    outm.pack(side="right", padx=4)
    if "--autorestart" in sys.argv: root.after(6000, lambda: do_restart(force=True))
    ttk.Button(topbar, text="Pair phone", command=lambda: open_pair_window(root)).pack(side="right", padx=4)
    import devices
    devv = tk.StringVar(value=devices.get_raw())
    deskv = tk.StringVar(value="")
    def pair():
        from tkinter import simpledialog
        if remote.role() == "desktop":   # the laptop is found automatically; this is only for a manual address
            h = simpledialog.askstring("Laptop", "NpuStudio finds the laptop by itself once NpuStudio is open there (with 'Share this laptop's chips' on).\n"
                                       "Or type the laptop's address (192.168.x.x or Tailscale 100.x.x.x):", initialvalue=remote.cfg().get("host", ""), parent=root)
            if not h: return False
            h = h.strip().split(":")[0]; json.dump({"host": h, "port": 47860}, open(remote.LAPTOP, "w")); remote._h["t"] = 0; return True
        c = remote.cfg()
        h = simpledialog.askstring("Desktop PC", "Desktop PC address (shown in the NpuStudio-Server window):", initialvalue=c.get("host", ""), parent=root)
        if not h: return False
        k = simpledialog.askstring("Desktop PC", "Pairing key (shown in the NpuStudio-Server window):", initialvalue=c.get("key", ""), parent=root)
        if not k: return False
        remote.save(h, k); return True
    def desk_poll():
        def w():
            i = remote.health(force=True) if remote.configured() else None
            P = remote.peer()
            txt = (f"{P}: online ({i.get('name', '')} {i.get('gpu', '?')})" if P == "Laptop" else f"{P}: online ({i.get('gpu', '?')})") if i else \
                  (f"{P}: offline" if remote.configured() or P == "Laptop" else "")
            root.after(0, deskv.set, txt)
            root.after(0, srv_paint)
        threading.Thread(target=w, daemon=True).start(); root.after(5000, desk_poll)
    def dev_changed(_=None):
        if devv.get() in devices.REMOTE:
            if not remote.configured() and not pair(): devv.set(devices.get_raw()); return
            if not remote.health(force=True):
                status.set("Desktop PC is offline - start NpuStudio-Server.bat on the desktop. Jobs will run on this laptop until it is online."
                           if remote.peer() == "Desktop PC" else "Laptop is offline - open NpuStudio on the laptop (Server button green there). Jobs run on this PC until then.")
        devices.set(devv.get())
        with up_lock: st["ups"].clear()
        st["up"] = None; root.after(50, switch_model)
        b = devices.bench(); tip = ", ".join(f"{k}: " + "/".join(f"{d} {v:.2f}s" for d, v in r.items() if isinstance(v, (int, float))) for k, r in list(b.items())[:2])
        status.set(f"Device: {devv.get()} - AI features will use it from the next job." + (f"  (benchmarks: {tip})" if tip else ""))
    dcb = ttk.Combobox(topbar, textvariable=devv, values=devices.choices(), width=14, state="readonly"); dcb.pack(side="right", padx=4)
    dcb.bind("<<ComboboxSelected>>", dev_changed); ttk.Label(topbar, text="Device").pack(side="right")
    ttk.Label(topbar, textvariable=deskv, foreground="#0068b5").pack(side="right", padx=8)
    ttk.Button(topbar, text="Pair desktop..." if remote.role() == "laptop" else "Laptop address...", command=lambda: (pair(), remote.health(force=True))).pack(side="left")
    # Server button: this PC's worker server. Red = not running, green = running, amber = starting / error / port in use.
    srvst = {"s": "stopped"}
    SRVCOL = {"running": "#1e9e3a", "stopped": "#d32f2f", "starting": "#f0a000", "error": "#f0a000", "conflict": "#f0a000"}
    srvb = tk.Menubutton(topbar, text="Server", relief="raised", fg="white", bg=SRVCOL["stopped"], activebackground=SRVCOL["stopped"], padx=8)
    srvm = tk.Menu(srvb, tearoff=0); srvb["menu"] = srvm; srvb.pack(side="left", padx=6)
    sharev = tk.BooleanVar(value=remote.sharing())
    def share_toggle():
        remote.set_sharing(sharev.get())
        if sharev.get(): remote.start_share()
        else: remote.stop_share()
        root.after(1500, srv_check)
    def srv_menu():
        srvm.delete(0, "end"); s_ = srvst["s"]
        lab = {"running": "Running (LAN + Tailscale, pairing key required)", "stopped": "Not running", "starting": "Starting...",
               "error": "Error: " + (remote.local_error() or "see error.log")[:80], "conflict": "Port 47860 is in use by another program"}[s_]
        srvm.add_command(label=lab, state="disabled")
        if remote.role() == "laptop":
            srvm.add_checkbutton(label="Share this laptop's chips", variable=sharev, command=share_toggle)
            if s_ != "running" and sharev.get(): srvm.add_command(label="Start now", command=lambda: (remote.start_share(), root.after(1500, srv_check)))
        elif s_ != "running": srvm.add_command(label="Start NpuStudio-Server", command=lambda: (remote.start_desktop_server(), root.after(3000, srv_check)))
    srvm.configure(postcommand=srv_menu)
    def srv_paint():
        c = SRVCOL[srvst["s"]]; srvb.configure(bg=c, activebackground=c,
            text={"running": "Server: on", "stopped": "Server: off", "starting": "Server: starting", "error": "Server: error", "conflict": "Server: port busy"}[srvst["s"]])
    def srv_check():
        def w(): srvst["s"] = remote.local_status(); root.after(0, srv_paint)
        threading.Thread(target=w, daemon=True).start()
    def srv_loop(): srv_check(); root.after(4000, srv_loop)
    if remote.role() == "laptop" and remote.configured() and remote.sharing(): srvst["s"] = "starting"; remote.start_share()
    root.after(200, srv_loop)
    root.after(1000, desk_poll)
    if "--monitor" in args: root.after(300, open_monitor)
    if "--size" in args: sizev.set(args[args.index("--size") + 1] + "x"); args = [a for a in args if a != args[args.index("--size") + 1]]
    if "--sharp" in args: modelv.set(list(MODELS)[1]); switch_model()
    if "--uptab" in args: nb.select(1)
    if "--gentab" in args: nb.select(2)
    skipv = {args[args.index(k) + 1] for k in ("--chgtext", "--gensize") if k in args and args.index(k) + 1 < len(args)}
    imgs = [a for a in args if not a.startswith("--") and not re.fullmatch(r"\d+x\d+", a) and a not in skipv]
    if imgs: root.after(500, load, imgs[0])
    if "--gentest" in args:
        if "--gensize" in args: gen_size(args[args.index("--gensize") + 1])
        nb.select(2); root.after(1500, gen_go)
    if "--edittab" in args:
        nb.select(3)
        if imgs: root.after(1500, lambda: to_edit(Image.open(imgs[0]).convert("RGB")))
        if "--chgwin" in args:
            def cw():
                w = edit_box["api"]["change"]()
                if w and "--chgtext" in args: w["ins"].insert("1.0", args[args.index("--chgtext") + 1]); root.after(500, w["find"])
            root.after(3000, cw)
    if "--zoomtest" in args and imgs:
        nb.select(1)
        def zt(stage=0):
            if stage == 0:
                if st["up"] is None or st["img"] is None: return root.after(500, zt, 0)
                run(); return root.after(500, zt, 1)
            if st["out"] is None: return root.after(500, zt, 1)
            open_zoom(); root.after(1500, lambda: zv.set_zoom(4.0))
        root.after(1000, zt)
    root.mainloop()

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--actiontest": action_test()
        elif len(sys.argv) > 1 and sys.argv[1] == "--chattest": chat_test(sys.argv[2] if len(sys.argv) > 2 else "In two sentences, what is an NPU?")
        elif len(sys.argv) > 1 and sys.argv[1] == "--uptest":
            # --uptest in.png fast|sharp scale out.png
            nm = list(MODELS)[0 if sys.argv[3] == "fast" else 1]; t = time.time(); u = Upscaler(nm)
            print(nm, "on", u.device, "load %.1fs" % (time.time() - t), flush=True)
            im = Image.open(sys.argv[2]).convert("RGB"); t = time.time()
            o = u.upscale_to(im, int(sys.argv[4])); dt = time.time() - t
            print(im.size, "->", o.size, "%.2fs" % dt, "%.0f in-px/s" % (im.size[0] * im.size[1] / dt), flush=True)
            t = time.time(); o.save(sys.argv[5], compress_level=1); print("saved %.1fs" % (time.time() - t), os.path.getsize(sys.argv[5]) >> 20, "MB")
            import ctypes
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("pf", ctypes.c_ulong)] + [(n, ctypes.c_size_t) for n in ("pws", "ws", "qpp", "qp", "qnpp", "qnp", "pu", "ppu")]
            m = PMC(); m.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32"); k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
            print("pmi ok", k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(m), m.cb))
            print("peak RAM %.2f GB (working set), peak commit %.2f GB" % (m.pws / 2**30, m.ppu / 2**30), flush=True)
        elif len(sys.argv) > 1 and sys.argv[1] == "--test": headless_test(sys.argv[2], sys.argv[3])
        else: gui()
    except Exception:
        logging.error(traceback.format_exc()); raise
