import os, time, logging, threading
import numpy as np
from PIL import Image
APP = os.path.dirname(os.path.abspath(__file__))
REPO = "OpenVINO/LCM_Dreamshaper_v7-int8-ov"
GDIR = os.path.join(APP, "models", "lcm-dreamshaper-v7-int8")
CACHE = os.path.join(APP, "models", "cache")
COMBOS = [("GPU", "NPU", "GPU"), ("CPU", "NPU", "CPU"), ("GPU", "GPU", "GPU"), ("CPU", "CPU", "CPU")]

def downloaded():
    return os.path.exists(os.path.join(GDIR, "unet", "openvino_model.xml")) and os.path.exists(os.path.join(GDIR, ".done"))

def download(progress=None):
    from huggingface_hub import snapshot_download
    stop = [False]
    def mon():
        while not stop[0]:
            sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(GDIR) for f in fs) if os.path.isdir(GDIR) else 0
            if progress: progress(min(sz / 1.05e9, 0.99))
            time.sleep(1)
    threading.Thread(target=mon, daemon=True).start()
    try: snapshot_download(REPO, local_dir=GDIR)
    finally: stop[0] = True
    open(os.path.join(GDIR, ".done"), "w").write("ok")

class ImageGen:
    def __init__(self, w=512, h=512, status=None):
        import openvino_genai as og
        self.og = og; self.size = None; self.devs = None; self.err = []
        self.pipe = og.Text2ImagePipeline(GDIR)
        self.build(w, h, status)

    def build(self, w, h, status=None):
        import devices
        gs = 8.0; self.pref = devices.get(); self.cuda = False
        import remote
        if remote.active():   # Desktop PC / Both machines: cudaeng.gen is routed to the desktop by remote.install()
            self.cuda = True; self.devs = (remote.peer(),) * 3; self.size = (w, h); return
        labs = devices.plan("gen")
        if not labs: raise RuntimeError(devices.unsupported("Image Gen"))
        if labs[0] == "NVIDIA":
            try:
                import cudaeng
                if cudaeng.available(): self.cuda = True; self.devs = ("NVIDIA",) * 3; self.size = (w, h); return
            except Exception: logging.exception("cuda gen")
        self.pipe.reshape(1, h, w, gs)
        combos = []
        if self.pref == "All" and "NPU" in labs and len(labs) > 1:
            g = devices.ov([l for l in labs if l != "NPU"][0]); combos.append((g, "NPU", g))
        if self.pref in ("Auto",) and "NPU" in labs:
            combos += [c for c in COMBOS[:1]]
        combos += [(devices.ov(l),) * 3 if l != "NPU" else ("CPU", "NPU", "CPU") for l in labs]
        for c in combos:
            try:
                if status: status(f"Compiling image model (text {c[0]}, UNet {c[1]}, VAE {c[2]})...")
                t = time.time()
                self.pipe.compile(c[0], c[1], c[2], CACHE_DIR=CACHE)
                self.devs = c; self.size = (w, h); mark_shape(w, h)
                logging.info("ImageGen %s on %s in %.1fs", (w, h), c, time.time() - t); return
            except Exception as e:
                self.err.append(f"{c}: {e}"); logging.warning("imagegen compile %s failed: %s", c, e)
        raise RuntimeError("No device combo worked: " + " | ".join(self.err)[:600])

    def generate(self, prompt, neg="", steps=4, seed=42, w=512, h=512, progress=None, cancel=None):
        import devices
        import remote
        if (w, h) != self.size or self.pref != devices.get() or (self.devs and self.devs[0] in ("Desktop PC", "Laptop")) != remote.active(): self.build(w, h)
        if self.cuda:
            import cudaeng; return cudaeng.gen(prompt, w, h, int(steps), int(seed), progress=progress)
        def cb(step, n, latent):
            if progress: progress(step + 1, n)
            return bool(cancel and cancel())
        kw = dict(num_inference_steps=int(steps), rng_seed=int(seed), callback=cb)
        if neg.strip(): kw["negative_prompt"] = neg
        try: t = self.pipe.generate(prompt, **kw)
        except Exception:
            if "negative_prompt" not in kw: raise
            kw.pop("negative_prompt"); t = self.pipe.generate(prompt, **kw)  # LCM may not use negatives
        return Image.fromarray(np.array(t.data)[0])

PRESETS = ["512x512", "512x768", "768x512", "800x600", "1024x768", "1152x864", "1280x600", "1280x720", "1280x768",
           "1280x800", "1280x960", "1280x1024", "1360x768", "1366x768", "1400x1050", "1440x900", "1600x900", "1600x1200",
           "1680x1050", "1792x1344", "1856x1392", "1920x1080", "1920x1200", "1920x1440", "2048x1152", "2048x1536",
           "2560x1440", "2560x1600", "2880x1800 (screen)", "1080x1920", "1440x2560", "Custom..."]
# base draw shapes (multiples of 64, long side 768) - one compiled UNet per shape, cached on disk
BUCKETS = [(512, 512), (768, 576), (768, 512), (768, 448), (768, 384)]
BUCKETS += [(h, w) for w, h in BUCKETS if w != h]
SHAPES_SEEN = os.path.join(CACHE, "gen_shapes.txt")

def plan_for(tw, th):
    import math
    if (tw, th) in BUCKETS: return (tw, th), 1
    bw, bh = min(BUCKETS, key=lambda b: abs(math.log((b[0] / b[1]) / (tw / th))))
    need = max(tw / bw, th / bh)
    up = 1 if need <= 1.0 else 2 if need <= 2 else 4 if need <= 4 else 8 if need <= 8 else 16
    return (bw, bh), up

def fit_crop(im, tw, th):
    r = max(tw / im.size[0], th / im.size[1])
    nw, nh = max(tw, round(im.size[0] * r)), max(th, round(im.size[1] * r))
    if (nw, nh) != im.size: im = im.resize((nw, nh), Image.LANCZOS)
    l, t = (nw - tw) // 2, (nh - th) // 2
    return im.crop((l, t, l + tw, t + th)) if (nw, nh) != (tw, th) else im

def shape_seen(w, h):
    try: return f"{w}x{h}" in open(SHAPES_SEEN).read().split()
    except Exception: return False

def mark_shape(w, h):
    os.makedirs(CACHE, exist_ok=True)
    if not shape_seen(w, h): open(SHAPES_SEEN, "a").write(f"{w}x{h}\n")

def build_gen_tab(root, nb, send_to_upscale=None, get_upscaler=None, up_models=None, send_to_edit=None):
    import tkinter as tk, random, queue
    from tkinter import ttk, filedialog, messagebox
    from PIL import ImageTk
    logging.getLogger("httpx").setLevel(logging.WARNING); logging.getLogger("httpx2").setLevel(logging.WARNING)
    f = ttk.Frame(nb); nb.add(f, text="Image Gen")
    st = {"gen": None, "img": None, "busy": False, "loading": False, "cancel": False}
    uiq = queue.Queue()
    def ui(fn, *a): uiq.put((fn, a))
    def drain():
        try:
            while True: fn, a = uiq.get_nowait(); fn(*a)
        except queue.Empty: pass
        root.after(50, drain)
    drain()
    top = ttk.Frame(f); top.pack(fill="x", padx=6, pady=(6, 2))
    ttk.Label(top, text="Prompt").grid(row=0, column=0, sticky="w")
    pr = tk.Text(top, height=3, wrap="word", font=("Segoe UI", 10)); pr.grid(row=0, column=1, columnspan=8, sticky="we", padx=4)
    pr.insert("1.0", "a cozy cabin in a snowy forest at sunset, highly detailed, photo")
    ttk.Label(top, text="Negative").grid(row=1, column=0, sticky="w", pady=3)
    ng = ttk.Entry(top); ng.grid(row=1, column=1, columnspan=8, sticky="we", padx=4, pady=3)
    top.columnconfigure(1, weight=1)
    row = ttk.Frame(f); row.pack(fill="x", padx=6, pady=2)
    stepsv = tk.IntVar(value=4); seedv = tk.StringVar(value="42"); sizev = tk.StringVar(value="512x512"); upv = tk.StringVar(value=(up_models or ["-"])[-1])
    ttk.Label(row, text="Steps").pack(side="left")
    ttk.Spinbox(row, from_=1, to=12, textvariable=stepsv, width=4).pack(side="left", padx=(2, 10))
    ttk.Label(row, text="Seed").pack(side="left")
    ttk.Entry(row, textvariable=seedv, width=10).pack(side="left", padx=2)
    ttk.Button(row, text="Random", width=7, command=lambda: seedv.set(str(random.randint(0, 2**31 - 1)))).pack(side="left", padx=(0, 10))
    ttk.Label(row, text="Size").pack(side="left")
    szcb = ttk.Combobox(row, textvariable=sizev, values=PRESETS, width=17, state="readonly"); szcb.pack(side="left", padx=(2, 6))
    last = {"v": "512x512"}
    def on_size(_=None):
        if sizev.get() == "Custom...":
            from tkinter import simpledialog
            v = simpledialog.askstring("NpuStudio", "Custom size as WIDTHxHEIGHT (e.g. 3000x2000):", parent=root)
            try:
                w, h = [int(x) for x in v.lower().replace("×", "x").split("x")]
                if not (64 <= w <= 16384 and 64 <= h <= 16384): raise ValueError
                sizev.set(f"{w}x{h}")
            except Exception: sizev.set(last["v"])
        last["v"] = sizev.get(); upd_plan()
    szcb.bind("<<ComboboxSelected>>", on_size)
    if up_models:
        ttk.Label(row, text="Upscale with").pack(side="left")
        ttk.Combobox(row, textvariable=upv, values=[m.split(" (")[0] for m in up_models], width=6, state="readonly").pack(side="left", padx=(2, 10))
        upv.set("Sharp")
    fixv = tk.BooleanVar(value=True)
    ttk.Checkbutton(row, text="Auto-fix faces", variable=fixv).pack(side="left", padx=(2, 10))
    gb = ttk.Button(row, text="Generate"); gb.pack(side="left", padx=4)
    sv = ttk.Button(row, text="Save", state="disabled"); sv.pack(side="left", padx=4)
    su = ttk.Button(row, text="Send to Upscale", state="disabled"); su.pack(side="left", padx=4)
    se = ttk.Button(row, text="Send to Edit", state="disabled"); se.pack(side="left", padx=4)
    dev = tk.StringVar(value="Model not loaded yet")
    ttk.Label(f, textvariable=dev, foreground="#0068b5").pack(anchor="e", padx=6)
    pic = ttk.Label(f, text="Your picture will appear here.", anchor="center"); pic.pack(fill="both", expand=True)
    pb = ttk.Progressbar(f, maximum=100); pb.pack(fill="x", padx=6)
    status = tk.StringVar(value="Type a prompt and press Generate. Runs fully offline once the model is downloaded (about 1 GB, one time).")
    ttk.Label(f, textvariable=status).pack(anchor="w", padx=6, pady=4)
    def target():
        return tuple(int(x) for x in sizev.get().split(" ")[0].split("x"))
    def plan_text():
        tw, th = target(); (bw, bh), up = plan_for(tw, th)
        t = f"Draw {bw}x{bh}" + (f" \u2192 upscale {up}x ({upv.get()})" if up > 1 else "") + (f" \u2192 {tw}x{th}" if (bw, bh) != (tw, th) else "")
        if not shape_seen(bw, bh): t += "   (first use of this shape: compiling takes a few minutes, then it is cached)"
        return t
    def upd_plan(_=None):
        if not st["busy"]: status.set(plan_text())
    def devtxt(g): return f"Text: {g.devs[0]}  |  UNet: {g.devs[1]}  |  VAE: {g.devs[2]}"
    def ensure():
        if st["gen"] is None:
            if not downloaded():
                download(lambda p: ui(lambda: (pb.configure(value=100 * p), status.set(f"Downloading image model (about 1 GB, one time only): {p*100:.0f}%"))))
            ui(status.set, "Compiling image model (first time takes ~3 min, then it is cached)...")
            w, h = plan_for(*target())[0]
            st["gen"] = ImageGen(w, h, status=lambda m: ui(status.set, m))
            ui(dev.set, devtxt(st["gen"]))
        return st["gen"]
    def preload(_=None):
        if nb.select() != str(f) or st["gen"] or st["loading"] or not downloaded(): return
        st["loading"] = True
        def w():
            try: ensure(); ui(status.set, "Ready.")
            except Exception as e: logging.exception("imagegen load"); ui(status.set, f"Model error: {e}")
            finally: st["loading"] = False
        threading.Thread(target=w, daemon=True).start()
    nb.bind("<<NotebookTabChanged>>", preload, add="+")
    def show(im):
        r = min(max(pic.winfo_width(), 300) / im.size[0], max(pic.winfo_height(), 300) / im.size[1], 1.5)
        ph = ImageTk.PhotoImage(im.resize((int(im.size[0] * r), int(im.size[1] * r)), Image.LANCZOS)); pic.configure(image=ph, text=""); pic.image = ph
    def generate():
        if st["busy"]: st["cancel"] = True; return
        prompt = pr.get("1.0", "end").strip()
        if not prompt: status.set("Type a prompt first."); return
        try: seed = int(seedv.get())
        except ValueError: seed = random.randint(0, 2**31 - 1); seedv.set(str(seed))
        tw, th = target(); (w, h), up = plan_for(tw, th); upname = next((m for m in (up_models or []) if m.startswith(upv.get())), None); steps = max(1, min(12, int(stepsv.get() or 4))); neg = ng.get()
        st["busy"] = True; __import__("busy").mark("Image Gen", True); st["cancel"] = False; gb.configure(text="Stop"); pb.configure(value=0)
        def work():
            try:
                while st["loading"]: time.sleep(0.3)
                g = ensure()
                pt = plan_text(); dw = 40 if up > 1 else 100
                if (w, h) != g.size:
                    ui(status.set, pt + "  -  " + ("compiling new shape, a few minutes..." if not shape_seen(w, h) else "switching shape..."))
                    g.build(w, h)
                mark_shape(w, h); pt = plan_text()
                t = time.time()
                im = g.generate(prompt, neg, steps, seed, w, h,
                                progress=lambda k, n: ui(lambda: (pb.configure(value=dw * k / n), status.set(f"{pt}  -  drawing step {k}/{n}"))),
                                cancel=lambda: st["cancel"])
                t_draw = time.time() - t
                if st["cancel"]: ui(status.set, "Stopped."); return
                upd = "-"
                if up > 1 and upname and get_upscaler:
                    ui(status.set, f"{pt}  -  loading upscaler..."); u = get_upscaler(upname); upd = u.device
                    ui(status.set, f"{pt}  -  upscaling on {u.device}...")
                    im = u.upscale_to(im, up, lambda k, n: ui(lambda: pb.configure(value=dw + (100 - dw) * k / n)))
                elif up > 1:
                    im = im.resize((w * up, h * up), Image.LANCZOS)
                im = fit_crop(im, tw, th)
                fxn = ""
                if fixv.get() and not st["cancel"]:
                    try:
                        import facefix, cudaeng
                        if cudaeng.available():
                            ui(status.set, f"{pt}  -  auto-fixing faces/eyes...")
                            im2, fxn, ok = facefix.fix(im, "eyes", prompt, status=lambda m: ui(status.set, f"{pt}  -  {m}"), min_gain=0.0)
                            if ok: im = im2
                            fxn = "; " + ("faces fixed" if ok else "face fix: " + fxn.replace("NOT DONE - ", "")[:80])
                    except Exception: logging.exception("gen auto face fix"); fxn = "; face fix failed"
                import json as _js; im.info["npustudio_gen"] = _js.dumps({"prompt": prompt, "negative": neg, "seed": seed, "steps": steps, "width": tw, "height": th, "model": "imagegen"})
                import outputs; outputs.autosave(im, "gen", prompt)
                dt = time.time() - t; t_up = dt - t_draw
                ow, oh = tw, th
                st["img"] = im; logging.info("imagegen %s steps=%s seed=%s %.2fs %s", (ow, oh), steps, seed, dt, g.devs)
                ui(lambda: (show(im), dev.set(devtxt(g)), sv.configure(state="normal"), su.configure(state="normal" if send_to_upscale else "disabled"), se.configure(state="normal" if send_to_edit else "disabled"),
                            pb.configure(value=100), status.set(f"Done {ow}x{oh} in {dt:.1f}s (draw {t_draw:.1f}s, UNet on {g.devs[1]}" + (f"; upscale {t_up:.1f}s on {upd}" if up > 1 else "") + f"; seed {seed}{fxn})")))
                logging.info("imagegen final %sx%s draw %.1fs up %.1fs (%s on %s)", w, h, t_draw, t_up, upname, upd)
            except Exception as e:
                logging.exception("imagegen"); ui(status.set, f"Error: {e}")
            finally:
                st["busy"] = False; __import__("busy").mark("Image Gen", False); ui(lambda: gb.configure(text="Generate"))
        threading.Thread(target=work, daemon=True).start()
    def save():
        if st["img"] is None: return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if p: st["img"].save(p, **({"quality": 95} if p.lower().endswith((".jpg", ".jpeg")) else {})); status.set("Saved " + p)
    gb.configure(command=generate); sv.configure(command=save)
    su.configure(command=lambda: st["img"] is not None and send_to_upscale(st["img"]))
    se.configure(command=lambda: st["img"] is not None and send_to_edit and send_to_edit(st["img"]))
    pr.bind("<Control-Return>", lambda e: (generate(), "break")[1])
    def set_size(v): sizev.set(v); last["v"] = v; upd_plan()
    return f, generate, pr, set_size

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if not downloaded():
        t = time.time(); download(lambda p: None); print("download %.0fs" % (time.time() - t), flush=True)
    t = time.time(); g = ImageGen(); print("devices", g.devs, "load %.1fs" % (time.time() - t), flush=True)
    for i in range(2):
        t = time.time()
        im = g.generate(sys.argv[1] if len(sys.argv) > 1 else "a cozy cabin in a snowy forest at sunset, highly detailed, photo", steps=4, seed=42 + i)
        print("gen %.2fs" % (time.time() - t), im.size, flush=True)
        im.save(os.path.join(APP, f"gen_sample{i}.png"))
