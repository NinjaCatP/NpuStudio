"""Edit tab: crop/rotate/flip/resize, adjustments + filters, AI background removal (IS-Net),
AI erase (LaMa) and AI prompt inpaint (SD 1.5 inpainting, OpenVINO GenAI)."""
import re
import os, time, logging, threading, queue, urllib.request
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageTk
APP = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(APP, "models"); CACHE = os.path.join(MD, "cache")
URLS = {
    "isnet": ("https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx", "isnet-general-use.onnx", 178e6),
    "lama": ("https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx", "lama_fp32.onnx", 208e6),
}
SD_DIR = os.path.join(MD, "sd-inpaint-int8")
DEVS = ["NPU", "GPU", "CPU"]

def fetch(key, progress=None):
    url, fn, size = URLS[key]; p = os.path.join(MD, fn)
    if os.path.exists(p) and os.path.getsize(p) > size * 0.9: return p
    os.makedirs(MD, exist_ok=True); tmp = p + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        tot = int(r.headers.get("Content-Length") or size); got = 0; last = -1
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b); got += len(b)
            if progress and int(got * 20 / tot) != last: last = int(got * 20 / tot); progress(got / tot)
    os.replace(tmp, p); return p

class OVModel:
    def __init__(self, key, shapes, status=None, devs=DEVS, outputs=None, cfg=None):
        import openvino as ov
        c = ov.Core(); c.set_property({"CACHE_DIR": CACHE})
        p = fetch(key, (lambda f: status(f"Downloading {key} model: {f*100:.0f}%")) if status else None)
        m = c.read_model(p)
        if outputs: m = ov.Model([m.outputs[i] for i in outputs], m.get_parameters(), key)
        m.reshape(shapes); self.err = []
        import devices
        if not devs: raise RuntimeError(devices.unsupported(key))
        for d in devs:
            od = devices.ov(d) if d in ("NPU", "Intel Arc", "NVIDIA", "CPU") else d
            if not any(x.startswith(od) for x in c.available_devices): continue
            try:
                if status: status(f"Compiling {key} on {d} (first time can take a minute)...")
                t = time.time(); self.cm = c.compile_model(m, od, (cfg or {}).get(d, (cfg or {}).get("*", {}) if od != "NPU" else {})); self.device = d
                logging.info("edit model %s on %s (%.1fs)", key, d, time.time() - t); return
            except Exception as e:
                self.err.append(f"{d}: {str(e)[:200]}"); logging.warning("edit %s compile on %s failed: %s", key, d, e)
        raise RuntimeError(f"{key}: no device worked: " + " | ".join(self.err))

# ---------------- pure image ops ----------------
FILTERS = ["None", "B&W", "Sepia", "Blur", "Sharpen", "Warm", "Cool"]
def adjust(im, b=1.0, c=1.0, s=1.0, sh=1.0, flt="None"):
    a = im.getchannel("A") if im.mode == "RGBA" else None
    x = im.convert("RGB")
    if b != 1: x = ImageEnhance.Brightness(x).enhance(b)
    if c != 1: x = ImageEnhance.Contrast(x).enhance(c)
    if s != 1: x = ImageEnhance.Color(x).enhance(s)
    if sh != 1: x = ImageEnhance.Sharpness(x).enhance(sh)
    if flt == "B&W": x = ImageOps.grayscale(x).convert("RGB")
    elif flt == "Sepia": x = ImageOps.colorize(ImageOps.grayscale(x), "#2e1d0e", "#fff3dc", mid="#a8875b")
    elif flt == "Blur": x = x.filter(ImageFilter.GaussianBlur(max(1, min(x.size) / 300)))
    elif flt == "Sharpen": x = x.filter(ImageFilter.UnsharpMask(2, 120, 2))
    elif flt in ("Warm", "Cool"):
        r, g, bb = x.split(); k = 1.08 if flt == "Warm" else 0.92
        r = r.point(lambda v: min(255, int(v * k))); bb = bb.point(lambda v: min(255, int(v / k))); x = Image.merge("RGB", (r, g, bb))
    if a is not None: x = x.convert("RGBA"); x.putalpha(a)
    return x

class AI:
    def __init__(self): self.m = {}; self.lock = threading.Lock(); self.pref = None
    def _check(self):
        import devices
        p = devices.get()
        if p != self.pref: self.m = {}; self.pref = p
        return devices
    def remove_bg(self, im, status=None):
        import remote
        if remote.active(): r, d = remote.call("remove_bg", im, status=status); return r, remote.peer() + " " + str(d)
        with self.lock:
            dv = self._check()
            if "isnet" not in self.m: self.m["isnet"] = OVModel("isnet", [1, 3, 1024, 1024], status, devs=dv.plan("isnet"), outputs=[0])
            om = self.m["isnet"]
            x = np.asarray(im.convert("RGB").resize((1024, 1024), Image.BILINEAR), np.float32) / 255.0
            x = (x - 0.5) / 1.0
            r = om.cm(x.transpose(2, 0, 1)[None])[om.cm.outputs[0]][0, 0]
            r = (r - r.min()) / max(1e-6, r.max() - r.min())
            a = Image.fromarray((r * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)
            out = im.convert("RGBA"); out.putalpha(a); return out, om.device

    def _region(self, im, mask, pad=0.6):
        bb = mask.getbbox()
        if not bb: raise ValueError("Paint over the area first (brush).")
        x0, y0, x1, y1 = bb; s = max(x1 - x0, y1 - y0); s = int(s * (1 + 2 * pad)) + 32
        s = max(s, 256); s = min(s, max(im.size))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        l = min(max(0, cx - s // 2), max(0, im.size[0] - s)); t = min(max(0, cy - s // 2), max(0, im.size[1] - s))
        return (l, t, min(im.size[0], l + s), min(im.size[1], t + s))

    def _paste(self, im, patch, mask, box):
        m = mask.crop(box).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(3))
        out = im.copy(); base = out.convert("RGB").crop(box); res = Image.composite(patch.resize(base.size, Image.LANCZOS), base, m)
        if out.mode == "RGBA": res = res.convert("RGBA"); res.putalpha(out.crop(box).getchannel("A"))
        out.paste(res, box[:2]); return out

    def erase(self, im, mask, status=None):
        import remote
        if remote.active(): r, d = remote.call("erase", im, mask, status=status); return r, remote.peer() + " " + str(d)
        with self.lock:
            dv = self._check()
            if "lama" not in self.m: self.m["lama"] = OVModel("lama", {"image": [1, 3, 512, 512], "mask": [1, 1, 512, 512]}, status, devs=dv.plan("lama"),
                                           cfg={"*": {"INFERENCE_PRECISION_HINT": "f32"}})  # NPU compiler hangs on LaMa FFT ops; GPU fp16 overflows
            om = self.m["lama"]; box = self._region(im, mask, 0.5)
            # dilate the mask so LaMa also replaces the object's edge/shadow halo
            dil = max(9, (int(0.012 * max(im.size)) // 2) * 2 + 1)
            mfull = mask.filter(ImageFilter.MaxFilter(dil))
            rgb = np.asarray(im.convert("RGB").crop(box)); mm = np.asarray(mfull.crop(box))
            h, w = mm.shape; S = max(h, w)
            # pad to a square (edge-reflect) so LaMa never sees a squashed picture
            rgb = np.pad(rgb, ((0, S - h), (0, S - w), (0, 0)), mode="reflect" if (S - h) < h and (S - w) < w else "edge")
            mm = np.pad(mm, ((0, S - h), (0, S - w)))
            x = np.asarray(Image.fromarray(rgb).resize((512, 512), Image.LANCZOS), np.float32).transpose(2, 0, 1)[None] / 255.0
            mk = (np.asarray(Image.fromarray(mm).resize((512, 512), Image.NEAREST)) > 127).astype(np.float32)
            r = om.cm({"image": x, "mask": mk[None, None]})[om.cm.outputs[0]][0].transpose(1, 2, 0)
            patch = Image.fromarray(np.clip(r, 0, 255).astype(np.uint8)).resize((S, S), Image.LANCZOS).crop((0, 0, w, h))
            return self._paste(im, patch, mfull, box), om.device

    def change(self, im, mask, prompt, steps=30, strength=1.0, seed=1, progress=None, status=None, guidance=7.5, negative=None, grow=9):
        import openvino_genai as og, openvino as ov, remote
        if remote.active():
            r, d = remote.call("change", im, mask, prompt, steps=steps, strength=strength, seed=seed, guidance=guidance, negative=negative, grow=grow, progress=progress, status=status)
            return r, ("Desktop PC",) * 3
        with self.lock:
            dv = self._check(); labs = dv.plan("inpaint")
            if not labs: raise RuntimeError(dv.unsupported("AI change (inpaint)"))
            box = self._region(im, mask, 0.4)
            a = np.asarray(im.convert("RGB").crop(box)); h, w = a.shape[:2]; S = max(h, w)
            a = np.pad(a, ((0, S - h), (0, S - w), (0, 0)), mode="reflect" if (S - h) < h and (S - w) < w else "edge")
            mm = np.pad(np.asarray(mask.crop(box)), ((0, S - h), (0, S - w)))
            rg = Image.fromarray(a).resize((512, 512), Image.LANCZOS)
            mk = Image.fromarray(mm).resize((512, 512), Image.NEAREST).filter(ImageFilter.MaxFilter(grow)).convert("RGB")
            if labs[0] == "NVIDIA":
                import cudaeng
                out = cudaeng.inpaint(rg, mk, prompt, steps=steps, strength=strength, guidance=guidance, seed=seed, progress=progress, status=status, negative=negative)
                devs = ("NVIDIA",) * 3
            else:
                # NPU UNet compiles, but openvino-genai's NPU UNet wrapper fails at run time for the 9-channel inpaint UNet
                # (output tensor shape mismatch), so the inpaint UNet runs on a GPU or the CPU.
                if "sd" not in self.m:
                    if not os.path.exists(os.path.join(SD_DIR, "unet", "openvino_model.xml")):
                        raise RuntimeError("SD inpainting model missing in models\\sd-inpaint-int8")
                    p = og.InpaintingPipeline(SD_DIR); p.reshape(1, 512, 512, guidance); err = []
                    for L in labs:
                        o = dv.ov(L)
                        try:
                            if status: status(f"Compiling inpaint model on {L} - first time takes a few minutes...")
                            t = time.time(); p.compile(o, o, o, CACHE_DIR=CACHE); self.m["sd"] = (p, (L,) * 3)
                            logging.info("inpaint on %s (%.1fs)", L, time.time() - t); break
                        except Exception as e: err.append(f"{L}: {str(e)[:200]}"); logging.warning("inpaint compile %s: %s", L, e)
                    if "sd" not in self.m: raise RuntimeError("inpaint: no device worked: " + " | ".join(err))
                p, devs = self.m["sd"]
                def cb(step, n, lat):
                    if progress: progress(step + 1, n)
                    return False
                r = p.generate(prompt, image=ov.Tensor(np.asarray(rg)[None].copy()), mask_image=ov.Tensor(np.asarray(mk)[None].copy()),
                               negative_prompt=negative or "blurry, deformed, low quality",
                               num_inference_steps=int(steps), strength=float(strength), rng_seed=int(seed), callback=cb)
                out = Image.fromarray(np.array(r.data)[0])
            patch = out.resize((S, S), Image.LANCZOS).crop((0, 0, w, h))
            return self._paste(im, patch, mask, box), devs

ASPECTS = {"Free": None, "1:1": 1.0, "4:3": 4 / 3, "3:2": 1.5, "16:10": 1.6, "16:9": 16 / 9, "2880x1800 (screen)": 2880 / 1800, "9:16": 9 / 16}

def build_edit_tab(root, nb, send_to_upscale=None, dnd=None, upscale=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
    f = ttk.Frame(nb); nb.add(f, text="Edit")
    ai = AI(); uiq = queue.Queue()
    st = {"img": None, "hist": [], "redo": [], "mask": None, "mode": "crop", "crop": None, "busy": False, "view": None, "prev": None, "drag": None}
    def ui(fn, *a): uiq.put((fn, a))
    def drain():
        try:
            while True: fn, a = uiq.get_nowait(); fn(*a)
        except queue.Empty: pass
        root.after(50, drain)
    drain()
    # --- toolbars ---
    r1 = ttk.Frame(f); r1.pack(fill="x", padx=6, pady=(6, 2))
    r2 = ttk.Frame(f); r2.pack(fill="x", padx=6, pady=2)
    r3 = ttk.Frame(f); r3.pack(fill="x", padx=6, pady=2)
    body = ttk.Frame(f); body.pack(fill="both", expand=True, padx=6)
    cv = tk.Canvas(body, bg="#2b2b2b", highlightthickness=0); cv.pack(fill="both", expand=True)
    pb = ttk.Progressbar(f, maximum=100); pb.pack(fill="x", padx=6)
    status = tk.StringVar(value="Open an image, drag one in, or use 'Send to Edit' from another tab.")
    ttk.Label(f, textvariable=status).pack(anchor="w", padx=6, pady=3)
    def B(p, t, cmd, **kw): kw.setdefault("width", max(4, len(t) + 1)); b = ttk.Button(p, text=t, command=cmd, **kw); b.pack(side="left", padx=2); return b
    # adjustments vars
    bv, cvv, sv, shv = (tk.DoubleVar(value=1.0) for _ in range(4)); fv = tk.StringVar(value="None")
    aspv = tk.StringVar(value="Free"); brush = tk.IntVar(value=30); modev = tk.StringVar(value="crop")
    prompt = tk.StringVar(value="a vase of flowers")

    def cur(): return st["img"]
    def adj_on(): return (bv.get(), cvv.get(), sv.get(), shv.get(), fv.get()) != (1.0, 1.0, 1.0, 1.0, "None")
    def push(new, msg=None):
        if st["img"] is not None: st["hist"].append(st["img"]); st["hist"][:] = st["hist"][-20:]
        st["img"] = new; st["redo"].clear(); reset_adj(redraw=False); clear_mask(False); st["crop"] = None; redraw()
        if msg: status.set(msg + f"  ({new.size[0]}x{new.size[1]})")
    def commit_adj():
        if st["img"] is not None and adj_on():
            new = adjust(st["img"], bv.get(), cvv.get(), sv.get(), shv.get(), fv.get()); push(new, "Applied adjustments")
    def reset_adj(redraw=True):
        for v in (bv, cvv, sv, shv): v.set(1.0)
        fv.set("None")
        if redraw: globals_redraw()
    def undo():
        if st["hist"]: st["redo"].append(st["img"]); st["img"] = st["hist"].pop(); clear_mask(False); reset_adj(False); redraw(); status.set("Undo")
    def redo():
        if st["redo"]: st["hist"].append(st["img"]); st["img"] = st["redo"].pop(); clear_mask(False); reset_adj(False); redraw(); status.set("Redo")
    def load(src):
        try:
            im = src if isinstance(src, Image.Image) else Image.open(src)
            im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") and "A" in im.convert("RGBA").getbands() and im.mode != "RGB" else im.convert("RGB")
            st["hist"].clear(); st["redo"].clear(); st["img"] = None; push(im, "Loaded" if isinstance(src, Image.Image) else f"Loaded {os.path.basename(src)}")
        except Exception as e: logging.exception("edit load"); messagebox.showerror("NpuStudio", str(e))
    def openf():
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if p: load(p)
    def final():
        commit_adj(); return st["img"]
    def save():
        im = final()
        if im is None: return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG (keeps transparency)", "*.png"), ("JPEG", "*.jpg")])
        import outputs; outputs.autosave(im, "edit", "final")
        if not p: return
        try:
            if p.lower().endswith((".jpg", ".jpeg")): im.convert("RGB").save(p, quality=95)
            else: im.save(p)
            status.set("Saved " + p)
        except Exception as e: logging.exception("edit save"); status.set(f"Save error: {e}")
    # --- view / mapping ---
    def view_geom():
        im = st["img"]; W, H = max(cv.winfo_width(), 50), max(cv.winfo_height(), 50)
        s = min(W / im.size[0], H / im.size[1], 4.0); w, h = int(im.size[0] * s), int(im.size[1] * s)
        return s, (W - w) // 2, (H - h) // 2, w, h
    def to_img(x, y):
        s, ox, oy, w, h = st["view"]; return (min(max((x - ox) / s, 0), st["img"].size[0]), min(max((y - oy) / s, 0), st["img"].size[1]))
    def globals_redraw(*_): redraw()
    def redraw(*_):
        cv.delete("all"); im = st["img"]
        if im is None: cv.create_text(cv.winfo_width() // 2, cv.winfo_height() // 2, text="No image", fill="#999"); return
        s, ox, oy, w, h = st["view"] = view_geom()
        p = im.resize((max(1, w), max(1, h)), Image.BILINEAR, reducing_gap=2.0)
        if adj_on(): p = adjust(p, bv.get(), cvv.get(), sv.get(), shv.get(), fv.get())
        if p.mode == "RGBA":
            chk = Image.new("RGB", p.size, "#cccccc"); d = ImageDraw.Draw(chk)
            for yy in range(0, p.size[1], 16):
                for xx in range((yy // 16 % 2) * 16, p.size[0], 32): d.rectangle([xx, yy, xx + 15, yy + 15], fill="#ffffff")
            chk.paste(p, (0, 0), p); p = chk
        if st["mask"] is not None and st["mask"].getbbox():
            mk = st["mask"].resize(p.size, Image.NEAREST); red = Image.new("RGB", p.size, (255, 40, 40))
            p = Image.composite(Image.blend(p.convert("RGB"), red, 0.5), p.convert("RGB"), mk)
        ph = ImageTk.PhotoImage(p); st["prev"] = ph; cv.create_image(ox, oy, anchor="nw", image=ph)
        if st["crop"]:
            x0, y0, x1, y1 = st["crop"]
            cv.create_rectangle(ox + x0 * s, oy + y0 * s, ox + x1 * s, oy + y1 * s, outline="#ffcc00", width=2, dash=(6, 3))
    cv.bind("<Configure>", lambda e: redraw())
    def clear_mask(redraw_=True):
        st["mask"] = None
        if redraw_: redraw()
    def press(e):
        if st["img"] is None or st["view"] is None: return
        x, y = to_img(e.x, e.y); st["drag"] = (x, y)
        if modev.get() == "brush": paint(e)
    def move(e):
        if st["drag"] is None or st["img"] is None: return
        if modev.get() == "brush": paint(e); return
        x0, y0 = st["drag"]; x1, y1 = to_img(e.x, e.y); a = ASPECTS[aspv.get()]
        if a:
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w / max(h, 1e-6) > a: w = h * a
            else: h = w / a
            x1 = x0 + (w if x1 >= x0 else -w); y1 = y0 + (h if y1 >= y0 else -h)
        st["crop"] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)); redraw()
    def release(e): st["drag"] = None
    def paint(e):
        im = st["img"]
        if st["mask"] is None or st["mask"].size != im.size: st["mask"] = Image.new("L", im.size, 0)
        x, y = to_img(e.x, e.y); r = brush.get() / st["view"][0]
        ImageDraw.Draw(st["mask"]).ellipse([x - r, y - r, x + r, y + r], fill=255)
        if not st.get("pend"): st["pend"] = True; root.after(30, lambda: (st.update(pend=False), redraw()))
    cv.bind("<ButtonPress-1>", press); cv.bind("<B1-Motion>", move); cv.bind("<ButtonRelease-1>", release)
    # --- ops ---
    def need():
        if st["img"] is None: status.set("Open an image first."); return False
        if st["busy"]: status.set("Busy - wait for the current AI job."); return False
        return True
    def op(fn, msg):
        if need(): commit_adj(); push(fn(st["img"]), msg)
    def do_crop():
        if not need(): return
        if not st["crop"]: status.set("Drag a rectangle on the picture first (Crop mode)."); return
        x0, y0, x1, y1 = [int(round(v)) for v in st["crop"]]
        if x1 - x0 < 2 or y1 - y0 < 2: return
        commit_adj(); push(st["img"].crop((x0, y0, x1, y1)), "Cropped")
    def do_resize():
        if not need(): return
        w, h = st["img"].size
        v = simpledialog.askstring("Resize", f"New size as WIDTHxHEIGHT, or just a width to keep the aspect (now {w}x{h}):", parent=root)
        if not v: return
        try:
            if "x" in v.lower(): nw, nh = [int(t) for t in v.lower().split("x")]
            else: nw = int(v); nh = round(h * nw / w)
            if not (1 <= nw <= 30000 and 1 <= nh <= 30000): raise ValueError
        except ValueError: status.set("Size not understood."); return
        op(lambda im: im.resize((nw, nh), Image.LANCZOS), "Resized")
    def run_ai(name, fn):
        if not need(): return
        commit_adj(); st["busy"] = True; __import__("busy").mark("Edit", True); pb.configure(mode="indeterminate"); pb.start(12); status.set(name + "...")
        im = st["img"]; mask = st["mask"]
        def work():
            t = time.time()
            try:
                res, dev = fn(im, mask)
                import outputs; outputs.autosave(res, "edit", name)
                dt = time.time() - t; logging.info("edit %s %.1fs on %s", name, dt, dev)
                ui(lambda: (push(res, f"{name} done in {dt:.1f}s on {dev if isinstance(dev, str) else 'UNet ' + dev[1]}"),))
            except Exception as e:
                logging.exception("edit " + name); ui(status.set, f"{name} error: {e}")
            finally:
                st["busy"] = False; __import__("busy").mark("Edit", False); ui(lambda: (pb.stop(), pb.configure(mode="determinate", value=100)))
        threading.Thread(target=work, daemon=True).start()
    def sstat(m): ui(status.set, m)
    def rm_bg(): run_ai("Remove background", lambda im, m: ai.remove_bg(im, sstat))
    def erase(): run_ai("AI erase", lambda im, m: ai.erase(im, m if m is not None else Image.new("L", im.size), sstat))
    # ---------------- AI Change pop-out window ----------------
    eng = {"chg": None, "seg": None, "ana": {}}
    def changer():
        if eng["chg"] is None:
            import aichange; eng["chg"] = aichange.Changer()
        return eng["chg"]
    def segger():
        if eng["seg"] is None:
            import aichange; eng["seg"] = aichange.Seg()
        return eng["seg"]
    def change():
        import aichange
        if st["img"] is None: status.set("Open an image first."); return
        if st.get("cw") is not None and st["cw"].winfo_exists(): st["cw"].lift(); return
        w = tk.Toplevel(root); w.title("AI change"); sw, sh = w.winfo_screenwidth(), w.winfo_screenheight(); w.geometry(f"{min(int(sw*0.9), 1800)}x{min(int(sh*0.85), 1000)}+20+20"); st["cw"] = w; w.transient(root)
        cs = {"auto": None, "auto_for": None, "pmax": 0, "img": None, "job": 0, "ph": None}
        rt = ttk.Frame(w); rt.pack(side="right", fill="y", padx=6, pady=6)
        left = ttk.Frame(w); left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        pv = tk.Canvas(left, bg="#2b2b2b", highlightthickness=0, width=520, height=420); pv.pack(fill="both", expand=True)
        legend = ttk.Label(left, text="Red = brush (paint on the main Edit picture)   Cyan = auto-mask   No mask = whole image", foreground="#666")
        legend.pack(anchor="w", pady=(4, 0))
        ttk.Label(rt, text="Analysis (what the AI sees)").pack(anchor="w")
        ana = tk.Text(rt, height=6, width=56, wrap="word"); ana.pack(fill="x")
        ar = ttk.Frame(rt); ar.pack(fill="x", pady=2); astat = tk.StringVar(value="")
        ttk.Button(ar, text="Re-analyze", command=lambda: analyze(True)).pack(side="left"); ttk.Label(ar, textvariable=astat, foreground="#555").pack(side="left", padx=6)
        ttk.Label(rt, text="What do you want to change?").pack(anchor="w", pady=(6, 0))
        ins = tk.Text(rt, height=3, width=56, wrap="word"); ins.pack(fill="x")
        ex = ttk.Label(rt, text='e.g. "change outfit to a red dress", "replace the rocket with a plant", "remove the helmet", "make it nighttime"', foreground="#666", wraplength=420)
        ex.pack(anchor="w")
        mr = ttk.Frame(rt); mr.pack(fill="x", pady=(6, 2))
        ttk.Button(mr, text="Find target", command=lambda: find()).pack(side="left")
        ttk.Label(mr, text="Grow/Shrink").pack(side="left", padx=(10, 2)); growv = tk.IntVar(value=8)
        ttk.Scale(mr, variable=growv, from_=-30, to=40, length=120, command=lambda v: draw()).pack(side="left")
        glbl = ttk.Label(mr, text="+8 px", width=7); glbl.pack(side="left")
        ttk.Button(mr, text="Clear auto-mask", command=lambda: (cs.update(auto=None, auto_for=None), draw())).pack(side="left", padx=4)
        mstat = tk.StringVar(value=""); ttk.Label(rt, textvariable=mstat, foreground="#0a5", wraplength=440).pack(anchor="w")
        sr = ttk.Frame(rt); sr.pack(fill="x", pady=(6, 2)); strv = tk.DoubleVar(value=1.0); strl = ttk.Label(sr, text="", width=24)
        ttk.Label(sr, text="Strength").pack(side="left"); ttk.Scale(sr, variable=strv, from_=0.2, to=1.0, length=180,
                                                                command=lambda v: strl.configure(text=f"{float(v):.2f}")).pack(side="left", padx=4); strl.pack(side="left")
        mr2 = ttk.Frame(rt); mr2.pack(fill="x", pady=(4, 0)); methv = tk.StringVar(value="Auto")
        ttk.Label(mr2, text="Method").pack(side="left")
        ttk.Combobox(mr2, textvariable=methv, values=["Auto", "Redraw", "Recolor"], state="readonly", width=9).pack(side="left", padx=4)
        ttk.Label(mr2, text="Redraw = draw a new thing (inpaint)   Recolor = same thing, new colour (pix2pix)", foreground="#666").pack(side="left")
        ttk.Label(rt, text="Prompt used").pack(anchor="w", pady=(6, 0)); pused = tk.StringVar(value="")
        ttk.Entry(rt, textvariable=pused, width=60, state="readonly").pack(fill="x")
        wpb = ttk.Progressbar(rt, maximum=100); wpb.pack(fill="x", pady=(8, 2)); wst = tk.StringVar(value="")
        ttk.Label(rt, textvariable=wst, wraplength=440).pack(anchor="w")
        br = ttk.Frame(rt); br.pack(fill="x", pady=6)
        apb = ttk.Button(br, text="Apply", command=lambda: apply()); apb.pack(side="left", padx=2)
        ttk.Button(br, text="Undo", command=lambda: (undo(), cs.update(auto=None, auto_for=None), draw())).pack(side="left", padx=2)
        ttk.Button(br, text="Cancel", command=lambda: close()).pack(side="left", padx=2)
        def close():
            cs["job"] += 1; st["cw"] = None; w.destroy()
        w.protocol("WM_DELETE_WINDOW", close)
        def wset(m): ui(wst.set, m)
        def cur_mode():
            instr = ins.get("1.0", "end").strip(); tgt, rep, kind = aichange.parse(instr) if instr else (None, "", "global")
            brush_ = st["mask"] is not None and st["mask"].getbbox() is not None
            if brush_: return "brush", tgt, rep, kind, instr
            if tgt: return "auto", tgt, rep, kind, instr
            return "whole", tgt, rep, kind, instr
        def auto_mask():
            if cs["auto"] is None: return None
            return aichange.grow(cs["auto"], int(growv.get()))
        def draw():
            if not w.winfo_exists() or st["img"] is None: return
            glbl.configure(text=f"{int(growv.get()):+d} px")
            im = st["img"]; W_, H_ = max(pv.winfo_width(), 100), max(pv.winfo_height(), 100)
            s_ = min(W_ / im.size[0], H_ / im.size[1]); tw, th = max(1, int(im.size[0] * s_)), max(1, int(im.size[1] * s_))
            p = im.convert("RGB").resize((tw, th), Image.BILINEAR)
            am = auto_mask()
            if am is not None and am.size == im.size:
                p = Image.composite(Image.blend(p, Image.new("RGB", p.size, (0, 220, 255)), 0.5), p, am.resize(p.size, Image.NEAREST))
            if st["mask"] is not None and st["mask"].getbbox():
                p = Image.composite(Image.blend(p, Image.new("RGB", p.size, (255, 40, 40)), 0.5), p, st["mask"].resize(p.size, Image.NEAREST))
            cs["ph"] = ImageTk.PhotoImage(p); pv.delete("all"); pv.create_image((W_ - tw) // 2, (H_ - th) // 2, anchor="nw", image=cs["ph"])
            mode = cur_mode()[0]
            mstat.set({"brush": "Area: your brush mask (only the red area changes).",
                       "auto": "Area: auto-mask of the target" + (f" (confidence {cs['pmax']:.2f})" if cs["auto"] is not None else " - press Find target or Apply to detect it."),
                       "whole": ("Area: whole image." if aichange.is_global(cur_mode()[4]) else "Area: none yet - name a thing to change, brush it, or say 'whole image'.")}[mode])
            if cs.get("onmode"): cs["onmode"]()
        def tick():
            if not w.winfo_exists(): return
            key = (id(st["img"]), st["mask"].getbbox() if st["mask"] is not None else None)
            if key != cs.get("key"): cs["key"] = key; draw()
            w.after(400, tick)
        pv.bind("<Configure>", lambda e: draw())
        def analyze(force=False):
            im = st["img"]; k = id(im)
            if not force and k in eng["ana"]: ana.delete("1.0", "end"); ana.insert("1.0", eng["ana"][k]); astat.set("(cached)"); return
            astat.set("Analyzing..."); ana.delete("1.0", "end")
            def job():
                try:
                    txt, dev, dt = changer().analyze(im, lambda m: ui(astat.set, m))
                    eng["ana"][k] = txt
                    def put():
                        if w.winfo_exists(): ana.delete("1.0", "end"); ana.insert("1.0", txt); astat.set(f"{dev}, {dt:.1f}s")
                    ui(put)
                except Exception as e:
                    logging.exception("analyze"); ui(astat.set, f"Analysis failed: {e}")
            threading.Thread(target=job, daemon=True).start()
        def find(then=None):
            mode, tgt, rep, kind, instr = cur_mode()
            if not tgt: wst.set("No target in the instruction - it will change the whole image."); return
            im = st["img"]; analysis = ana.get("1.0", "end").strip(); wst.set(f"Finding '{tgt}'...")
            def job():
                try:
                    t = time.time(); r = None; how = ""
                    import facefix
                    m, pm, how = facefix.find_mask(im, tgt, instr, analysis)
                    dt = time.time() - t; logging.info("find '%s' -> %s conf %.2f %.2fs", tgt, how, pm, dt)
                    def done():
                        cs.update(auto=m, auto_for=(id(im), instr), pmax=pm); draw()
                        if pm < 0.2 or not m.getbbox():
                            cs.update(auto=None); draw(); wst.set(f"Couldn't find '{tgt}' (confidence {pm:.2f}): brush it or click Find target. Nothing was changed."); return
                        else:
                            mm = methv.get() if methv.get() != "Auto" else aichange.method_for(aichange.parse(instr)[1])
                            wst.set(f"Found '{tgt}' with {how} in {dt:.2f}s. Method: {'redraw (inpaint)' if mm == 'Redraw' else 'recolor (pix2pix)'}. Adjust Grow/Shrink or brush, then Apply.")
                        if then: then()
                    ui(done)
                except Exception as e:
                    logging.exception("seg"); ui(wst.set, f"Find failed: {e}")
            threading.Thread(target=job, daemon=True).start()
        def apply():
            if st["busy"]: wst.set("Busy - wait for the current AI job."); return
            mode, tgt, rep, kind, instr = cur_mode()
            if not instr: wst.set("Type what you want to change."); return
            try:
                import facefix; _fix = facefix.fix_what(instr)
            except Exception: _fix = None
            if _fix: mode = "whole"   # "fix eyes"/"fix face": the detailer finds the faces itself
            elif mode == "auto" and cs.get("auto_for") != (id(st["img"]), instr): find(then=apply); return
            im = st["img"]; analysis = ana.get("1.0", "end").strip()
            mask = st["mask"] if mode == "brush" else (auto_mask() if mode == "auto" else None)
            if mode == "auto" and (mask is None or not mask.getbbox()): wst.set("Target not found - paint it with the brush."); return
            if mode == "whole" and not _fix and not aichange.is_global(instr):
                wst.set("Couldn't tell what to change. Name the thing (e.g. 'change outfit to a red dress'), brush it, "
                        "or say 'whole image' for a whole-picture change. Nothing was changed."); return
            strength = float(strv.get()); job = cs["job"] = cs["job"] + 1
            commit_adj(); st["busy"] = True; __import__("busy").mark("AI Change", True); apb.state(["disabled"]); wpb.configure(mode="indeterminate"); wpb.start(12)
            def prog(k, n): ui(lambda: (wpb.stop(), wpb.configure(mode="determinate", value=100 * k / n), wst.set(f"Step {k}/{n}")))
            def work():
                t = time.time()
                try:
                    c = changer()
                    if mask is not None:
                        res, dev, how, p = c.edit_masked(ai, im, mask, instr, analysis, strength=max(0.5, strength), progress=prog, status=wset, upscaler=upscale, method=methv.get())
                    else:
                        res, dev, how, p = c.edit_whole(im, instr, analysis, strength=min(strength, 0.6), progress=prog, status=wset, upscaler=upscale)
                    import outputs; outputs.autosave(res, "aichange", instr)
                    dt = time.time() - t; dv = dev if isinstance(dev, str) else "/".join(dict.fromkeys(dev))
                    logging.info("AI change '%s' mode=%s how=%s dev=%s %.1fs prompt=%s", instr, mode, how, dv, dt, p)
                    def done():
                        if job != cs["job"]: return
                        push(res, f"AI change done in {dt:.1f}s ({how} on {dv})")
                        if w.winfo_exists(): pused.set(p); wst.set(f"Done in {dt:.1f}s - {how} on {dv}. Undo to revert."); cs.update(auto=None, auto_for=None); draw()
                        if w.winfo_exists() and res is not im: eng["ana"].clear(); analyze(force=True)  # the picture changed: describe the new one
                    ui(done)
                except Exception as e:
                    logging.exception("AI change"); ui(wst.set, f"Error: {e}")
                finally:
                    st["busy"] = False; __import__("busy").mark("AI Change", False)
                    ui(lambda: (wpb.stop(), wpb.configure(mode="determinate", value=100), w.winfo_exists() and apb.state(["!disabled"])))
            threading.Thread(target=work, daemon=True).start()
        def set_str():
            m = cur_mode()[0]
            if m != cs.get("lastmode"): cs["lastmode"] = m; strv.set(1.0 if m != "whole" else 0.4); strl.configure(text=f"{strv.get():.2f}")
        def on_mode(*_):
            draw(); mode, tgt, rep, kind, instr = cur_mode()
            if mode == "auto" and cs.get("auto_for") != (id(st["img"]), instr) and not st["busy"]:
                if cs.get("fjob"): w.after_cancel(cs["fjob"])
                cs["fjob"] = w.after(700, lambda: (cs.update(fjob=None), cur_mode()[4] == instr and find()))
        cs["onmode"] = set_str; ins.bind("<KeyRelease>", lambda e: w.after(300, on_mode)); on_mode()
        analyze(); tick(); ins.focus_set()
        return {"win": w, "ins": ins, "apply": apply, "find": find, "ana": ana}
    # row 1: file + transform
    B(r1, "Open", openf); B(r1, "Save", save); B(r1, "Undo", undo); B(r1, "Redo", redo)
    ttk.Separator(r1, orient="vertical").pack(side="left", fill="y", padx=6)
    B(r1, "\u27f2 90", lambda: op(lambda im: im.transpose(Image.ROTATE_90), "Rotated left"))
    B(r1, "\u27f3 90", lambda: op(lambda im: im.transpose(Image.ROTATE_270), "Rotated right"))
    B(r1, "180", lambda: op(lambda im: im.transpose(Image.ROTATE_180), "Rotated 180"))
    B(r1, "Flip H", lambda: op(lambda im: im.transpose(Image.FLIP_LEFT_RIGHT), "Flipped"))
    B(r1, "Flip V", lambda: op(lambda im: im.transpose(Image.FLIP_TOP_BOTTOM), "Flipped"))
    B(r1, "Resize...", do_resize)
    ttk.Separator(r1, orient="vertical").pack(side="left", fill="y", padx=6)
    ttk.Radiobutton(r1, text="Crop", variable=modev, value="crop").pack(side="left")
    ttk.Combobox(r1, textvariable=aspv, values=list(ASPECTS), width=16, state="readonly").pack(side="left", padx=2)
    B(r1, "Apply crop", do_crop)
    if send_to_upscale: B(r1, "Send to Upscale", lambda: st["img"] is not None and send_to_upscale(final()))
    # row 2: adjustments
    def sl(lbl, var, lo, hi):
        ttk.Label(r2, text=lbl).pack(side="left")
        ttk.Scale(r2, variable=var, from_=lo, to=hi, length=110, command=lambda v: sched()).pack(side="left", padx=(2, 8))
    def sched():
        if not st.get("adjp"): st["adjp"] = True; root.after(60, lambda: (st.update(adjp=False), redraw()))
    sl("Brightness", bv, 0.2, 2.0); sl("Contrast", cvv, 0.2, 2.0); sl("Saturation", sv, 0.0, 2.5); sl("Sharpness", shv, 0.0, 3.0)
    ttk.Label(r2, text="Filter").pack(side="left")
    fcb = ttk.Combobox(r2, textvariable=fv, values=FILTERS, width=8, state="readonly"); fcb.pack(side="left", padx=2)
    fcb.bind("<<ComboboxSelected>>", lambda e: redraw())
    B(r2, "Apply", commit_adj); B(r2, "Reset", reset_adj)
    # row 3: AI
    B(r3, "AI remove background", rm_bg)
    ttk.Separator(r3, orient="vertical").pack(side="left", fill="y", padx=6)
    ttk.Radiobutton(r3, text="Brush", variable=modev, value="brush").pack(side="left")
    ttk.Scale(r3, variable=brush, from_=5, to=120, length=80).pack(side="left", padx=2)
    B(r3, "Clear mask", clear_mask); B(r3, "AI erase", erase)
    B(r3, "AI change...", change)
    root.bind_all("<Control-z>", lambda e: nb.select() == str(f) and undo())
    root.bind_all("<Control-y>", lambda e: nb.select() == str(f) and redo())
    if dnd:
        try:
            for w in (f, cv):
                w.drop_target_register(dnd)
                w.dnd_bind("<<Drop>>", lambda ev: (root.after(10, load, root.tk.splitlist(ev.data)[0]), ev.action)[1])
        except Exception: logging.exception("edit dnd")
    api = {"load": load, "tab": f, "rm_bg": rm_bg, "erase": erase, "change": change, "st": st, "modev": modev, "prompt": prompt, "fv": fv, "sv": sv, "redraw": redraw}
    return api

if __name__ == "__main__":
    # CLI test: python editor.py in.png outprefix  (mask = centered ellipse)
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    im = Image.open(sys.argv[1]).convert("RGB"); pre = sys.argv[2]; ai = AI()
    adjust(im, 1.1, 1.2, 1.3, 1.5, "Warm").save(pre + "adjust.png")
    adjust(im, flt="Sepia").save(pre + "sepia.png")
    t = time.time(); r, d = ai.remove_bg(im, print); print("removebg", d, "%.2fs" % (time.time() - t)); r.save(pre + "removebg.png")
    t = time.time(); r, d = ai.remove_bg(im); print("removebg warm", d, "%.2fs" % (time.time() - t))
    W, H = im.size; mk = Image.new("L", im.size, 0)
    box = [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.4, 0.4, 0.6, 0.6]
    ImageDraw.Draw(mk).ellipse([W * box[0], H * box[1], W * box[2], H * box[3]], fill=255)
    vis = im.copy(); vis.paste((255, 0, 0), mask=mk); vis.save(pre + "mask.png")
    t = time.time(); r, d = ai.erase(im, mk, print); print("erase", d, "%.2fs" % (time.time() - t)); r.save(pre + "erase.png")
    t = time.time(); r, d = ai.erase(im, mk); print("erase warm", d, "%.2fs" % (time.time() - t))
    if os.path.exists(os.path.join(SD_DIR, "unet")):
        t = time.time(); r, d = ai.change(im, mk, sys.argv[4] if len(sys.argv) > 4 else "a red rose", status=print); print("change", d, "%.2fs" % (time.time() - t)); r.save(pre + "change.png")
        t = time.time(); r, d = ai.change(im, mk, "a bright yellow sunflower", seed=3); print("change warm", d, "%.2fs" % (time.time() - t)); r.save(pre + "change2.png")
