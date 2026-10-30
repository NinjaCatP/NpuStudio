"""AI Change: local image analysis (Phi-3.5-vision, OpenVINO GenAI VLMPipeline), prompt building,
and whole-image img2img (LCM Dreamshaper, Image2ImagePipeline). Masked changes use editor.AI.change (SD inpaint)."""
import os, re, time, math, logging, threading
import numpy as np
from PIL import Image
APP = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(APP, "models"); CACHE = os.path.join(MD, "cache")
VLMS = [("OpenVINO/Phi-3.5-vision-instruct-int4-ov", "phi35-vision-int4", 2.6e9),
        ("OpenVINO/InternVL2-1B-int4-ov", "internvl2-1b-int4", 0.8e9)]
GDIR = os.path.join(MD, "lcm-dreamshaper-v7-int8")
BUCKETS = [(512, 512), (768, 576), (768, 512), (768, 448), (768, 384)]
BUCKETS += [(h, w) for w, h in BUCKETS if w != h]
VLM_DEVS = os.environ.get("NPU_VLM_DEVS", "GPU,CPU").split(",")
I2I_COMBOS = [("GPU", "NPU", "GPU"), ("GPU", "GPU", "GPU"), ("CPU", "CPU", "CPU")]
IDLE_UNLOAD = 300  # seconds: free the vision model's RAM when not used

def _download(repo, d, size, status):
    from huggingface_hub import snapshot_download
    if os.path.exists(os.path.join(d, ".done")): return
    stop = [False]
    def mon():
        while not stop[0]:
            sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) if os.path.isdir(d) else 0
            if status: status(f"Downloading vision model (one time, ~{size/1e9:.1f} GB): {min(99, sz*100/size):.0f}%")
            time.sleep(2)
    threading.Thread(target=mon, daemon=True).start()
    try: snapshot_download(repo, local_dir=d)
    finally: stop[0] = True
    open(os.path.join(d, ".done"), "w").write("ok")

def bucket_for(w, h):
    return min(BUCKETS, key=lambda b: abs(math.log((b[0] / b[1]) / (w / h))))

STYLE_RE = r"(watercolou?r|oil painting|painting|sketch|drawing|cartoon|anime|pixel art|illustration)"
def style_of(analysis):
    m = re.search(STYLE_RE, analysis or "", re.I); return m.group(1).lower() if m else "photo"

def template_prompt(analysis, instr):
    subj = re.split(r"[.\n]", analysis.strip())[0] if analysis.strip() else ""
    subj = re.sub(r"^(the|this) (image|picture|photo) (shows|depicts|features)\s*", "", subj, flags=re.I).strip(" ,.")
    ch = re.sub(r"(?i)^(please\s+)?(make|turn|change|convert)\s+(it|this|the (image|picture|photo|scene))\s+(into|to)?\s*", "", instr.strip().rstrip("."))
    return ", ".join(x for x in [subj, ch, "highly detailed, best quality"] if x)

def masked_prompt(analysis, rep):
    """Just the new object + light context: works far better for inpainting than a long scene prompt."""
    rep = re.sub(r"(?i)^(an?|the)\s+", "", rep.strip().rstrip("."))
    light = ""
    m = re.search(r"(?i)\b(bright|soft|warm|dim|dark|studio|natural|indoor|outdoor)\s+(lighting|light)\b", analysis or "")
    if m: light = m.group(0).lower()
    return ", ".join(x for x in [rep, light, style_of(analysis), "high quality, detailed"] if x)

CONTRA = [(r"\b(night|nighttime|dark|evening|midnight)\b", r"\b(bright|sunny|daylight|daytime|well[- ]lit|bright lighting|sunlight)\b"),
          (r"\b(day|daytime|sunny|morning|noon)\b", r"\b(night|nighttime|dark|moonlight)\b"),
          (STYLE_RE, r"\b(photo(graph)?( style)?|photorealistic|realistic)\b"),
          (r"\b(winter|snow)\b", r"\b(summer|sunny)\b")]
def clean_prompt(p, instr):
    """Drop words that contradict the request (e.g. 'bright lighting' for night)."""
    for want, bad in CONTRA:
        if re.search(want, instr, re.I):
            parts = [x.strip() for x in p.split(",")]
            p = ", ".join(x for x in parts if x and not re.search(bad, x, re.I))
    return re.sub(r"\s+", " ", p).strip(" ,")

IP2P_WHOLE = dict(steps=30, image_guidance=1.8, guidance=7.5)   # 1.8 keeps faces (tested)
IP2P_MASKED = dict(steps=30, image_guidance=1.5, guidance=7.5)

class Changer:
    def __init__(self):
        self.lock = threading.Lock(); self.vlm = None; self.vdev = None; self.vname = None
        self.i2i = None; self.i2i_shape = None; self.i2i_pref = None; self.i2i_devs = None; self.last_use = 0
        threading.Thread(target=self._idle, daemon=True).start()

    def _idle(self):
        while True:
            time.sleep(30)
            if self.vlm is not None and time.time() - self.last_use > IDLE_UNLOAD and self.lock.acquire(False):
                try: self.vlm = None; logging.info("aichange: vision model unloaded (idle)")
                finally: self.lock.release()

    def _load_vlm(self, status):
        import openvino_genai as og
        if self.vlm is not None: return
        err = []
        for repo, sub, size in VLMS:
            d = os.path.join(MD, sub)
            try: _download(repo, d, size, status)
            except Exception as e: err.append(f"{repo}: {e}"); continue
            for dev in VLM_DEVS:
                try:
                    if status: status(f"Loading vision model on {dev} (first time can take a minute)...")
                    t = time.time()
                    cfg = {"CACHE_DIR": CACHE} if dev != "NPU" else {"CACHE_DIR": CACHE}
                    self.vlm = og.VLMPipeline(d, dev, **cfg); self.vdev = dev; self.vname = repo.split("/")[1]
                    logging.info("aichange VLM %s on %s (%.1fs)", repo, dev, time.time() - t); return
                except Exception as e: err.append(f"{repo} {dev}: {str(e)[:200]}"); logging.warning("VLM %s %s: %s", repo, dev, e)
        raise RuntimeError("No vision model worked: " + " | ".join(err)[:600])

    def _ask(self, prompt, im=None, n=160):
        import openvino as ov, openvino_genai as og
        cfg = og.GenerationConfig(); cfg.max_new_tokens = n
        kw = {"generation_config": cfg}
        if im is not None:
            s = im.convert("RGB"); s.thumbnail((768, 768))
            kw["images"] = [ov.Tensor(np.asarray(s)[None].copy())]
        self.vlm.start_chat()
        try: r = self.vlm.generate(prompt, **kw)
        finally: self.vlm.finish_chat()
        return (r.texts[0] if hasattr(r, "texts") else str(r)).strip()

    def analyze(self, im, status=None):
        with self.lock:
            self._load_vlm(status); self.last_use = time.time()
            if status: status(f"Analyzing image on {self.vdev}...")
            t = time.time()
            txt = self._ask("Describe this image in 2-3 short sentences: the scene, the main objects and people, "
                            "their colors, the lighting and the style (photo, painting, etc.).", im)
            self.last_use = time.time(); return txt, self.vdev, time.time() - t

    FEWSHOT = (
        "Rewrite the image description so it describes the FINAL image after the change. Output only the new description as "
        "comma-separated keywords. Never write an instruction (no 'make', 'change', 'replace', 'turn').\n"
        "Image: a woman in a blue coat on a city street, bright daylight, photo. Change: make it nighttime -> "
        "a woman in a blue coat on a city street at night, dark sky, glowing street lights, moonlight, photo\n"
        "Image: a red barn in a green field, sunny, photo. Change: turn it into an oil painting -> "
        "oil painting of a red barn in a green field, visible brush strokes, painterly, canvas texture\n"
        "Image: a dog on a sofa, soft indoor light, photo. Change: make it winter -> "
        "a dog on a sofa, snowy window, cold winter light, cozy, photo\n")

    def build_prompt(self, analysis, instr, masked, status=None):
        """Returns (prompt, how). Masked: replacement object + light context. Global: description of the final scene."""
        tgt, rep, kind = parse(instr)
        if masked:
            return masked_prompt(analysis, rep or instr), "direct"
        if self.vlm is None or not self.lock.acquire(False): return clean_prompt(template_prompt(analysis, instr), instr), "template"
        try:
            self.last_use = time.time()
            q = self.FEWSHOT + f"Image: {analysis.strip()} Change: {instr.strip()} ->"
            p = self._ask(q, None, 80).strip().strip('"').splitlines()[0].strip()
            p = re.sub(r"^(prompt|description|final image|output)\s*:\s*", "", p, flags=re.I).split("->")[-1].strip()
            if not p or re.match(r"(?i)^(make|change|replace|turn|convert|add)\b", p): p = template_prompt(analysis, instr)
            return clean_prompt(p, instr), self.vname
        except Exception:
            logging.exception("build_prompt"); return clean_prompt(template_prompt(analysis, instr), instr), "template"
        finally: self.lock.release()

    def _i2i(self, w, h, status):
        import openvino_genai as og
        if self.i2i is None: self.i2i = og.Image2ImagePipeline(GDIR)
        import devices
        if self.i2i_shape == (w, h) and self.i2i_pref == devices.get(): return
        self.i2i_pref = devices.get()
        import devices
        self.i2i.reshape(1, h, w, 8.0); err = []
        labs = devices.plan("img2img")
        if not labs: raise RuntimeError(devices.unsupported("img2img"))
        if devices.get() == "All" and len(labs) > 1:   # fastest device per stage: UNet on the best, text/VAE on the best GPU
            gpu = [l for l in labs if l != "NPU"] or labs
            combos = [(devices.ov(gpu[0]), devices.ov(labs[0]), devices.ov(gpu[0]))]
        else: combos = []
        combos += [(devices.ov(l),) * 3 for l in labs]
        for c in combos:
            try:
                if status: status(f"Compiling img2img {w}x{h} (text {c[0]}, UNet {c[1]}, VAE {c[2]}) - first time takes a few minutes...")
                t = time.time(); self.i2i.compile(c[0], c[1], c[2], CACHE_DIR=CACHE)
                self.i2i_shape = (w, h); self.i2i_devs = c; logging.info("img2img %s on %s (%.1fs)", (w, h), c, time.time() - t); return
            except Exception as e: err.append(f"{c}: {str(e)[:200]}"); logging.warning("img2img %s: %s", c, e)
        raise RuntimeError("img2img: no device worked: " + " | ".join(err))

    def edit_whole(self, im, instr, analysis="", strength=0.4, seed=1, progress=None, status=None, upscaler=None):
        """Whole-image change. NVIDIA: InstructPix2Pix (takes the instruction directly, keeps faces).
        Otherwise LCM img2img with a rewritten scene prompt. Returns (img, devices, how, prompt_used)."""
        import devices
        if devices.plan("ip2p"):
            try:
                import cudaeng
                if cudaeng.available():
                    kw = dict(IP2P_WHOLE); ins2 = instr
                    if re.search(STYLE_RE, instr, re.I):   # styles: 512 px + "make it a ..." phrasing tested best
                        kw["side"] = 512; ins2 = re.sub(r"(?i)^(please\s+)?(turn|convert|change)\s+(it|this|the (image|picture|photo))\s+(into|to)\s+", "make it ", instr.strip())
                    t = time.time(); r = cudaeng.ip2p(im, ins2, progress=progress, status=status, seed=seed, **kw); instr = ins2
                    out, how = self._back(im, r, upscaler, status)
                    logging.info("ip2p %.1fs", time.time() - t)
                    return out, ("NVIDIA",), "InstructPix2Pix + " + how, instr
            except Exception: logging.exception("ip2p failed, falling back to img2img")
        p, _ = self.build_prompt(analysis, instr, False, status)
        out, devs, how = self.whole(im, p, strength, seed, progress, status, upscaler)
        return out, devs, "LCM img2img + " + how, p

    ATTR = r"\b(outfit|clothes|clothing|shirt|top|dress|jacket|coat|suit|pants|skirt|shoes|hair|skin|eyes|eye|lips|eyebrows|beard|face|arms|legs|hands|color|colour|hat)\b"
    def edit_masked(self, ai, im, mask, instr, analysis="", strength=1.0, progress=None, status=None, upscaler=None, method="Auto"):
        """Masked change, routed by intent (method Auto/Redraw/Recolor):
          recolor = colour/material change of the same thing -> InstructPix2Pix kept inside the mask (NVIDIA)
          redraw  = new object / garment ("outfit to red dress", "replace X with Y") -> SD inpainting, strength 1.0,
                    style words from the analysis, negative prompt naming the old thing, bigger feathered mask with
                    face/hands/hair protected. Returns (img, devs, how, prompt)."""
        import devices
        tgt, rep, kind = parse(instr)
        if kind == "remove":
            r, d = ai.erase(im, mask, status); return r, d, "Method: erase (LaMa)", "(erase)"
        cuda_ok = False
        if devices.plan("ip2p"):
            try:
                import cudaeng; cuda_ok = cudaeng.available()
            except Exception: pass
        m = method if method in ("Redraw", "Recolor") else method_for(rep)
        if m == "Recolor" and cuda_ok:
            import cudaeng
            r = cudaeng.ip2p(im, instr, progress=progress, status=status, **IP2P_MASKED)
            full, how = self._back(im, r, upscaler, status)
            return ai._paste(im, full.convert("RGB"), mask, (0, 0) + im.size), ("NVIDIA",), "Method: recolor (InstructPix2Pix in mask) + " + how, instr
        p, neg = redraw_prompt(analysis, tgt, rep)
        garment = is_garment(tgt, rep); col = color_of(rep)
        if tgt == "arms" and POSE_KEY.get(rep):
            try:
                import posegen
                if status: status("Finding her pose (OpenPose)...")
                res = posegen.redraw_pose(im, POSE_KEY[rep], change_prompt(analysis, tgt, rep, None, False), progress=progress, status=status)
                if res is not None:
                    r, how = res
                    if im.mode == "RGBA": r = r.convert("RGBA"); r.putalpha(im.getchannel("A"))
                    return r, ("NVIDIA",), how, rep
            except Exception: logging.exception("pose redraw")
        if tgt == "arms": hard, soft = pose_mask(im, mask); garment = False
        elif re.fullmatch(r"(?i)\s*hair\s*", tgt or ""): hard, soft = hair_mask(im, mask)
        elif garment: hard, soft = outfit_mask(im, mask, tgt)
        else:
            hard = redraw_mask(im, mask, tgt).point(lambda v: 255 if v > 100 else 0); soft = hard.filter(__import__("PIL.ImageFilter").ImageFilter.GaussianBlur(1.5))
            if re.search(r"\b(background|sky|wall|scene)\b", tgt or "", re.I):  # never paint over the person
                hs = bg_mask(ai, im, hard)
                if hs: hard, soft = hs
        if not hard.getbbox(): hard, soft = mask.convert("L"), mask.convert("L")
        ctl = control_mode(tgt, rep)
        hair_short = bool(re.fullmatch(r"(?i)\s*hair\s*", tgt or "")) and bool(re.search(r"\b(short|bob|pixie|buzz|bald)\b", rep or "", re.I))
        cn_ok = False
        try:
            import cudaeng; cn_ok = cudaeng.available() and bool(devices.plan("inpaint")) and devices.plan("inpaint")[0] == "NVIDIA"
        except Exception: pass
        best = None; N = 4; d = ("NVIDIA",)
        if cn_ok:
            # general engine: ControlNet (anime lineart) inpaint inside the target mask. "keep" = outline held, interior
            # lines removed (same-shape swaps); "loose" = only the surroundings guide it, so the shape may change inside the mask
            cp, cneg = change_prompt(analysis, tgt, rep, col, garment)
            for i in range(N):
                if status: status(f"Drawing '{rep}' - version {i + 1} of {N} (ControlNet {ctl}, NVIDIA)...")
                try: r = cn_redraw(im, hard, soft, cp, cneg, seed=11 + i * 7, progress=progress, status=status, mode=ctl)
                except Exception: logging.exception("controlnet redraw"); break
                cs_ = color_score(r, hard, col) if col else 1.0; ch = changed_score(im, r, hard)
                sc = cs_ * (1.0 if ch > 0.35 else 0.4)
                if hair_short: sc = hair_short_score(im, r, hard, col)
                if tgt == "arms":
                    try:
                        import posegen
                        if posegen.hand_on_chest(r): sc = -1; logging.info("candidate %d rejected: hand on chest/torso", i)
                    except Exception: logging.exception("chest filter")
                logging.info("redraw candidate %d colour %.2f changed %.2f -> %.2f", i, cs_, ch, sc)
                if os.environ.get("NPUSTUDIO_DEBUG"): r.save(f"cand_{i}.jpg", quality=85)
                if best is None or sc > best[0]: best = (sc, r, d, cs_, ch)
            p = cp
        if best is None:  # no NVIDIA: plain inpaint (OpenVINO / CUDA), single pass unless a colour can be checked
            for i in range(N if col else 1):
                if status: status(f"Drawing version {i + 1}...")
                r, d = ai.change(im, hard, p, steps=40, strength=1.0, seed=1 + i * 7, guidance=8.0, progress=progress, status=None, negative=neg, grow=5)
                r = Image.composite(r.convert("RGB"), im.convert("RGB"), soft)
                cs_ = color_score(r, hard, col) if col else 1.0; ch = changed_score(im, r, hard); sc = cs_ * (1.0 if ch > 0.35 else 0.4)
                if best is None or sc > best[0]: best = (sc, r, d, cs_, ch)
        sc, r, d, cs_, ch = best
        how = f"Method: redraw ({'ControlNet ' + ctl if cn_ok else 'inpaint'}, best of {N}, changed {ch:.2f}" + (f", colour {cs_:.2f})" if col else ")")
        if hair_short and sc >= 0.6: how += f" (short-hair check {sc:.2f})"
        elif hair_short: how += f" - short-hair check failed ({sc:.2f})"; sc = 1.0
        if sc < 0: how = "Method: redraw - every version put a hand on the body, so nothing was kept; try again"; r = im.convert("RGB")
        if sc < COLOR_OK and col and (garment or re.search(r"\bhair\b", f"{tgt} {rep}", re.I)) and not re.search(SHAPE_CHANGE, rep or "", re.I):
            r = recolor_fill(im.convert("RGB"), hard, col); how = f"Method: LAST RESORT recolor of the original (redraw failed: changed {ch:.2f}, colour {cs_:.2f})"
        elif sc < COLOR_OK: how += " - weak result, try another wording or brush the area"
        if im.mode == "RGBA": r = r.convert("RGBA"); r.putalpha(im.getchannel("A"))
        return r, d, how, p

    def _back(self, im, out, upscaler, status):
        W, H = im.size; how = "Lanczos"; f = max(W / out.size[0], H / out.size[1])
        if upscaler is not None and f > 1.5:
            try:
                if status: status("Upscaling result back to full size (Sharp)...")
                out = upscaler(out, f); how = "Sharp"
            except Exception: logging.exception("aichange upscale")
        out = out.resize((W, H), Image.LANCZOS)
        if im.mode == "RGBA": out = out.convert("RGBA"); out.putalpha(im.getchannel("A"))
        return out, how

    def whole(self, im, prompt, strength=0.4, seed=1, progress=None, status=None, upscaler=None):
        import openvino as ov
        if not os.path.exists(os.path.join(GDIR, "unet", "openvino_model.xml")):
            raise RuntimeError("Image Gen model not downloaded yet - open the Image Gen tab once first.")
        W, H = im.size; bw, bh = bucket_for(W, H)
        with self.lock:
            self._i2i(bw, bh, status)
            src = im.convert("RGB").resize((bw, bh), Image.LANCZOS)
            steps = max(4, math.ceil(4 / max(0.15, strength)))  # ~4 effective LCM steps
            def cb(step, n, lat):
                if progress: progress(step + 1, n)
                return False
            r = self.i2i.generate(prompt, image=ov.Tensor(np.asarray(src)[None].copy()), strength=float(strength),
                                  num_inference_steps=steps, rng_seed=int(seed), callback=cb)
            out = Image.fromarray(np.array(r.data)[0]); devs = self.i2i_devs
        how = "Lanczos"
        if upscaler is not None and max(W / bw, H / bh) > 1.5:
            try:
                if status: status("Upscaling result back to full size (Sharp)...")
                out = upscaler(out, max(W / bw, H / bh)); how = "Sharp"
            except Exception: logging.exception("aichange upscale")
        out = out.resize((W, H), Image.LANCZOS)
        if im.mode == "RGBA": out = out.convert("RGBA"); out.putalpha(im.getchannel("A"))
        return out, devs, how

# ---------------- text-prompted segmentation (CLIPSeg rd64-refined, OpenVINO IR) ----------------
SEG_DIR = os.path.join(MD, "clipseg")
class Seg:
    def __init__(self, devs=("NPU", "GPU", "CPU")):
        import openvino as ov, openvino_tokenizers  # noqa: registers tokenizer ops
        c = ov.Core(); c.set_property({"CACHE_DIR": CACHE})
        self.tok = c.compile_model(os.path.join(SEG_DIR, "clipseg_tokenizer.xml"), "CPU"); self.err = []
        m = c.read_model(os.path.join(SEG_DIR, "clipseg.xml"))
        for d in devs:
            try: t = time.time(); self.cm = c.compile_model(m, d); self.device = d; logging.info("clipseg on %s (%.1fs)", d, time.time() - t); return
            except Exception as e: self.err.append(f"{d}: {str(e)[:150]}"); logging.warning("clipseg %s: %s", d, e)
        raise RuntimeError("CLIPSeg: no device worked: " + " | ".join(self.err))
    def probs(self, im, text):
        r = self.tok([text]); ids = np.full((1, 77), 49407, np.int64); am = np.zeros((1, 77), np.int64)
        t = list(r.values()); a = np.asarray(t[0])[0][:77]; ids[0, :len(a)] = a; am[0, :len(a)] = 1
        x = np.asarray(im.convert("RGB").resize((352, 352), Image.BILINEAR), np.float32) / 255.0
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
        lg = np.asarray(self.cm([ids, am, x])[0]).reshape(352, 352)
        return 1 / (1 + np.exp(-lg))
    def mask(self, im, targets, thresh=0.4):
        """Union of probabilities over synonyms; adaptive threshold; returns L mask at image size."""
        p = np.max([self.probs(im, t) for t in targets], axis=0)
        th = max(0.12 if p.max() < 0.5 else 0.25, min(thresh, p.max() * 0.5))  # lower floor for drawings / weak matches
        m = Image.fromarray(((p > th) * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)
        from PIL import ImageFilter
        m = m.filter(ImageFilter.GaussianBlur(2)).point(lambda v: 255 if v > 127 else 0)
        return m, float(p.max())

def grow(mask, px):
    from PIL import ImageFilter
    if px == 0: return mask
    k = abs(int(px)) * 2 + 1
    return mask.filter(ImageFilter.MaxFilter(k) if px > 0 else ImageFilter.MinFilter(k))

GLOBAL_WORDS = r"\b(night|nighttime|day|daytime|sunset|sunrise|winter|summer|autumn|fall|snow|rain|watercolou?r|painting|oil|sketch|cartoon|anime|pixel|style|vintage|black and white|b&w|cinematic|lighting|mood|whole|everything|entire|image|picture|photo)\b"
POSS = r"(?:the\s+|her\s+|his\s+|their\s+|its\s+|my\s+|your\s+|this\s+|that\s+)?"
SCENE = r"(it|this|that|the image|the picture|the photo|everything|image|picture|photo|the scene|scene|the whole (?:image|picture|photo))"
POSE_TXT = {"wave": "waving with one hand raised", "waving": "waving with one hand raised", "arms": "arms crossed over chest",
            "arm": "arms crossed over chest", "cross": "arms crossed over chest", "crossed": "arms crossed over chest", "arms crossed": "arms crossed over chest",
            "hands": "hands on hips", "hand": "hands on hips", "thumbs": "giving a thumbs up", "thumb": "giving a thumbs up",
            "peace": "making a peace sign with her hand", "salute": "saluting", "saluting": "saluting", "point": "pointing with her finger",
            "pointing": "pointing with her finger", "raise": "one arm raised", "raised": "one arm raised", "raises": "one arm raised"}
POSE_KEY = {"waving with one hand raised": "wave", "arms crossed over chest": "crossed", "hands on hips": "hips", "one arm raised": "raise"}
def parse(instr):
    """-> (target or None, replacement, kind) where kind in remove/replace/global."""
    _p = re.search(r"(?i)\b(wav(e|ing)|arms? crossed|cross(ed)? (her|his|their) arms|hands? on (her|his) hips|thumbs? up|peace sign|salut(e|ing)|point(ing)?|raise[sd]? (her|his|their|an?) (arm|hand)s?)\b", instr or "")
    if _p: return "arms", POSE_TXT.get(_p.group(1).lower().split()[0], _p.group(0).lower()), "replace"
    _h = re.search(r"(?i)\b(?:hold|holding|hold up|give (?:her|him|them) an?)\s+(?:an?\s+|the\s+|some\s+)?([a-z \-]+)$", instr or "")
    if _h: return "arms", f"holding a {_h.group(1).strip()} in her hand", "replace"
    _m = re.match(r"(?i)^\s*(?:make|turn)\s+(?:the\s+|her\s+|his\s+|their\s+)?(hair|outfit|dress|shirt|hat|background|sky)\s+(?!to\b|into\b|an?\b)([a-z ,\-]+?)\s*$", instr or "")
    if _m: return _m.group(1).lower(), f"{_m.group(2).strip()} {_m.group(1).lower()}", "replace"
    s = instr.strip().rstrip(".!").strip(); s = re.sub(r"^(?:please|can you|could you)\s+", "", s, flags=re.I)
    s = fix_verb(s)
    if re.search(r"\b(whole|entire) (image|picture|photo|scene)\b", s, re.I): return None, s, "global"
    m = re.match(r"^(?:give|add)\s+(?:her|him|them|the \w+|it)?\s*(?:a\s+|an\s+|some\s+)?(.+?)(?:\s+to\s+" + POSS + r"(.+))?$", s, re.I)
    if m:  # "give her blue eyes", "give him a beard", "add a hat to the man"
        what = m.group(1).strip()
        try:
            import bodyparse; k = bodyparse.key_for(what)
        except Exception: k = None
        if k: return k[1], what, "replace"
        if m.group(2): return m.group(2).strip(), what, "replace"
        return None, s, "global"
    pats = [(r"^(?:remove|delete|erase|get rid of)\s+" + POSS + r"(.+)$", "remove"),
            (r"^(?:replace|swap)\s+" + POSS + r"(.+?)\s+(?:with|for)\s+(.+)$", "replace"),
            (r"^(?:change|turn|transform|convert|make|switch)\s+" + POSS + r"(.+?)\s+(?:to|into)\s+(?:an?\s+|the\s+)?(.+)$", "replace"),
            (r"^(?:make)\s+" + POSS + r"(.+?)\s+(?:into\s+)?(?:an?)\s+(.+)$", "replace"),
            (r"^(?:make|paint|color|colour|dye)\s+" + POSS + r"(.+?)\s+(?:look\s+|be\s+)?(\S+(?:\s+\S+)?)$", "replace"),
            (r"^" + POSS + r"(.+?)\s+(?:to|into|->)\s+(?:an?\s+|the\s+)?(.+)$", "replace")]
    for p, kind in pats:
        m = re.match(p, s, re.I)
        if not m: continue
        tgt = m.group(1).strip()
        if kind != "remove" and re.fullmatch(SCENE, tgt, re.I):
            return None, m.group(2).strip(), "global"
        if kind == "remove": return tgt, "", "remove"
        # "make her hair bright blue": keep body-part noun as target if the split went wrong
        try:
            import bodyparse
            full = (tgt + " " + m.group(2)).split()
            if not bodyparse.key_for(tgt):
                for i in range(len(full), 0, -1):
                    if bodyparse.key_for(" ".join(full[:i])) and bodyparse.key_for(full[i - 1]):
                        tgt, rep = " ".join(full[:i]), " ".join(full[i:]); return tgt, rep or m.group(2).strip(), "replace"
        except Exception: pass
        return tgt, m.group(2).strip(), "replace"
    # no verb pattern: any known object noun anywhere ("red dress for the outfit", "outfit red")
    t = noun_in(s)
    if t and not is_global(s): return t, s, "replace"
    return None, s, "global"

COLORS = r"(red|blue|green|yellow|orange|purple|violet|pink|black|white|grey|gray|brown|blonde|blond|silver|gold|golden|beige|navy|teal|cyan|magenta|maroon|turquoise|dark|light|bright|pale|pastel|neon|leather|denim|silk|velvet|wool|metal|metallic|shiny|matte|striped|plaid|polka|dotted|floral|darker|lighter|colou?red|colorful|colourful)"
def method_for(rep):
    """'Recolor' when the new value is only colours/materials/adjectives ("blue", "dark red", "leather"), else 'Redraw'."""
    words = [w for w in re.findall(r"[a-z\-]+", (rep or "").lower()) if w not in ("a", "an", "the", "and", "with", "more", "very", "color", "colour", "shade", "of", "in", "to")]
    words = [w for w in words if not re.fullmatch(r"(hair|outfit|clothes|background|sky)", w)] or words
    if words and all(re.fullmatch(COLORS, w) for w in words): return "Recolor"
    return "Redraw"

ILLU = r"\b(cartoon|anime|manga|illustration|illustrated|drawing|drawn|comic|animated|cel|vector|digital art|chibi)\b"
def redraw_prompt(analysis, tgt, rep):
    rep = re.sub(r"(?i)^(an?|the)\s+", "", (rep or "").strip().rstrip("."))
    if re.search(ILLU, analysis or "", re.I): style = "anime illustration, clean line art, bold outlines, flat vibrant colors"
    else: style = style_of(analysis) + ", realistic, natural lighting"
    garment = re.search(r"\b(outfit|clothes|clothing|shirt|top|jacket|coat|suit|uniform|pants|skirt)\b", (tgt or "") + " " + rep, re.I)
    p = ", ".join(x for x in [("wearing a " + rep) if garment else rep, rep, style, "high quality, detailed"] if x)
    old = [tgt or ""]
    for w in re.findall(r"\b(?:\w+\s)?(?:space ?suit|uniform|suit|jacket|coat|shirt|scrubs|dress|apron)\b", analysis or "", re.I)[:3]:
        if w.lower() not in rep.lower(): old.append(w.lower())
    col = color_of(rep)
    if garment and col:
        p = ", ".join([f"wearing a {rep}", f"{col} {col} fabric", f"solid {col} {rep}", style.split(",")[0], "flat colors, high quality"])
        others = [c for c in ("white", "brown", "beige", "tan", "skin tone", "nude", "flesh", "black", "grey", "blue", "green", "yellow", "pink", "orange", "purple") if c not in col]
        old += [", ".join(others), "skin tone clothes, bare skin, nude"]
    neg = ", ".join([x for x in old if x] + ["blurry, deformed, low quality, extra limbs, bad anatomy, changed body shape"])
    return p, neg

def redraw_mask(im, mask, tgt):
    """Grow + feather the target mask, but keep face, hair and hands out of it (for clothing)."""
    from PIL import ImageFilter
    m = grow(mask, max(6, int(min(im.size) * 0.012)))
    try:
        import bodyparse
        k = bodyparse.key_for(tgt or "")
        if k and k[0] == "human" and k[1] not in ("hair", "face", "head", "skin", "body", "arms", "arm", "hands", "hand"):
            keep = bodyparse.mask(im, "face")[0]; hair = bodyparse.mask(im, "hair")[0]; arms = bodyparse.mask(im, "hands")[0]
            prot = np.maximum.reduce([np.asarray(keep), np.asarray(hair), np.asarray(arms)])
            prot = np.asarray(Image.fromarray(prot).filter(ImageFilter.MaxFilter(5)))
            m = Image.fromarray(np.where(prot > 127, 0, np.asarray(m)).astype(np.uint8))
    except Exception: logging.exception("redraw_mask protect")
    return m.filter(ImageFilter.GaussianBlur(3))

VERBS = ["change", "replace", "make", "turn", "swap", "remove", "delete", "erase", "transform", "convert", "paint", "color", "colour", "dye", "give", "add", "switch"]
def fix_verb(s):
    """Fuzzy-correct a misspelled first verb: charge/chnage/chane -> change, repalce -> replace."""
    import difflib
    w = s.split(" ", 1)
    if not w or w[0].lower() in VERBS: return s
    m = difflib.get_close_matches(w[0].lower(), VERBS, n=1, cutoff=0.6)
    return (m[0] + (" " + w[1] if len(w) > 1 else "")) if m else s

OBJ_NOUNS = r"\b(outfit|clothes|clothing|dress|shirt|top|jacket|coat|suit|uniform|pants|skirt|shoes|hat|helmet|hair|face|eyes|lips|eyebrows|beard|arms|hands|legs|skin|background|sky|wall|flag|rocket|car|tree|dog|cat|bag|glasses|sunglasses|scarf|belt)\b"
def noun_in(s):
    m = re.findall(OBJ_NOUNS, s, re.I)
    return m[0].lower() if m else None

def is_global(s):
    return bool(re.search(GLOBAL_WORDS, s, re.I)) and not re.search(OBJ_NOUNS, s, re.I) or bool(re.search(r"\bwhole (image|picture|photo)\b|\bentire (image|picture|photo)\b", s, re.I))

SYN = {"outfit": ["clothing", "suit", "outfit"], "clothes": ["clothing", "suit"], "shirt": ["shirt", "top"], "hair": ["hair"],
       "background": ["background", "wall"], "sky": ["sky"], "face": ["face"]}
def targets_for(tgt, analysis=""):
    t = tgt.lower(); out = [tgt]
    for k, v in SYN.items():
        if k in t: out += v
    # if the analysis names a more specific thing (e.g. "orange spacesuit"), add it
    for w in re.findall(r"\b(?:\w+\s)?(?:space ?suit|suit|dress|jacket|shirt|coat|uniform)\b", analysis, re.I)[:2]:
        if any(k in t for k in ("outfit", "cloth", "suit", "dress", "jacket", "shirt", "coat", "uniform")): out.append(w)
    return list(dict.fromkeys(out))


# ---------------- clothing changes: whole-outfit mask, colour pick, recolor fallback ----------------
COLOR_RGB = {"red": (205, 20, 35), "blue": (30, 70, 200), "green": (30, 150, 60), "yellow": (240, 210, 30), "orange": (240, 120, 20),
             "purple": (120, 40, 170), "violet": (140, 60, 200), "pink": (240, 90, 160), "black": (20, 20, 22), "white": (245, 245, 245),
             "grey": (128, 128, 128), "gray": (128, 128, 128), "brown": (110, 65, 30), "navy": (20, 30, 90), "teal": (0, 128, 128),
             "cyan": (0, 190, 220), "magenta": (210, 0, 150), "maroon": (110, 15, 25), "turquoise": (40, 200, 190), "gold": (212, 175, 55),
             "golden": (212, 175, 55), "silver": (190, 190, 195), "beige": (220, 200, 165)}
COLOR_OK = 0.6
GARMENT = r"\b(outfit|clothes|clothing|shirt|top|dress|jacket|coat|suit|uniform|pants|skirt|blouse|sweater|hoodie|gown|robe|kimono|bikini|swimsuit)\b"
def is_garment(tgt, rep): return bool(re.search(GARMENT, f"{tgt or ''} {rep or ''}", re.I))
def color_of(rep):
    for w in re.findall(r"[a-z]+", (rep or "").lower()):
        if w in COLOR_RGB: return w
    return None

def _seg():
    global _SEG
    try: return _SEG
    except NameError: pass
    try: _SEG = Seg(("GPU", "CPU"))
    except Exception: logging.exception("clipseg for outfit mask"); _SEG = None
    return _SEG

def outfit_mask(im, mask, tgt):
    """-> (hard clothing mask, soft edge mask). Coarse area = union of the given mask, SegFormer clothes classes
    (upper-clothes/dress/skirt/pants) and CLIPSeg 'outfit'; face/hair/arms/legs/skin are cut out; then the mask is
    snapped to the garment's own colour so it follows the original outline exactly (the body is never redrawn)."""
    from PIL import ImageFilter
    W, H = im.size; u = np.asarray(mask.convert("L")) > 127; prot = np.zeros((H, W), bool); seg_clothes = np.zeros((H, W), bool)
    try:
        import bodyparse
        if bodyparse.available():
            hn = bodyparse.net("human-parse"); lab = hn.seg(im)
            seg_clothes = np.isin(lab, hn.ids(["upper-clothes", "dress", "skirt", "pants", "belt"])); u |= seg_clothes
            prot |= np.isin(lab, hn.ids(["face", "hair", "left-arm", "right-arm", "left-leg", "right-leg"]))
    except Exception: logging.exception("outfit_mask bodyparse")
    s = _seg()
    def cs(words, lo, frac):
        p = np.max([s.probs(im, t) for t in words], axis=0)
        return np.asarray(Image.fromarray(((p > max(lo, p.max() * frac)) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) > 127
    if s is not None:
        try:
            u |= cs(("outfit", "clothing", "dress", "shirt", "skirt", "uniform"), 0.2, 0.35)
            prot |= cs(("face", "hair"), 0.35, 0.5)
        except Exception: logging.exception("outfit_mask clipseg")
    g = max(3, int(min(W, H) * 0.006))
    coarse = np.asarray(grow(Image.fromarray((u * 255).astype(np.uint8)), g * 4)) > 127
    core = u & ~prot
    a = np.asarray(im.convert("RGB"), np.float32)
    if core.sum() > 200:
        # garment colour = dominant colour of the core (median of its brighter half avoids line art / shadows)
        px = a[core]; lum = px.mean(1); ref = np.median(px[lum >= np.median(lum)], 0)
        d = np.sqrt(((a - ref) ** 2).sum(-1))
        sat = a.max(-1) - a.min(-1); rsat = ref.max() - ref.min()
        ok = (d < 70) | ((np.abs(sat - rsat) < 40) & (np.abs(a.mean(-1) - ref.mean()) < 80) & (a.mean(-1) > 60))  # same fabric in shadow
        m = ok & coarse & ~prot
        mi = Image.fromarray((m * 255).astype(np.uint8))
        k = g * 2 + 1  # close: pulls in line art / seams / badge inside the garment, not the outer body outline
        mi = mi.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MedianFilter(5))
        hard = np.asarray(mi) > 127
        hard &= ~prot
    else:
        hard = np.asarray(grow(Image.fromarray((u * 255).astype(np.uint8)), g)) > 127
        hard &= ~prot
    hard = Image.fromarray((hard * 255).astype(np.uint8))
    return hard, hard.filter(ImageFilter.GaussianBlur(1.2))

def color_score(img, mask, col):
    """0..1: how close the masked pixels are to the requested colour (hue for chromatic colours, value for black/white/grey)."""
    import colorsys
    m = np.asarray(mask) > 127
    if not m.any(): return 0.0
    hsv = np.asarray(img.convert("RGB").convert("HSV"), np.float32)[m] / 255.0
    th, ts, tv = colorsys.rgb_to_hsv(*[c / 255 for c in COLOR_RGB[col]])
    if ts < 0.15:  # achromatic
        return float(np.mean((hsv[:, 1] < 0.25) & (np.abs(hsv[:, 2] - tv) < 0.3)))
    dh = np.abs(hsv[:, 0] - th); dh = np.minimum(dh, 1 - dh)
    ok = (dh < 0.08) & (hsv[:, 1] > 0.25) & (hsv[:, 2] > 0.12)
    return float(np.mean(ok))

def recolor_fill(img, mask, col):
    """Paint the masked garment in the target colour, keeping the original shading and line art."""
    a = np.asarray(img.convert("RGB"), np.float32); m = (np.asarray(mask.filter(__import__("PIL.ImageFilter").ImageFilter.GaussianBlur(1)), np.float32) / 255.0)[..., None]
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])[..., None] / 255.0
    ref = np.percentile(lum[m[..., 0] > 0.5], 90) if (m > 0.5).any() else 1.0
    shade = np.clip(lum / max(ref, 1e-3), 0, 1.15)
    tgt = np.array(COLOR_RGB[col], np.float32)[None, None] * shade
    out = a * (1 - m) + np.clip(tgt, 0, 255) * m
    return Image.fromarray(out.astype(np.uint8))


SHAPE_CHANGE = r"\b(short|long|longer|shorter|curly|straight|wavy|bun|ponytail|braid|braids|bald|crown|hoodie|jacket|coat|cape|hat|helmet|wings|beach|forest|city|space|mountains?|ocean|sea|garden|castle|sky|bigger|smaller|replace|swap)\b"
def control_mode(tgt, rep):
    """'keep' = hold the target's outline (same-shape swap, e.g. outfit -> dress); 'loose' = new shape allowed inside the mask."""
    t = f"{tgt or ''}".lower()
    if t == "arms": return "loose"
    if re.search(r"\b(background|sky|wall|scene|hair|hat|cap|helmet)\b", t): return "loose"
    if re.search(r"\b(hoodie|jacket|coat|cape|wings|crown)\b", rep or "", re.I): return "loose"
    if is_garment(tgt, rep): return "keep"
    return "loose" if re.search(SHAPE_CHANGE, rep or "", re.I) else "keep"

def change_prompt(analysis, tgt, rep, col, garment):
    anime = bool(re.search(ILLU, analysis or "", re.I))
    style = "anime illustration, clean line art, flat cel shading" if anime else "photo, realistic, natural lighting"
    t = (tgt or "").lower()
    if garment: subj = f"a woman wearing a plain {rep}, simple solid-color {rep}" + (f", {col} {col} fabric, solid {col}" if col else "")
    elif t == "arms": subj = f"a woman {rep}, bare arm, detailed hand, five fingers"
    elif re.search(r"\bhair\b", t):
        r2 = re.sub(r"\bhair\b", "", rep or "").strip()
        subj = f"{r2} hair, {r2} hairstyle" + (f", {col} {col} hair" if col else "")
        if re.search(r"\b(short|bob|pixie)\b", rep or "", re.I): subj += ", short bob haircut ending above the shoulders, plain wall background behind her neck and shoulders"
    elif re.search(r"\b(background|sky|wall|scene)\b", t): subj = f"{rep} background, {rep} scenery, detailed"
    else: subj = f"{rep}, a {rep}" + (f", {col} {col}" if col else "")
    p = f"{subj}, {style}, high quality"
    others = ", ".join(c for c in ("white", "brown", "beige", "black", "grey", "blue", "green", "yellow", "pink", "purple", "red") if c != col) if col else ""
    old = [w for w in re.findall(r"\b(?:\w+\s)?(?:uniform|suit|jacket|coat|shirt|dress|cap|hat|helmet|wall|background)\b", analysis or "", re.I)[:3]
           if w.lower() not in (rep or "").lower()]
    if garment: old += ["pattern, print, text, logo, floral, graffiti", "nurse uniform", "collar", "name badge", "buttons", "zipper", "belt", "skin tone fabric", "bare skin", "nude"]
    if t == "arms": old += ["hand on chest, hand on breast, touching body, groping, hands on torso"]
    if re.search(r"\bhair\b", t) and re.search(r"\b(short|bob|pixie)\b", rep or "", re.I): old += ["long hair, rainbow hair, colorful hair, multicolored hair"]
    neg = ", ".join([x for x in [tgt if tgt and tgt.lower() not in (rep or "").lower() else "", *old, others] if x] +
                    ["extra limbs, deformed, changed body shape, blurry, low quality"])
    return p, neg

def control_map(im, hard, mode="keep"):
    """Anime lineart control (white lines on black) from the original: all line art outside the clothing,
    the clothing's outer outline kept, the clothing's interior lines (collar, badge, zipper, belt) removed."""
    from PIL import ImageFilter
    g = np.asarray(im.convert("L"), np.float32)
    blur = np.asarray(im.convert("L").filter(ImageFilter.GaussianBlur(3)), np.float32)
    lines = (g < blur - 18) | (g < 60)
    m = np.asarray(hard) > 127
    inner = np.asarray(hard.filter(ImageFilter.MinFilter(9))) > 127
    outline = m & ~inner
    lines = (lines & ~inner) | outline if mode == "keep" else (lines & ~m)
    return Image.fromarray((lines * 255).astype(np.uint8)).convert("RGB")

def cn_redraw(im, hard, soft, prompt, neg, seed=1, progress=None, status=None, side=768, mode="keep"):
    import cudaeng
    from PIL import ImageFilter
    W, H = im.size; bb = hard.getbbox(); pad = int(max(bb[2] - bb[0], bb[3] - bb[1]) * 0.15)
    box = (max(0, bb[0] - pad), max(0, bb[1] - pad), min(W, bb[2] + pad), min(H, bb[3] + pad))
    bw, bh = box[2] - box[0], box[3] - box[1]; s = side / max(bw, bh); size = (max(64, round(bw * s / 8) * 8), max(64, round(bh * s / 8) * 8))
    src = im.convert("RGB").crop(box).resize(size, Image.LANCZOS)
    mk = hard.filter(ImageFilter.MaxFilter(5)).crop(box).resize(size, Image.NEAREST).convert("RGB")
    ctl = control_map(im, hard, mode).crop(box).resize(size, Image.BILINEAR)
    out = cudaeng.inpaint_cn(src, mk, ctl, prompt, neg, seed=seed, progress=progress, status=status, cn_scale=0.8 if mode == "keep" else 0.5)
    full = im.convert("RGB").copy(); full.paste(out.resize((bw, bh), Image.LANCZOS), box[:2])
    return Image.composite(full, im.convert("RGB"), soft)

def changed_score(orig, new, hard):
    """0..1: how much the garment's structure changed (1 - correlation of luminance detail inside the mask).
    A pure recolor keeps the same folds/collar/badge and scores near 0."""
    from PIL import ImageFilter
    m = np.asarray(hard.filter(ImageFilter.MinFilter(5))) > 127
    if m.sum() < 100: return 0.0
    def detail(x):
        g = np.asarray(x.convert("L"), np.float32); b = np.asarray(x.convert("L").filter(ImageFilter.GaussianBlur(4)), np.float32)
        return (g - b)[m]
    a, b = detail(orig), detail(new)
    if a.std() < 1e-3 or b.std() < 1e-3: return 1.0
    return float(max(0.0, min(1.0, 1 - np.corrcoef(a, b)[0, 1])))


def pose_mask(im, mask):
    """Pose / holding changes: redraw only the arms + hands and the space around them (where a new arm position or
    held item can go). Face and hair stay fixed; torso, legs and silhouette outside this area are untouched."""
    from PIL import ImageFilter
    W, H = im.size; arms = np.asarray(mask.convert("L")) > 127; prot = np.zeros((H, W), bool)
    try:
        import bodyparse
        if bodyparse.available():
            hn = bodyparse.net("human-parse"); lab = hn.seg(im)
            arms |= np.isin(lab, hn.ids(["left-arm", "right-arm"])); prot |= np.isin(lab, hn.ids(["face", "hair"]))
    except Exception: logging.exception("pose_mask bodyparse")
    s = _seg()
    if s is not None:
        try:
            p = np.max([s.probs(im, t) for t in ("arm", "hand", "arms")], axis=0)
            arms |= np.asarray(Image.fromarray(((p > max(0.25, p.max() * 0.4)) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) > 127
            q = np.max([s.probs(im, t) for t in ("face", "hair")], axis=0)
            prot |= np.asarray(Image.fromarray(((q > max(0.35, q.max() * 0.5)) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) > 127
        except Exception: logging.exception("pose_mask clipseg")
    try:  # keep the torso / outfit silhouette fixed: the new arm can only go into the arm area and the space beside it
        body = np.asarray(outfit_mask(im, Image.new("L", im.size, 0), "outfit")[0]) > 127
        prot |= body & ~arms
    except Exception: logging.exception("pose_mask body protect")
    g = max(8, int(min(W, H) * 0.06))
    m = np.asarray(grow(Image.fromarray((arms * 255).astype(np.uint8)), g)) > 127
    hard = Image.fromarray(((m & ~prot) * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    return hard, hard.filter(ImageFilter.GaussianBlur(2))


def hair_mask(im, mask):
    """Whole hair, including long anime hair over the shoulders: seed from the human parser + CLIPSeg, then grow through
    connected hair-coloured pixels (saturated or bright, not skin, not the grey wall, not the white outfit). Face cut out."""
    from PIL import ImageFilter
    W, H = im.size; seed = np.asarray(mask.convert("L")) > 127; prot = np.zeros((H, W), bool)
    try:
        import bodyparse
        if bodyparse.available():
            hn = bodyparse.net("human-parse"); lab = hn.seg(im)
            seed |= np.isin(lab, hn.ids(["hair"])); prot |= np.isin(lab, hn.ids(["face"]))
    except Exception: logging.exception("hair_mask bodyparse")
    s = _seg()
    if s is not None:
        try:
            p = np.max([s.probs(im, t) for t in ("hair", "long hair", "colorful hair")], axis=0)
            seed |= np.asarray(Image.fromarray(((p > max(0.3, p.max() * 0.5)) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) > 127
            q = np.max([s.probs(im, t) for t in ("eyes", "mouth", "hat", "nurse cap")], axis=0)
            prot |= np.asarray(Image.fromarray(((q > max(0.4, q.max() * 0.6)) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) > 127
        except Exception: logging.exception("hair_mask clipseg")
    hsv = np.asarray(im.convert("RGB").convert("HSV"), np.float32) / 255.0; hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    skin = (hh > 0.02) & (hh < 0.11) & (ss > 0.25) & (vv < 0.8)
    ok = ((ss > 0.3) | ((vv > 0.85) & (ss > 0.12))) & ~skin & ~prot
    hair = seed & ok
    hi = Image.fromarray((hair * 255).astype(np.uint8)); okI = ok
    for _ in range(80):  # geodesic growth through hair-coloured pixels
        n = (np.asarray(hi.filter(ImageFilter.MaxFilter(7))) > 127) & okI
        if n.sum() == hair.sum(): break
        hair = n; hi = Image.fromarray((hair * 255).astype(np.uint8))
    hi = Image.fromarray((hair * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))  # drop thin outline leaks
    hair = (np.asarray(hi) > 127) | (seed & ok)
    hi = Image.fromarray((hair * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))  # fill strand gaps
    m = np.asarray(grow(hi, max(3, int(min(W, H) * 0.006)))) > 127
    hard = Image.fromarray(((m & ~prot) * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    return hard, hard.filter(ImageFilter.GaussianBlur(1.5))

def hair_short_score(orig, new, hard, col):
    """For 'short <colour> hair': no old hair colours left in the hair mask below the chin, and the top is the colour."""
    m = np.asarray(hard) > 127
    if not m.any(): return 0.0
    ys = np.nonzero(m)[0]; y0, y1 = ys.min(), ys.max(); chin = int(y0 + (y1 - y0) * 0.35)
    hsv_o = np.asarray(orig.convert("RGB").convert("HSV"), np.float32) / 255.0; hsv_n = np.asarray(new.convert("RGB").convert("HSV"), np.float32) / 255.0
    low = m.copy(); low[:chin] = False
    def hairish(h): return ((h[..., 1] > 0.3) | ((h[..., 2] > 0.85) & (h[..., 1] > 0.12)))
    rem = float((hairish(hsv_n) & hairish(hsv_o) & (np.abs(hsv_n[..., 0] - hsv_o[..., 0]) < 0.05))[low].mean()) if low.any() else 0.0
    top = m.copy(); top[chin:] = False
    c = color_score(new, Image.fromarray((top * 255).astype(np.uint8)), col) if col else 1.0
    logging.info("short-hair: remnants %.2f colour-on-top %.2f", rem, c)
    return max(0.0, c - rem * 1.5)

def bg_mask(ai, im, hard):
    """Background target: background = everything outside the person's IS-Net alpha matte, extended 2 px under her edge;
    the result is blended back with that matte so no ring of the old background is left."""
    from PIL import ImageFilter
    try:
        a = ai.remove_bg(im)[0].getchannel("A")
        alpha = np.asarray(a, np.float32) / 255.0
        if (alpha > 0.5).mean() < 0.01: return None
        bg = Image.fromarray(((alpha < 0.5) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))  # slightly under her edge
        soft = Image.fromarray(((1 - alpha) * 255).astype(np.uint8))  # composite weight = matte
        return bg, soft
    except Exception: logging.exception("bg_mask isnet"); return None

