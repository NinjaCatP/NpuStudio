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
        mk = redraw_mask(im, mask, tgt)
        r, d = ai.change(im, mk, p, steps=40, strength=1.0, guidance=7.5, progress=progress, status=status, negative=neg, grow=15)
        note = "" if m == "Redraw" else " (recolor needs NVIDIA)"
        return r, d, "Method: redraw (inpaint)" + note, p

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
def parse(instr):
    """-> (target or None, replacement, kind) where kind in remove/replace/global."""
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
    neg = ", ".join([x for x in old if x] + ["blurry, deformed, low quality, extra limbs, bad anatomy"])
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
