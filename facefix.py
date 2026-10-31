"""Face / eye part masks and an ADetailer-style face & eye detailer for NpuStudio.

* find_part(im, text)  -> tight mask for eyes / left eye / right eye / pupils / eyebrows / mouth / nose / face.
  Faces are found with the anime face cascade (lbpcascade_animeface), OpenCV's realistic face cascades, the human
  parser's face label and CLIPSeg 'face' (in that order, merged). Every part is segmented on an UPSCALED face crop
  (768 px) with the face parser (realistic) and CLIPSeg (anime), and eyes get a second, tighter pass on an upscaled
  eye-region crop, because eyes are only a few pixels at full-image scale.
* fix(im, what="eyes"|"face") -> detail pass: crop the eye/face region with padding, upscale it so the eyes are
  worked on at >= 512-768 px, inpaint with low denoise (0.3-0.45) keeping the detected iris colour, score candidates
  (CLIP 'detailed symmetrical eyes' vs 'deformed', left/right symmetry, iris colour kept, nothing outside changed)
  and paste the best back at native resolution with a feathered edge. Detail only: identity, age, face shape and the
  body are never changed (prompt + negatives + mask limited to the eyes/face).
"""
import os, re, time, logging, threading, urllib.request, numpy as np
from PIL import Image, ImageFilter, ImageOps

APP = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(APP, "models")
CASCADE_URL = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
CASCADE = os.path.join(MD, "lbpcascade_animeface.xml")
WORK = 768          # working resolution of the upscaled crops

PARTS = [  # (regex, key)
    (r"(?:left|her left|his left)\s+(?:eye|iris|pupil)", "left eye"),
    (r"(?:right|her right|his right)\s+(?:eye|iris|pupil)", "right eye"),
    (r"pupils?|iris(?:es)?|irides", "pupils"),
    (r"eye\s*brows?|brows?", "eyebrows"),
    (r"eyes?|eyeballs?", "eyes"),
    (r"mouth|lips?|smile", "mouth"),
    (r"nose", "nose"),
    (r"face|facial features", "face"),
]
def part_key(text):
    s = (text or "").lower()
    for rx, k in PARTS:
        if re.search(r"\b(?:" + rx + r")\b", s): return k
    return None

FIX_RE = r"(?i)\b(?:fix|repair|correct|clean\s*up|improve|enhance|detail|redo|refine|sharpen|restore)\s+(?:the\s+|her\s+|his\s+|their\s+|its\s+)?(eyes?|face|faces|pupils?|iris)\b|\b(?:eyes?|face)\s+(?:look\s+)?(?:deformed|broken|messed up|weird|wrong|bad|ugly)\b"
def fix_what(instr):
    """'fix eyes' / 'repair her face' / 'the eyes look deformed' -> 'eyes' | 'face' | None."""
    m = re.search(FIX_RE, instr or "")
    if not m: return None
    return "face" if re.search(r"(?i)\bfaces?\b", m.group(0)) else "eyes"

# ---------------- face detection ----------------
_cv = {}
def _cascade(name):
    import cv2
    if name in _cv: return _cv[name]
    if not hasattr(cv2, "CascadeClassifier"): _cv[name] = None; return None   # OpenCV 5 moved Haar/LBP cascades out of the main build
    if name == "anime":
        if not os.path.exists(CASCADE):
            os.makedirs(MD, exist_ok=True); tmp = CASCADE + ".part"
            urllib.request.urlretrieve(CASCADE_URL, tmp); os.replace(tmp, CASCADE)
        c = cv2.CascadeClassifier(CASCADE)
    else:
        c = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))
    _cv[name] = c if not c.empty() else None
    return _cv[name]

def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0

def detect_faces(im, max_faces=6):
    """-> list of (x0, y0, x1, y1, source) face boxes, biggest first."""
    W, H = im.size; boxes = []
    try:
        import cv2
        g = np.asarray(im.convert("L")); g = cv2.equalizeHist(g); mn = max(24, int(min(W, H) * 0.04))
        for name, src in (("anime", "anime cascade"), ("haarcascade_frontalface_default.xml", "face cascade"), ("haarcascade_frontalface_alt2.xml", "face cascade")):
            c = _cascade(name)
            if c is None: continue
            for (x, y, w, h) in c.detectMultiScale(g, scaleFactor=1.08, minNeighbors=5, minSize=(mn, mn)):
                boxes.append((int(x), int(y), int(x + w), int(y + h), src))
            if boxes and name == "anime": break   # anime cascade hit: realistic cascades only add false positives on drawings
    except Exception: logging.exception("face cascades")
    if not boxes:
        try:
            import bodyparse
            if bodyparse.available():
                hn = bodyparse.net("human-parse"); lab = hn.seg(im); f = np.isin(lab, hn.ids(["face"]))
                b = _blobs(f, min_area=max(80, W * H * 0.0005))
                boxes += [(x0, y0, x1, y1, "body parser") for (x0, y0, x1, y1) in b]
        except Exception: logging.exception("face via body parser")
    if not boxes:
        try:
            if True:
                p = clip_probs(im, "a face")
                f = np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)) > 110
                boxes += [(x0, y0, x1, y1, "CLIPSeg") for (x0, y0, x1, y1) in _blobs(f, min_area=max(80, W * H * 0.0005))]
        except Exception: logging.exception("face via clipseg")
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])); out = []
    big = (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1]) if boxes else 0
    for b in boxes:
        a = (b[2] - b[0]) * (b[3] - b[1]); ar = (b[2] - b[0]) / max(1, b[3] - b[1])
        if a < big * 0.2 or not (0.4 < ar < 2.5): continue   # parser/CLIPSeg crumbs (ears, hands) are not faces
        if all(_iou(b, o) < 0.3 for o in out): out.append(b)
    return out[:max_faces]

def _blobs(a, min_area=50):
    try:
        from scipy import ndimage as nd
        lab, n = nd.label(a); res = []
        for sl in nd.find_objects(lab):
            if sl is None: continue
            ys, xs = sl
            if (ys.stop - ys.start) * (xs.stop - xs.start) >= min_area: res.append((xs.start, ys.start, xs.stop, ys.stop))
        return res
    except Exception:
        ys, xs = np.nonzero(a)
        return [(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)] if len(xs) else []

def _pad_box(b, f, W, H, square=True):
    x0, y0, x1, y1 = b[:4]; w, h = x1 - x0, y1 - y0
    if square: s = max(w, h); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; w = h = s; x0, y0 = cx - s / 2, cy - s / 2; x1, y1 = x0 + s, y0 + s
    p = max(w, h) * f
    return (int(max(0, x0 - p)), int(max(0, y0 - p)), int(min(W, x1 + p)), int(min(H, y1 + p)))

def _up(crop, side=WORK):
    s = side / max(crop.size); size = (max(64, round(crop.size[0] * s / 8) * 8), max(64, round(crop.size[1] * s / 8) * 8))
    return crop.resize(size, Image.LANCZOS), size

# ---------------- part masks ----------------
CLIP_TXT = {"eyes": ["eyes", "anime eyes", "the eyes of the girl"], "pupils": ["iris and pupil of the eyes", "pupils"],
            "eyebrows": ["eyebrows"], "mouth": ["mouth", "lips"], "nose": ["nose"], "face": ["face"]}
FP = {"eyes": ["l_eye", "r_eye"], "pupils": ["l_eye", "r_eye"], "eyebrows": ["l_brow", "r_brow"], "mouth": ["mouth", "u_lip", "l_lip"],
      "nose": ["nose"], "face": ["skin", "l_eye", "r_eye", "l_brow", "r_brow", "nose", "mouth", "u_lip", "l_lip"]}

_HSEG = {}
def clip_probs(img, text):
    """CLIPSeg probabilities (352x352): the OpenVINO IR when installed, else the same model on CUDA/CPU via transformers."""
    try:
        import aichange
        s = aichange._seg() if os.path.exists(os.path.join(aichange.SEG_DIR, "clipseg.xml")) else None
        if s is not None: return s.probs(img, text)
    except Exception: logging.exception("clipseg ov")
    import torch
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    if "m" not in _HSEG:
        hf = os.path.join(MD, "hf"); dev = "cuda" if torch.cuda.is_available() else "cpu"
        _HSEG["p"] = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=hf)
        _HSEG["m"] = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=hf).to(dev).eval(); _HSEG["d"] = dev
    inp = _HSEG["p"](text=[text], images=[img.convert("RGB")], return_tensors="pt", padding=True).to(_HSEG["d"])
    with torch.no_grad(): lg = _HSEG["m"](**inp).logits
    return torch.sigmoid(lg).reshape(352, 352).float().cpu().numpy()

def seg_mask(im, targets, thresh=0.4):
    """Same rule as aichange.Seg.mask, on whichever CLIPSeg is available -> (L mask, max prob)."""
    p = np.max([clip_probs(im, t) for t in targets], axis=0)
    th = max(0.12 if p.max() < 0.5 else 0.25, min(thresh, p.max() * 0.5))
    m = Image.fromarray(((p > th) * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)
    return m.filter(ImageFilter.GaussianBlur(2)).point(lambda v: 255 if v > 127 else 0), float(p.max())

def _atext(a):
    if isinstance(a, (tuple, list)): a = a[0] if a else ""
    return a if isinstance(a, str) else ("" if a is None else str(a))

def find_mask(im, tgt, instr="", analysis=""):
    """The one target finder for AI Change (desktop app, laptop app, phone API): face parts on upscaled face crops,
    then the body/clothing parser, then CLIPSeg. -> (L mask, confidence, how)."""
    analysis = _atext(analysis); tgt = _atext(tgt) if isinstance(tgt, (tuple, list)) else tgt
    mm = multi_mask(im, instr, analysis) if len(body_parts(instr)) >= 2 else None
    if mm is not None: return mm
    t = time.time(); r = None
    if part_key(tgt or "") or part_key(instr or ""):
        try: r = find_part(im, tgt if part_key(tgt or "") else instr)
        except Exception: logging.exception("face part find")
        if r is not None and r[1] > 0: return r
    if re.search(r"(?i)\b(hands?|fingers?|palms?)\b", tgt or ""):   # the parser has arms only: keep the hand ends
        try:
            import bodyparse
            arms = None
            if bodyparse.available():
                bb = bodyparse.mask(im, "arms")
                if bb is not None and bb[1] > 0: arms = np.asarray(bb[0]) > 127
            W, H = im.size; m = np.zeros((H, W), bool); pmax = 0.0
            boxes = [(x0, y0, x1, y1) for (x0, y0, x1, y1) in _blobs(_grow_b(arms, 0.03), W * H * 0.002)] if arms is not None and arms.any() else [(0, 0, W, H)]
            for bx in boxes:   # hand vs arm decided on an upscaled crop of each arm, not the 352px whole image
                bx = _pad_box(bx, 0.1, W, H, square=False); c = im.convert("RGB").crop(bx)
                hp = clip_probs(c, "a hand with fingers"); ap = clip_probs(c, "an arm, forearm and sleeve")
                pmax = max(pmax, float(hp.max()))
                sel = (hp > max(0.25, float(hp.max()) * 0.5)) & (hp > ap)
                sub = np.asarray(Image.fromarray((sel * 255).astype(np.uint8)).resize(c.size, Image.BILINEAR)) > 127
                m[bx[1]:bx[3], bx[0]:bx[2]] |= sub
            if arms is not None: m &= _grow_b(arms, 0.01)
            if m.any(): return Image.fromarray((m * 255).astype(np.uint8)), pmax, f"CLIPSeg hand-vs-arm on upscaled arm crops{' within the parsed arms' if arms is not None else ''}, {time.time() - t:.2f}s"
        except Exception: logging.exception("hands")
    try:
        import bodyparse
        if bodyparse.available():
            b = bodyparse.mask(im, tgt)
            if b is not None and b[1] > 0 and b[0].getbbox(): return b
    except Exception: logging.exception("bodyparse find")
    import aichange
    tg = aichange.targets_for(tgt, analysis); m, pm = seg_mask(im, tg)
    return m, pm, f"CLIPSeg ({', '.join(tg[:3])}), {time.time() - t:.2f}s"

def _clip_mask(img, texts, th=None):
    try: p = np.max([clip_probs(img, t) for t in texts], axis=0)
    except Exception: logging.exception("clipseg"); return None, 0.0
    pm = float(p.max())
    t = th if th is not None else max(0.2, pm * 0.45)
    m = np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) > t * 255
    return m, pm

def _face_part(face_img, key):
    """Part mask (bool, face_img size) inside one upscaled face crop -> (mask, conf, how)."""
    k = "eyes" if key in ("left eye", "right eye") else key
    fpm = None; how = ""
    try:
        import bodyparse
        if bodyparse.available():
            fn = bodyparse.net("face-parse"); fl = fn.seg(face_img); fpm = np.isin(fl, fn.ids(FP[k]))
            if k != "face":
                n = len(_blobs(fpm, min_area=face_img.size[0] * face_img.size[1] * 0.0004)); a = fpm.mean()
                if not (n >= 1 and 0.0008 < a < 0.12): fpm = None   # face parser lost on drawings -> CLIPSeg
            elif fpm.mean() < 0.08: fpm = None
            if fpm is not None: how = "face parser"
    except Exception: logging.exception("face parser part"); fpm = None
    cm, pm = _clip_mask(face_img, CLIP_TXT[k])
    if fpm is not None and cm is not None and k in ("eyes", "pupils"):
        m = fpm | (cm & _grow_b(fpm, 0.04)); conf = 0.95; how = "face parser + CLIPSeg"   # parser shape, CLIPSeg fills missed lashes
    elif fpm is not None: m, conf = fpm, 0.95
    elif cm is not None: m, conf, how = cm, pm, "CLIPSeg"
    else: return None, 0.0, "no segmenter"
    if k in ("eyes", "pupils"):
        m = _refine_eyes(face_img, m, "eyes")
        if k == "pupils": m = _iris(face_img, m); how += " + iris colour"
    elif k in ("mouth", "eyebrows") and (m is None or m.mean() < 0.0004 or how == "CLIPSeg" or conf < 0.3):
        eyes, _, _ = _face_part(face_img, "eyes")
        alt = _mouth(face_img, eyes) if k == "mouth" else _brows(face_img, eyes)
        if alt is not None and alt.any(): m, conf, how = alt, 0.7, f"{how} + {k} search on the upscaled face"
    elif k == "face":
        hp = _face_from_human(face_img)
        if hp is not None and (len(_blobs(m, face_img.size[0] * face_img.size[1] * 0.002)) > 2 or m.mean() < hp.mean() * 0.5):
            m, how = hp | m, "face parser + human parser face (filled)"
        m = _fill(m)
    return m, conf, how

def _fill(m):
    try:
        from scipy import ndimage as nd
        return nd.binary_fill_holes(nd.binary_closing(m, iterations=3))
    except Exception: return m

def _face_from_human(face_img):
    try:
        import bodyparse
        hn = bodyparse.net("human-parse"); lab = hn.seg(face_img); f = np.isin(lab, hn.ids(["face"]))
        bl = sorted(_blobs(f, 50), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
        if not bl: return None
        b = bl[0]; keep = np.zeros_like(f); keep[b[1]:b[3], b[0]:b[2]] = f[b[1]:b[3], b[0]:b[2]]
        return _fill(keep)
    except Exception: logging.exception("human face"); return None

def _eye_boxes(eyes, W, H):
    if eyes is None: return []
    return sorted(sorted(_blobs(eyes, W * H * 0.0003), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:2], key=lambda b: b[0])

def _iris(face_img, eyes):
    """Iris + pupil = the coloured/dark core of each eye: eye mask minus white sclera and skin, largest blob, holes filled."""
    rgb = np.asarray(face_img.convert("RGB"), np.float32); hsv = np.asarray(face_img.convert("HSV"), np.float32) / 255.0
    core = eyes & ~((hsv[..., 2] > 0.8) & (hsv[..., 1] < 0.2)) & ~_skin_like(rgb, face_img)
    out = np.zeros_like(eyes)
    for b in _eye_boxes(eyes, *face_img.size):
        sub = core[b[1]:b[3], b[0]:b[2]]
        h = b[3] - b[1]; sub[: max(1, int(h * 0.12))] = False      # drop the lash line on top
        bl = sorted(_blobs(sub, 4), key=lambda x: -(x[2] - x[0]) * (x[3] - x[1]))
        if bl:
            x = bl[0]; one = np.zeros_like(sub); one[x[1]:x[3], x[0]:x[2]] = sub[x[1]:x[3], x[0]:x[2]]
            out[b[1]:b[3], b[0]:b[2]] |= _fill(one)
    return out if out.any() else eyes

def _not_skin_band(face_img, box):
    W, H = face_img.size; x0, y0, x1, y1 = [int(v) for v in box]; x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0: return None, None
    rgb = np.asarray(face_img.convert("RGB"), np.float32)[y0:y1, x0:x1]
    return ~_skin_like(rgb, face_img), (x0, y0, x1, y1)

def _brows(face_img, eyes):
    """Anime brows: thin non-skin strokes in a band just above each eye (bangs covering them are left alone)."""
    W, H = face_img.size; out = np.zeros((H, W), bool)
    for b in _eye_boxes(eyes, W, H):
        eh, ew = b[3] - b[1], b[2] - b[0]
        sub, bx = _not_skin_band(face_img, (b[0] - ew * 0.1, b[1] - eh * 1.1, b[2] + ew * 0.1, b[1] - eh * 0.15))
        if sub is None: continue
        rows = sub.mean(1); good = rows < 0.6        # a full row of non-skin = bangs, not a brow
        sub = sub & good[:, None]
        if sub.mean() > 0.02: out[bx[1]:bx[3], bx[0]:bx[2]] |= sub
    return out

def _mouth(face_img, eyes):
    """Mouth on an upscaled lower-face crop: CLIPSeg 'mouth/lips' there, else the non-skin stroke below the nose line."""
    W, H = face_img.size; eb = _eye_boxes(eyes, W, H)
    if len(eb) == 2:
        cx = (eb[0][2] + eb[1][0]) / 2; ed = eb[1][0] - eb[0][2] + (eb[0][2] - eb[0][0]); ey = max(eb[0][3], eb[1][3])
        box = (cx - ed * 0.45, ey + ed * 0.45, cx + ed * 0.45, ey + ed * 1.1)
    else: box = (W * 0.35, H * 0.6, W * 0.65, H * 0.85)
    box = tuple(int(v) for v in (max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8: return None
    c = face_img.crop(box); cu, _ = _up(c, 352); m, pm = _clip_mask(cu, CLIP_TXT["mouth"])
    out = np.zeros((H, W), bool)
    sub = None
    if m is not None and pm > 0.2:
        sub = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(c.size, Image.BILINEAR)) > 127
    if sub is None or not (0.002 < sub.mean() < 0.5):
        sub, _ = _not_skin_band(face_img, box)
    fm = _face_from_human(face_img)
    if sub is not None and fm is not None: sub = sub & fm[box[1]:box[3], box[0]:box[2]]
    if sub is not None and 0 < sub.mean() < 0.15: out[box[1]:box[3], box[0]:box[2]] = sub   # a big blob = not a mouth
    return out

def _grow_b(m, f):
    r = max(1, int(m.shape[1] * f)) | 1
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(min(r * 2 + 1, 61)))) > 127

def _refine_eyes(face_img, m, k):
    """Second pass on an upscaled crop around each eye: tighter edges, then pupils = iris core (no sclera/skin)."""
    W, H = face_img.size; out = np.zeros_like(m)
    for (x0, y0, x1, y1) in sorted(_blobs(m, min_area=W * H * 0.0003), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:2]:
        bx = _pad_box((x0, y0, x1, y1), 0.35, W, H, square=False); c = face_img.crop(bx); cu, sz = _up(c, 512)
        em, pm = _clip_mask(cu, CLIP_TXT["eyes"] if k == "eyes" else CLIP_TXT["pupils"])
        old = m[bx[1]:bx[3], bx[0]:bx[2]]
        if em is None or pm < 0.25: sub = old
        else:
            sub = np.asarray(Image.fromarray((em * 255).astype(np.uint8)).resize(c.size, Image.BILINEAR)) > 127
            sub = sub & _grow_b(old, 0.06)
        rgb0 = np.asarray(c.convert("RGB"), np.float32)
        ns = ~_skin_like(rgb0, face_img, 40)        # anime eyes = lashes, sclera, iris: never skin or blush
        from scipy import ndimage as nd
        sub = nd.binary_fill_holes(nd.binary_closing(sub & ns, iterations=2)) & sub
        if k == "pupils":
            hsv = np.asarray(c.convert("HSV"), np.float32) / 255.0; rgb = np.asarray(c.convert("RGB"), np.float32)
            sclera = (hsv[..., 2] > 0.82) & (hsv[..., 1] < 0.18); skin = _skin_like(rgb, face_img)
            sub = sub & ~sclera & ~skin
        out[bx[1]:bx[3], bx[0]:bx[2]] |= sub
    return out if out.any() else m

def _skin_like(rgb, face_img, tol=28):
    W, H = face_img.size
    ref = np.asarray(face_img.convert("RGB").crop((int(W * .35), int(H * .62), int(W * .65), int(H * .72))), np.float32).reshape(-1, 3)
    c = np.median(ref, 0); d = rgb - c
    blush = (rgb[..., 0] > 150) & (rgb[..., 0] > rgb[..., 1] + 25) & (rgb[..., 1] > 90) & (rgb[..., 2] > rgb[..., 1] - 15)   # pink cheeks count as skin
    return (np.sqrt((d ** 2).sum(-1)) < tol) | blush

def find_part(im, text):
    """-> (L mask at image size, confidence, how) or None when `text` isn't a face part."""
    key = part_key(text)
    if not key: return None
    t = time.time(); W, H = im.size; faces = detect_faces(im)
    if not faces: return Image.new("L", im.size, 0), 0.0, "no face found"
    full = np.zeros((H, W), bool); conf = 0.0; hows = set()
    for f in faces:
        box = _pad_box(f, 0.25, W, H); crop = im.convert("RGB").crop(box); cu, sz = _up(crop)
        m, c, how = _face_part(cu, key)
        if m is None: continue
        if key in ("left eye", "right eye"):   # the subject's left eye is on the image's right
            bl = sorted(_blobs(m, min_area=sz[0] * sz[1] * 0.0003), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:2]
            if len(bl) == 2:
                bl.sort(key=lambda b: b[0]); keep = bl[1] if key == "left eye" else bl[0]
                mm = np.zeros_like(m); mm[keep[1]:keep[3], keep[0]:keep[2]] = m[keep[1]:keep[3], keep[0]:keep[2]]; m = mm
        mb = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(crop.size, Image.BILINEAR)) > 110
        full[box[1]:box[3], box[0]:box[2]] |= mb; conf = max(conf, c); hows.add(f"{f[4]} + {how}")
    mk = Image.fromarray((full * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3))
    return mk, (conf if mk.getbbox() else 0.0), f"{key}: {', '.join(sorted(hows)) or 'nothing'} on {len(faces)} face(s), upscaled crops, {time.time() - t:.2f}s"

# ---------------- colour / scoring helpers ----------------
def iris_color(crop, eye_mask):
    """Dominant iris colour name inside the eye mask (sclera, lashes and skin excluded)."""
    try:
        import aichange
        rgb = np.asarray(crop.convert("RGB"), np.float32); hsv = np.asarray(crop.convert("HSV"), np.float32) / 255.0
        sel = eye_mask & (hsv[..., 2] > 0.18) & ~((hsv[..., 2] > 0.82) & (hsv[..., 1] < 0.18)) & (hsv[..., 1] > 0.2)
        if sel.sum() < 15: return None, None
        hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        peach = (hh * 360 < 45) & (vv > 0.7) & (ss < 0.5)            # skin / blush inside the eye box
        strong = sel & (ss > 0.3) & ~peach
        if strong.sum() < 15:
            c = np.median(rgb[sel], 0); v = float(np.median(vv[sel]))
            return ("grey" if v > 0.35 else "dark"), c
        hist, edges = np.histogram(hh[strong], bins=18, range=(0, 1), weights=ss[strong] ** 2)
        k = int(np.argmax(hist)); inb = strong & (hh >= edges[k]) & (hh < edges[k + 1])
        c = np.median(rgb[inb], 0); h = float(np.median(hh[inb])); v = float(np.median(vv[inb]))
        hd = h * 360
        if v < 0.3: name = "dark brown"
        elif hd < 15 or hd >= 340: name = "red" if v > 0.75 else ("reddish brown" if v < 0.5 else "reddish amber")
        elif hd < 40: name = "amber" if v > 0.6 else "brown"
        elif hd < 70: name = "golden"
        elif hd < 160: name = "green"
        elif hd < 200: name = "teal"
        elif hd < 255: name = "blue"
        elif hd < 290: name = "purple"
        else: name = "pink"
        return name, c
    except Exception: logging.exception("iris colour"); return None, None

_CLIP = {}
def plaus(img):
    """CLIP probability that an eye-region crop shows good eyes rather than deformed ones (0..1)."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        if "m" not in _CLIP:
            hf = os.path.join(MD, "hf"); dev = "cuda" if torch.cuda.is_available() else "cpu"
            _CLIP["m"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=hf).to(dev).eval()
            _CLIP["p"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=hf); _CLIP["d"] = dev
        txt = ["beautiful detailed symmetrical eyes with clear round pupils", "deformed asymmetrical eyes", "blurry melted smudged eyes",
               "mismatched eyes with distorted pupils"]
        inp = _CLIP["p"](text=txt, images=img.convert("RGB"), return_tensors="pt", padding=True).to(_CLIP["d"])
        with torch.no_grad(): pr = _CLIP["m"](**inp).logits_per_image.softmax(-1)[0]
        return float(pr[0])
    except Exception: logging.exception("eye plausibility"); return None

def symmetry(img, eye_mask):
    """0..1 similarity of the two eyes (one mirrored) - edges + colour, scale-normalised."""
    bl = sorted(_blobs(eye_mask, min_area=eye_mask.size * 0.0005), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:2]
    if len(bl) < 2: return None
    bl.sort(key=lambda b: b[0]); g = img.convert("RGB")
    def patch(b, flip):
        w, h = b[2] - b[0], b[3] - b[1]; p = 0.15
        c = g.crop((int(b[0] - w * p), int(b[1] - h * p), int(b[2] + w * p), int(b[3] + h * p))).resize((64, 48), Image.LANCZOS)
        return ImageOps.mirror(c) if flip else c
    a, b = patch(bl[0], False), patch(bl[1], True)
    ea = np.asarray(a.convert("L").filter(ImageFilter.FIND_EDGES), np.float32).ravel(); eb = np.asarray(b.convert("L").filter(ImageFilter.FIND_EDGES), np.float32).ravel()
    ec = np.corrcoef(ea, eb)[0, 1] if ea.std() > 1 and eb.std() > 1 else 0.0
    ca, cb = np.asarray(a, np.float32), np.asarray(b, np.float32); cc = 1 - min(1.0, np.abs(ca.mean((0, 1)) - cb.mean((0, 1))).mean() / 60)
    sa = (bl[0][2] - bl[0][0]) * (bl[0][3] - bl[0][1]); sb = (bl[1][2] - bl[1][0]) * (bl[1][3] - bl[1][1]); sz = min(sa, sb) / max(sa, sb)
    return float(max(0.0, 0.45 * max(0.0, ec) + 0.3 * cc + 0.25 * sz))

# ---------------- the detailer ----------------
NEG = ("deformed eyes, asymmetrical eyes, cross-eyed, lazy eye, extra pupils, missing pupils, melted, blurry, smudged, "
       "heterochromia, different eye colors, mismatched eyes, distorted face, bad anatomy, extra eyes, "
       "child, childlike, younger, baby face, loli, different person, makeup change, text, watermark")

def _style(im, analysis=""):
    analysis = _atext(analysis)
    try:
        import aichange
        if re.search(aichange.ILLU, analysis or "", re.I): return "anime"
    except Exception: pass
    a = np.asarray(im.convert("RGB").resize((128, 128)), np.float32); q = (a // 32).reshape(-1, 3)
    return "anime" if len(np.unique(q, axis=0)) < 90 else "photo"   # flat cel shading -> few colours

def _prompt(what, col, style):
    if what == "eyes":
        c = f"{col} iris, " if col else ""
        p = f"detailed symmetrical eyes, {c}matching iris color in both eyes, clear round pupils, crisp eyelashes, sharp clean eye outline, catchlight"
    else:
        p = "detailed face, symmetrical eyes, clear pupils, clean facial features, same person, same expression"
    return (f"anime illustration, clean lineart, {p}, high quality" if style == "anime" else f"photo, {p}, natural skin texture, high quality"), NEG

def _control(img):
    from PIL import ImageFilter as F
    g = np.asarray(img.convert("L"), np.float32); bl = np.asarray(img.convert("L").filter(F.GaussianBlur(3)), np.float32)
    return Image.fromarray((((g < bl - 14) | (g < 50)) * 255).astype(np.uint8)).convert("RGB")

PREVIEW = None   # set to {} by facetest.py to get the best candidate back

def fix(im, what="eyes", analysis="", n=4, progress=None, status=None, faces=None, strength=None, min_gain=0.0):
    """ADetailer-style detail pass on every detected face. -> (image, report str, ok bool)."""
    import cudaeng
    if not cudaeng.available(): return im, "NOT DONE - face/eye fix needs NVIDIA (CUDA) here or the desktop PC", False
    im = im.convert("RGB"); W, H = im.size; faces = faces if faces is not None else detect_faces(im)
    if not faces: return im, "NOT DONE - no face found (brush the eyes and use a normal change instead)", False
    out = im.copy(); notes = []; ok_any = False; style = _style(im, analysis)
    for fi, f in enumerate(faces):
        fbox = _pad_box(f, 0.25, W, H); fcrop = im.crop(fbox); fu, fsz = _up(fcrop)
        em, _, how = _face_part(fu, "eyes")
        if em is None or not em.any():
            notes.append(f"face {fi + 1}: eyes not found"); continue
        if what == "eyes":
            region = _grow_b(em, 0.035)   # eyes + lashes + a little lid
        else:
            fm, _, _ = _face_part(fu, "face"); region = (fm if fm is not None else em) | _grow_b(em, 0.03)
        # working crop: around the region, upscaled so the eyes are worked on at high resolution
        rb = _blobs(region, 1); x0 = min(b[0] for b in rb); y0 = min(b[1] for b in rb); x1 = max(b[2] for b in rb); y1 = max(b[3] for b in rb)
        sx, sy = fcrop.size[0] / fsz[0], fcrop.size[1] / fsz[1]
        rbox = (fbox[0] + x0 * sx, fbox[1] + y0 * sy, fbox[0] + x1 * sx, fbox[1] + y1 * sy)
        wbox = _pad_box(rbox, 0.3 if what == "eyes" else 0.12, W, H, square=False)
        native = im.crop(wbox); work, wsz = _up(native, WORK)
        rm = Image.fromarray((region * 255).astype(np.uint8)).resize(fcrop.size, Image.BILINEAR)
        rfull = Image.new("L", (W, H), 0); rfull.paste(rm, fbox[:2])
        rwork = rfull.crop(wbox).resize(wsz, Image.BILINEAR).point(lambda v: 255 if v > 100 else 0)
        ewfull = Image.new("L", (W, H), 0); ewfull.paste(Image.fromarray((em * 255).astype(np.uint8)).resize(fcrop.size, Image.BILINEAR), fbox[:2])
        ework = np.asarray(ewfull.crop(wbox).resize(wsz, Image.BILINEAR)) > 110
        col, crgb = iris_color(work, ework)
        prompt, neg = _prompt(what, col, style); ctl = _control(work)
        strengths = [strength] * n if strength else [0.38, 0.45, 0.32, 0.42, 0.36, 0.4][:n]
        mk = rwork.filter(ImageFilter.MaxFilter(5)).convert("RGB")
        base_p = plaus(work); base_s = symmetry(work, ework)
        if status: status(f"Fixing {what} on face {fi + 1}/{len(faces)} ({how}; iris {col or '?'}; working at {wsz[0]}x{wsz[1]} from {native.size[0]}x{native.size[1]})...")
        def make(k):
            return cudaeng.inpaint_cn(work, mk, ctl, prompt, neg, steps=30, guidance=7.0, cn_scale=0.45, seed=101 + fi * 31 + k * 7,
                                      strength=strengths[k], progress=progress)
        cands = {}
        try:
            import remote
            if remote.both(): cands = {k: r for k, r in enumerate(remote.run_split(make, n)) if r is not None}
        except Exception: logging.exception("both machines face fix")
        best = None
        for k in range(n):
            try:
                r = cands[k] if k in cands else make(k)
            except Exception: logging.exception("face fix candidate"); continue
            r = Image.composite(r.convert("RGB"), work, rwork.filter(ImageFilter.GaussianBlur(3)))
            p = plaus(r); s = symmetry(r, ework); c2, rgb2 = iris_color(r, ework)
            ck = 1.0 if crgb is None or rgb2 is None else max(0.0, 1 - float(np.abs(crgb - rgb2).mean()) / 80)
            sc = 0.55 * (p if p is not None else 0.5) + 0.25 * (s if s is not None else 0.5) + 0.2 * ck
            logging.info("facefix face %d cand %d strength %.2f plaus %s sym %s colour %.2f (%s) -> %.3f", fi, k, strengths[k], p, s, ck, c2, sc)
            if ck < 0.45: continue   # iris colour changed: reject (detail only)
            if best is None or sc > best[0]: best = (sc, r, p, s)
        base = 0.55 * (base_p if base_p is not None else 0.5) + 0.25 * (base_s if base_s is not None else 0.5) + 0.2
        if best is None:
            notes.append(f"face {fi + 1}: every version changed the eye colour or failed - kept"); continue
        if PREVIEW is not None:   # test harness: keep the best candidate even when it is rejected
            pv = PREVIEW.setdefault("img", im.copy()); pv.paste(Image.composite(best[1].resize(native.size, Image.LANCZOS), native, rfull.crop(wbox).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(max(1.5, min(native.size) * 0.012)))), wbox[:2])
        if best[0] < base + min_gain:   # only paste when the detailer scored at least as well as the original
            notes.append(f"face {fi + 1}: original already scored better ({base:.2f} vs {best[0]:.2f}) - kept"); continue
        # paste back at native resolution with a feathered edge (only the eye/face region changes)
        back = best[1].resize(native.size, Image.LANCZOS)
        fr = max(1.5, min(native.size) * 0.012)
        feather = rfull.crop(wbox).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(fr))
        out.paste(Image.composite(back, native, feather), wbox[:2]); ok_any = True
        notes.append(f"face {fi + 1}: {what} fixed (score {base:.2f} -> {best[0]:.2f}, iris {col or '?'}, worked at {wsz[0]}x{wsz[1]})")
    head = "Method: face/eye detailer (" + ("; ".join(notes)) + ")"
    return (out if ok_any else im), (head if ok_any else "NOT DONE - " + "; ".join(notes)), ok_any

# ---------------- body-part detail repair + reshaping guard ----------------
BODY_RE = r"(?i)\b(arms?|legs?|hands?|fingers?|feet|foot|toes?)\b"
BODY_KEY = {"arm": "arms", "arms": "arms", "leg": "legs", "legs": "legs", "hand": "hands", "hands": "hands", "finger": "hands",
            "fingers": "hands", "foot": "feet", "feet": "feet", "toe": "feet", "toes": "feet"}
FIXVERB = r"(?i)\b(?:fix|repair|correct|clean\s*up|improve|redo|refine|restore)\b"
SIZE_WORDS = (r"(?i)\b(bigger|biggest|larger|largest|smaller|smallest|thinner|thicker|slimmer|skinnier|fatter|curvier|curvy|longer|shorter|"
              r"wider|narrower|taller|enlarge[ds]?|enlarging|shrink|shrunk|increase[ds]?|reduce[ds]?|huge|giant|tiny|busty|voluptuous|muscular)\b")
RESHAPE_PARTS = (r"(?i)\b(breasts?|boobs?|bust|chest|tits?|cleavage|butt|buttocks?|ass|hips?|waist|thighs?|belly|stomach|body|figure|"
                 r"curves|legs?|arms?|hands?|feet|shoulders?|neck|head|face|eyes?|lips|mouth|nose)\b")

def reshape_block(instr):
    """Body reshaping is not supported: size/shape words aimed at a body part (or the body) -> refusal text, else None."""
    s = instr or ""
    if re.search(r"(?i)\b(nude|naked|topless|bottomless|undress\w*|see[- ]?through|lingerie|underwear|bra|panties|bikini)\b", s) or \
       re.search(r"(?i)\b(transparent|sheer|translucent)\b.*\b(cloth\w*|outfit|dress|shirt|top|skirt|pants|jeans|shorts|blouse|jacket|coat|uniform|swimsuit|garment|apparel)\b|\b(cloth\w*|outfit|dress|shirt|top|skirt|pants|jeans|shorts|blouse|jacket|coat|uniform|swimsuit|garment|apparel)\b.*\b(transparent|sheer|translucent)\b", s) or \
       re.search(r"(?i)\b(remove|erase|delete|take off|strip)\s+(?:all\s+|her\s+|his\s+|their\s+|the\s+|of\s+)*(cloth\w*|outfit|dress|shirt|top|skirt|pants|jeans|shorts|blouse|jacket|coat|uniform|swimsuit|garment|apparel)\b", s):
        return ("Removing clothing or making it see-through isn't supported. You can change an outfit to a different "
                "(fully clothed) one, e.g. 'change the dress to a red sweater dress'. Nothing was changed.")
    if re.search(SIZE_WORDS, s) and re.search(RESHAPE_PARTS, s):
        return ("Body reshaping isn't supported - NpuStudio keeps the silhouette and proportions as they are. "
                "You can change clothes, colours, the background, or use 'fix arms/legs/hands' / 'fix eyes' for detail repair. Nothing was changed.")
    return None

def strip_size_words(p):
    """Hard guard for prompts: never emit size/shape words (whatever the analysis or a stale prompt contained)."""
    p = re.sub(SIZE_WORDS, "", p or ""); p = re.sub(r"(?i)\b(a|an)\s*(?=,|$)", "", p)
    return re.sub(r"\s*,\s*(,\s*)+", ", ", re.sub(r"\s{2,}", " ", p)).strip(" ,")

def body_parts(text):
    """All limb parts named in the text, in order: 'fix legs and arms' -> ['legs', 'arms']."""
    out = []
    for m in re.finditer(BODY_RE, text or ""):
        k = BODY_KEY[m.group(1).lower()]
        if k not in out: out.append(k)
    return out

def body_fix(instr):
    """'fix legs and arms' / 'repair her hands' -> ['legs', 'arms'] (detail repair, not a redraw), else None."""
    if not re.search(FIXVERB, instr or ""): return None
    if re.search(r"(?i)\b(eyes?|face|faces|pupils?|iris)\b", instr or "") and not body_parts(instr): return None
    p = body_parts(instr); return p or None

def multi_mask(im, instr, analysis=""):
    """Union of the masks of every limb part named in the instruction -> (mask, conf, how) or None if fewer than 2 parts."""
    parts = body_parts(instr)
    if len(parts) < 2: return None
    acc = None; confs = []; hows = []
    for p in parts:
        try: m, c, h = find_mask(im, p, p, analysis)
        except Exception: logging.exception("multi part %s", p); continue
        ok = m is not None and m.getbbox() is not None and c > 0.2
        hows.append(f"{p}: {'found' if ok else 'not found'} ({c:.2f})")
        if ok:
            confs.append(c); a = np.asarray(m.convert("L")) > 127; acc = a if acc is None else (acc | a)
    if acc is None: return Image.new("L", im.size, 0), 0.0, "; ".join(hows)
    return Image.fromarray((acc * 255).astype(np.uint8)), min(confs), "parts - " + "; ".join(hows)

BODY_NEG = ("deformed, bad anatomy, extra limbs, extra arms, extra legs, missing limbs, extra fingers, missing fingers, fused fingers, "
            "mutated hands, malformed hands, broken joints, twisted limbs, different pose, blurry, smudged, lowres, "
            "nsfw, nude, child, younger, bigger, smaller, different body shape")

def _person(img):
    try:
        import bodyparse
        if bodyparse.available():
            hn = bodyparse.net("human-parse"); lab = hn.seg(img); return ~np.isin(lab, hn.ids(["background"]))
    except Exception: logging.exception("person mask")
    return None

def fix_body(im, mask, parts, analysis="", n=3, progress=None, status=None):
    """Detail repair of limbs inside `mask`: upscaled crop, low denoise (0.3-0.42), lineart ControlNet keeps the outline and pose;
    a candidate is pasted only when the person's silhouette is unchanged (IoU >= 0.97) and CLIP rates it at least as well drawn."""
    import cudaeng
    W, H = im.size; mk = mask.convert("L").point(lambda v: 255 if v > 100 else 0)
    bb = mk.getbbox()
    if not bb: return im, "NOT DONE - couldn't find the " + " and ".join(parts), False
    wbox = _pad_box(bb, 0.08, W, H, square=False); native = im.convert("RGB").crop(wbox)
    sc = 768 / max(native.size); ws = (max(8, int(native.size[0] * sc) // 8 * 8), max(8, int(native.size[1] * sc) // 8 * 8))
    work = native.resize(ws, Image.LANCZOS); mw = mk.crop(wbox).resize(ws, Image.BILINEAR).filter(ImageFilter.MaxFilter(5))
    ctl = _control(work); style = _style(im, analysis)
    names = " and ".join(parts)
    prompt = strip_size_words(f"well-drawn {names}, clean anatomy, correct natural joints" + (", correct five fingers" if "hands" in parts else "") +
                              ", same pose, same proportions, " + ("anime illustration, clean line art, flat cel shading" if style == "anime" else "photo, natural detail") + ", high quality")
    txt = [f"well-drawn {names} with correct anatomy", f"deformed {names}", f"extra or missing limbs", f"mutated malformed {names}"]
    def score(img):
        try:
            import torch
            plaus(work)   # loads CLIP
            inp = _CLIP["p"](text=txt, images=img, return_tensors="pt", padding=True).to(_CLIP["d"])
            with torch.no_grad(): return float(_CLIP["m"](**inp).logits_per_image.softmax(-1)[0][0])
        except Exception: logging.exception("body score"); return None
    base = score(work); pb = _person(work); best = None
    for k, st_ in enumerate([0.35, 0.42, 0.3][:n]):
        if status: status(f"Repairing {names} ({k + 1}/{n}, strength {st_:.2f})...")
        try:
            r = cudaeng.inpaint_cn(work, mw.convert("RGB"), ctl, prompt, BODY_NEG, steps=30, guidance=7.0, cn_scale=0.7, seed=211 + k * 13, progress=progress, strength=st_)
        except Exception: logging.exception("body fix candidate"); continue
        r = Image.composite(r.convert("RGB"), work, mw.filter(ImageFilter.GaussianBlur(3)))
        iou = 1.0; pr = _person(r)
        if pb is not None and pr is not None: iou = float((pb & pr).sum()) / max(1, float((pb | pr).sum()))
        s = score(r); logging.info("body fix cand %d strength %.2f silhouette IoU %.3f score %s (base %s)", k, st_, iou, s, base)
        if iou < 0.97: continue
        if best is None or (s or 0) > best[0]: best = (s or 0, r, iou, st_)
    if best is None: return im, f"NOT DONE - every version changed the silhouette; original kept ({names})", False
    if base is not None and best[0] < base: return im, f"NOT DONE - original {names} already scored better ({base:.2f} vs {best[0]:.2f}) - kept", False
    back = best[1].resize(native.size, Image.LANCZOS); feather = mk.crop(wbox).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(2))
    out = im.copy(); out.paste(Image.composite(back, native, feather) if im.mode == "RGB" else Image.composite(back, native, feather).convert(im.mode), wbox[:2])
    return out, (f"Method: limb detail repair ({names}; strength {best[3]:.2f}, silhouette IoU {best[2]:.3f}, score "
                 f"{(base or 0):.2f} -> {best[0]:.2f}, worked at {ws[0]}x{ws[1]})"), True
