"""Body-part / clothing masks: SegFormer human parser (mattmdjaga/segformer_b2_clothes) + face parser
(jonathandinu/face-parsing) in OpenVINO, on the device picker's choice."""
import os, json, time, logging, re, numpy as np
from PIL import Image, ImageFilter
APP = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(APP, "models"); CACHE = os.path.join(MD, "cache")
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
# human-parser label names (from labels.json, lower-cased)
CLOTHES = ["hat", "upper-clothes", "skirt", "pants", "dress", "belt", "left-shoe", "right-shoe", "bag", "scarf", "sunglasses"]
HUMAN = {
    "hair": ["hair"], "hat": ["hat"], "cap": ["hat"], "helmet": ["hat"], "sunglasses": ["sunglasses"], "glasses": ["sunglasses"],
    "shirt": ["upper-clothes"], "top": ["upper-clothes"], "jacket": ["upper-clothes"], "coat": ["upper-clothes"],
    "sweater": ["upper-clothes"], "hoodie": ["upper-clothes"], "blouse": ["upper-clothes"], "t-shirt": ["upper-clothes"],
    "pants": ["pants"], "trousers": ["pants"], "jeans": ["pants"], "shorts": ["pants"], "skirt": ["skirt"], "dress": ["dress"],
    "belt": ["belt"], "bag": ["bag"], "purse": ["bag"], "scarf": ["scarf"],
    "shoes": ["left-shoe", "right-shoe"], "shoe": ["left-shoe", "right-shoe"], "boots": ["left-shoe", "right-shoe"],
    "sneakers": ["left-shoe", "right-shoe"], "feet": ["left-shoe", "right-shoe"], "foot": ["left-shoe", "right-shoe"],
    "outfit": CLOTHES, "clothes": CLOTHES, "clothing": CLOTHES, "suit": ["upper-clothes", "pants", "dress", "skirt"],
    "uniform": ["upper-clothes", "pants", "dress", "skirt"], "face": ["face"], "head": ["face", "hair"],
    "arms": ["left-arm", "right-arm"], "arm": ["left-arm", "right-arm"], "hands": ["left-arm", "right-arm"], "hand": ["left-arm", "right-arm"],
    "legs": ["left-leg", "right-leg"], "leg": ["left-leg", "right-leg"],
    "skin": ["face", "left-arm", "right-arm", "left-leg", "right-leg"],
    "body": ["face", "hair", "left-arm", "right-arm", "left-leg", "right-leg"] + CLOTHES, "person": None, "man": None, "woman": None,
}
FACE = {"eyes": ["l_eye", "r_eye"], "eye": ["l_eye", "r_eye"], "lips": ["u_lip", "l_lip"], "lip": ["u_lip", "l_lip"],
        "mouth": ["mouth", "u_lip", "l_lip"], "nose": ["nose"], "eyebrows": ["l_brow", "r_brow"], "brows": ["l_brow", "r_brow"],
        "eyebrow": ["l_brow", "r_brow"], "ears": ["l_ear", "r_ear"], "ear": ["l_ear", "r_ear"], "neck": ["neck"],
        "beard": "BEARD", "mustache": "BEARD", "moustache": "BEARD", "chin": "BEARD"}

def key_for(tgt):
    """Return ('face'|'human', key) when the target is a body part / clothing, else None."""
    words = re.findall(r"[a-z\-]+", (tgt or "").lower())
    for w in reversed(words):  # last noun usually the thing ("her long hair")
        if w in FACE: return "face", w
        if w in HUMAN: return "human", w
    return None

class _Net:
    def __init__(self, name, plan_feature="isnet"):
        import openvino as ov, devices
        d = os.path.join(MD, name); self.labels = {int(k): v.lower() for k, v in json.load(open(os.path.join(d, "labels.json"))).items()}
        c = ov.Core(); c.set_property({"CACHE_DIR": CACHE}); m = c.read_model(os.path.join(d, "model.xml")); err = []
        labs = devices.plan(plan_feature) or ["CPU"]
        if "CPU" not in labs: labs = labs + ["CPU"]
        for lab in labs:
            try: t = time.time(); self.cm = c.compile_model(m, devices.ov(lab)); self.device = lab; logging.info("%s on %s (%.1fs)", name, lab, time.time() - t); return
            except Exception as e: err.append(f"{lab}: {e}"); logging.warning("%s on %s failed: %s", name, lab, str(e)[:200])
        raise RuntimeError(f"{name}: no device worked: " + " | ".join(err))
    def run(self, im):
        x = (np.asarray(im.convert("RGB").resize((512, 512), Image.BILINEAR), np.float32) / 255.0 - MEAN) / STD
        lg = np.asarray(self.cm(x.transpose(2, 0, 1)[None].astype(np.float32))[0])[0]  # C,128,128
        return lg
    def seg(self, im):
        lg = self.run(im); up = []
        for c in range(lg.shape[0]):
            up.append(np.asarray(Image.fromarray(lg[c]).resize(im.size, Image.BILINEAR)))
        return np.argmax(np.stack(up), 0)
    def ids(self, names): return [i for i, n in self.labels.items() if n in names]

_nets = {}; _pref = None
def net(name):
    global _pref
    import devices
    if devices.get() != _pref: _nets.clear(); _pref = devices.get()
    if name not in _nets: _nets[name] = _Net(name)
    return _nets[name]

def available():
    return all(os.path.exists(os.path.join(MD, n, "model.xml")) for n in ("human-parse", "face-parse"))

def _clean(a, im):
    m = Image.fromarray((a * 255).astype(np.uint8))
    return m.filter(ImageFilter.MedianFilter(5)) if m.getbbox() else m

def mask(im, tgt):
    """-> (L mask, confidence 0..1, info) or None when the target isn't a body part/clothing."""
    k = key_for(tgt)
    if not k: return None
    kind, w = k; t = time.time(); hn = net("human-parse"); lab = hn.seg(im)
    if kind == "human":
        names = HUMAN[w]
        if names is None: names = [n for n in hn.labels.values() if n != "background"]
        a = np.isin(lab, hn.ids(names))
        info = f"human parser ({'+'.join(names[:4])}{'...' if len(names) > 4 else ''}) on {hn.device}"
    else:
        face = np.isin(lab, hn.ids(["face", "hair"]))
        if not face.any(): return Image.new("L", im.size, 0), 0.0, "no face found"
        ys, xs = np.nonzero(face); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pw, ph = int((x1 - x0) * 0.25) + 8, int((y1 - y0) * 0.25) + 8
        box = (max(0, x0 - pw), max(0, y0 - ph), min(im.size[0], x1 + pw), min(im.size[1], y1 + ph))
        fn = net("face-parse"); crop = im.crop(box); fl = fn.seg(crop); spec = FACE[w]
        if spec == "BEARD":
            nose = np.isin(fl, fn.ids(["nose"])); skin = np.isin(fl, fn.ids(["skin", "mouth", "u_lip", "l_lip"]))
            cut = int(np.nonzero(nose)[0].mean()) if nose.any() else fl.shape[0] // 2
            sub = skin.copy(); sub[:cut] = False; sub &= ~np.isin(fl, fn.ids(["mouth", "u_lip", "l_lip"]))
            names = ["lower face"]
        else:
            sub = np.isin(fl, fn.ids(spec)); names = spec
        a = np.zeros(lab.shape, bool); a[box[1]:box[3], box[0]:box[2]] = sub
        info = f"face parser ({'+'.join(names)}) on {fn.device}"
    conf = 0.95 if (a.sum() > 30 if kind == "face" else a.mean() > 0.003) else 0.0
    return _clean(a, im), conf, f"{info}, {time.time() - t:.2f}s"
