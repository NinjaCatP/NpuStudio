"""Desktop output folder: every phone/remote job saves its FULL-size result here (output\\gen, edit, upscale, aichange...).
Configurable in the desktop app (Outputs menu); stored in models/output.json."""
import os, re, json, time, threading, logging
APP = os.path.dirname(os.path.abspath(__file__)); CFG = os.path.join(APP, "models", "output.json")
DEFAULT = os.path.join(APP, "output"); _lock = threading.Lock()

def folder():
    try: d = json.load(open(CFG)).get("dir") or DEFAULT
    except Exception: d = DEFAULT
    return os.path.abspath(d)

def set_folder(d):
    os.makedirs(os.path.dirname(CFG), exist_ok=True)
    json.dump({"dir": os.path.abspath(d) if d else ""}, open(CFG, "w"))

def _slug(t, n=40):
    t = re.sub(r"[^A-Za-z0-9]+", "-", str(t or "")).strip("-").lower()
    return t[:n].rstrip("-")

def save(im, kind, tag=""):
    """Save a PIL image as output/<kind>/YYYY-MM-DD/HHMM_<tag>.png (local time; JPEG q95 over 24 MP). Returns the path."""
    d = os.path.join(folder(), kind, time.strftime("%Y-%m-%d")); os.makedirs(d, exist_ok=True)
    big = im.size[0] * im.size[1] > 24e6; ext = "jpg" if big else "png"
    with _lock:
        base = "_".join(x for x in (time.strftime("%H%M"), _slug(tag).replace("-", "_") or kind) if x)
        p = os.path.join(d, f"{base}.{ext}"); n = 2
        while os.path.exists(p): p = os.path.join(d, f"{base}_{n}.{ext}"); n += 1
        open(p, "wb").close()
    if big: im.convert("RGB").save(p, "JPEG", quality=95)
    else: im.save(p, "PNG", compress_level=3)
    return p

def autosave(im, kind, tag=""):
    """For the desktop/laptop apps: save, log, never raise. Returns the path or None."""
    try: p = save(im, kind, tag); logging.info("saved %s", p); return p
    except Exception: logging.exception("autosave %s", kind); return None

def open_folder():
    d = folder(); os.makedirs(d, exist_ok=True); os.startfile(d)
