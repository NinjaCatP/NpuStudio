"""Device picker: Auto / NPU / Intel Arc / NVIDIA / All. Preference saved in models/device.json (app folder only)."""
import os, json, logging
APP = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(APP, "models")
PREF = os.path.join(MD, "device.json"); BENCH = os.path.join(MD, "devbench.json")
CHOICES = ["Auto", "NPU", "Intel Arc", "NVIDIA", "All", "Desktop PC", "Both machines", "Laptop NPU", "Laptop Intel Arc", "Laptop RTX 4070"]
REMOTE = ("Desktop PC", "Both machines", "Laptop NPU", "Laptop Intel Arc", "Laptop RTX 4070")  # handled by remote.py; local chips then run as Auto
def remote_choices():
    """Laptop (paired to a desktop): Desktop PC + Both machines. Desktop (runs the server): the laptop's chips + Both machines."""
    try:
        import remote
        if remote.role() == "desktop": return ["Laptop NPU", "Laptop Intel Arc", "Laptop RTX 4070", "Both machines"]
    except Exception: pass
    return ["Desktop PC", "Both machines"]
# which hardware each feature can use (from testing on this laptop)
#  key "cuda" = NVIDIA through torch/diffusers; OV names are mapped at runtime.
FEATURE_HW = {
    "upscale_fast": ["NPU", "Intel Arc", "NVIDIA", "CPU"],
    "upscale_sharp": ["NPU", "Intel Arc", "NVIDIA", "CPU"],
    "lama": ["Intel Arc", "NVIDIA", "CPU"],       # NPU hangs compiling LaMa
    "isnet": ["NPU", "Intel Arc", "NVIDIA", "CPU"],
    "inpaint": ["Intel Arc", "NVIDIA", "CPU"],    # NPU inpaint UNet fails at run time
    "img2img": ["NPU", "Intel Arc", "NVIDIA", "CPU"],
    "gen": ["NPU", "Intel Arc", "NVIDIA", "CPU"],
    "ip2p": ["NVIDIA"],                           # InstructPix2Pix: torch CUDA only
}
_map = None
import threading
_lock = threading.Lock()

def ov_map():
    """label -> OpenVINO device string, from the device names (thread-safe; built once)."""
    global _map
    with _lock:
        if _map is None:
            m = {}
            try:
                import openvino as ov; c = ov.Core()
                for d in c.available_devices:
                    n = c.get_property(d, "FULL_DEVICE_NAME")
                    if d == "NPU": m["NPU"] = d
                    elif d == "CPU": m["CPU"] = d
                    elif "NVIDIA" in n.upper(): m["NVIDIA"] = d
                    elif d.startswith("GPU"): m.setdefault("Intel Arc", d)
            except Exception: logging.exception("ov_map")
            _map = m
        return _map

def present():
    m = ov_map(); return [c for c in ("NPU", "Intel Arc", "NVIDIA") if c in m]

def choices():
    p = present(); return ["Auto"] + p + (["All"] if len(p) > 1 else []) + remote_choices()

def get_raw():
    """The picker value, including Desktop PC / Both machines."""
    try: v = json.load(open(PREF)).get("device", "Auto")
    except Exception: v = "Auto"
    return v if v in choices() else "Auto"

def get():
    """Local device preference. With Desktop PC / Both machines picked, local work runs as Auto."""
    v = get_raw(); return "Auto" if v in REMOTE else v

def set(v):
    os.makedirs(MD, exist_ok=True); json.dump({"device": v}, open(PREF, "w"))

def bench():
    try: return json.load(open(BENCH))
    except Exception: return {}

def _speed(feature, label):
    t = bench().get(feature, {}).get(label)
    return t if isinstance(t, (int, float)) else 1e9

def plan(feature, pref=None):
    """Ordered list of hardware labels to use. Single choice is respected exactly (empty list => unsupported).
    Auto => fastest first (benchmarks), with fallbacks. All => every working device (caller may run them in parallel)."""
    pref = pref or get()
    if pref in REMOTE: pref = "Auto"
    ok = [h for h in FEATURE_HW.get(feature, ["CPU"]) if h in ov_map() or h == "CPU"]
    if pref in ("NPU", "Intel Arc", "NVIDIA"):
        return [pref] if pref in ok else []
    order = sorted(ok, key=lambda h: (_speed(feature, h), ["NVIDIA", "Intel Arc", "NPU", "CPU"].index(h)))
    if pref == "All": return [h for h in order if h != "CPU"] or order
    return order

def ov(label): return ov_map().get(label, "CPU")

def unsupported(feature, pref=None):
    pref = pref or get()
    return f"{feature} can't run on {pref} - pick another Device (supported: {', '.join(FEATURE_HW.get(feature, []))})"
