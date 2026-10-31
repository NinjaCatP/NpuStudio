"""Tiny wire format for NpuStudio desktop link: JSON header + binary blobs (PNG for images, .npy for arrays).
No pickle, so a request can only carry data, never code."""
import io, json, struct
import numpy as np
from PIL import Image

def pack(obj):
    blobs = []
    def enc(o):
        if isinstance(o, Image.Image):
            b = io.BytesIO(); o.save(b, "PNG", compress_level=1); blobs.append(b.getvalue()); return {"__img__": len(blobs) - 1}
        if isinstance(o, np.ndarray):
            b = io.BytesIO(); np.save(b, o, allow_pickle=False); blobs.append(b.getvalue()); return {"__npy__": len(blobs) - 1}
        if isinstance(o, tuple): return {"__tup__": [enc(x) for x in o]}
        if isinstance(o, list): return [enc(x) for x in o]
        if isinstance(o, dict): return {str(k): enc(v) for k, v in o.items()}
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if o is None or isinstance(o, (bool, int, float, str)): return o
        raise TypeError(f"can't send {type(o).__name__}")
    h = json.dumps({"o": enc(obj), "n": [len(b) for b in blobs]}).encode()
    return struct.pack("<I", len(h)) + h + b"".join(blobs)

def unpack(data):
    n = struct.unpack("<I", data[:4])[0]; h = json.loads(data[4:4 + n]); off = 4 + n; blobs = []
    for L in h["n"]: blobs.append(data[off:off + L]); off += L
    def dec(o):
        if isinstance(o, dict):
            if "__img__" in o: im = Image.open(io.BytesIO(blobs[o["__img__"]])); im.load(); return im
            if "__npy__" in o: return np.load(io.BytesIO(blobs[o["__npy__"]]), allow_pickle=False)
            if "__tup__" in o: return tuple(dec(x) for x in o["__tup__"])
            return {k: dec(v) for k, v in o.items()}
        if isinstance(o, list): return [dec(x) for x in o]
        return o
    return dec(h["o"])
