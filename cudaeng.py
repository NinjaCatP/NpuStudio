"""NVIDIA (CUDA) engines via torch + diffusers. Only one pipeline lives in VRAM at a time (8 GB card)."""
import os, time, threading, logging, math
import numpy as np
from PIL import Image
APP = os.path.dirname(os.path.abspath(__file__)); HF = os.path.join(APP, "models", "hf")
REPOS = {"ip2p": "timbrooks/instruct-pix2pix",
         "inpaint": "stable-diffusion-v1-5/stable-diffusion-inpainting",
         "lcm": "SimianLuo/LCM_Dreamshaper_v7"}
_lock = threading.Lock(); _cur = {"name": None, "pipe": None}

def available():
    try:
        import torch; return torch.cuda.is_available()
    except Exception: return False

def _pipe(name, status=None):
    import torch, diffusers
    if _cur["name"] == name: return _cur["pipe"]
    _cur["pipe"] = None; _cur["name"] = None; torch.cuda.empty_cache()
    cls = {"ip2p": diffusers.StableDiffusionInstructPix2PixPipeline,
           "inpaint": diffusers.StableDiffusionInpaintPipeline,
           "lcm": diffusers.DiffusionPipeline}[name]
    if status: status(f"Loading {name} on NVIDIA (first time downloads ~2-4 GB)...")
    t = time.time(); kw = dict(torch_dtype=torch.float16, cache_dir=HF, safety_checker=None, requires_safety_checker=False)
    try: p = cls.from_pretrained(REPOS[name], variant="fp16", **kw)
    except Exception: p = cls.from_pretrained(REPOS[name], **kw)
    p = p.to("cuda"); p.set_progress_bar_config(disable=True)
    _cur.update(name=name, pipe=p); logging.info("cuda %s loaded %.1fs", name, time.time() - t); return p

def unload():
    with _lock:
        _cur.update(name=None, pipe=None)
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass

def _cb(progress, n):
    def f(pipe, i, t, kw):
        if progress: progress(i + 1, n)
        return kw
    return f

def _fit(im, side=768):
    """Resize so the long side is `side`, both dims multiple of 8."""
    W, H = im.size; s = side / max(W, H)
    return im.resize((max(8, round(W * s / 8) * 8), max(8, round(H * s / 8) * 8)), Image.LANCZOS)

def ip2p(im, instr, steps=20, image_guidance=1.5, guidance=7.5, seed=1, progress=None, status=None, side=768):
    import torch
    with _lock:
        p = _pipe("ip2p", status); src = _fit(im.convert("RGB"), side)
        if status: status("InstructPix2Pix on NVIDIA...")
        r = p(instr, image=src, num_inference_steps=steps, image_guidance_scale=image_guidance, guidance_scale=guidance,
              generator=torch.Generator("cuda").manual_seed(int(seed)), callback_on_step_end=_cb(progress, steps)).images[0]
    return r

def inpaint(rg512, mk512, prompt, steps=30, strength=1.0, guidance=7.5, seed=1, progress=None, status=None, negative=None):
    import torch
    with _lock:
        p = _pipe("inpaint", status)
        n = max(1, int(steps * strength))
        return p(prompt, image=rg512, mask_image=mk512, num_inference_steps=steps, strength=strength, guidance_scale=guidance,
                 negative_prompt=negative or "blurry, deformed, low quality", generator=torch.Generator("cuda").manual_seed(int(seed)),
                 callback_on_step_end=_cb(progress, n)).images[0]

def gen(prompt, w, h, steps=4, seed=1, progress=None, status=None):
    import torch
    with _lock:
        p = _pipe("lcm", status)
        return p(prompt, width=w, height=h, num_inference_steps=steps, guidance_scale=8.0,
                 generator=torch.Generator("cuda").manual_seed(int(seed)), callback_on_step_end=_cb(progress, steps)).images[0]

# ---------------- Real-ESRGAN x4plus on CUDA (OpenVINO on NVIDIA is ~5 s/tile; torch fp16 is far faster) ----------------
ESR_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
ESR_PTH = os.path.join(APP, "models", "RealESRGAN_x4plus.pth")
_esr = {"m": None}; _esr_lock = threading.Lock()

def _rrdb():
    import torch, torch.nn as nn, torch.nn.functional as F
    class RDB(nn.Module):
        def __init__(s, nf=64, gc=32):
            super().__init__()
            s.conv1 = nn.Conv2d(nf, gc, 3, 1, 1); s.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            s.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1); s.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            s.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1); s.l = nn.LeakyReLU(0.2, True)
        def forward(s, x):
            x1 = s.l(s.conv1(x)); x2 = s.l(s.conv2(torch.cat((x, x1), 1))); x3 = s.l(s.conv3(torch.cat((x, x1, x2), 1)))
            x4 = s.l(s.conv4(torch.cat((x, x1, x2, x3), 1))); return s.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x
    class RRDB(nn.Module):
        def __init__(s, nf=64):
            super().__init__(); s.rdb1 = RDB(nf); s.rdb2 = RDB(nf); s.rdb3 = RDB(nf)
        def forward(s, x): return s.rdb3(s.rdb2(s.rdb1(x))) * 0.2 + x
    class Net(nn.Module):
        def __init__(s, nf=64, nb=23):
            super().__init__()
            s.conv_first = nn.Conv2d(3, nf, 3, 1, 1); s.body = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
            s.conv_body = nn.Conv2d(nf, nf, 3, 1, 1); s.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1); s.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
            s.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1); s.conv_last = nn.Conv2d(nf, 3, 3, 1, 1); s.l = nn.LeakyReLU(0.2, True)
        def forward(s, x):
            f = s.conv_first(x); f = f + s.conv_body(s.body(f))
            f = s.l(s.conv_up1(F.interpolate(f, scale_factor=2, mode="nearest")))
            f = s.l(s.conv_up2(F.interpolate(f, scale_factor=2, mode="nearest")))
            return s.conv_last(s.l(s.conv_hr(f)))
    return Net()

def esr_model():
    import torch
    with _esr_lock:
        if _esr["m"] is None:
            if not os.path.exists(ESR_PTH):
                import urllib.request; urllib.request.urlretrieve(ESR_URL, ESR_PTH + ".part"); os.replace(ESR_PTH + ".part", ESR_PTH)
            sd = torch.load(ESR_PTH, map_location="cpu", weights_only=True); sd = sd.get("params_ema", sd.get("params", sd))
            torch.backends.cudnn.benchmark = True
            m = _rrdb(); m.load_state_dict(sd); _esr["m"] = m.half().cuda().eval().to(memory_format=torch.channels_last)
        return _esr["m"]

def esr_tile(tile):
    """uint8 HxWx3 -> uint8 4Hx4Wx3 (RGB)."""
    import torch
    m = esr_model()
    with torch.inference_mode():
        x = (torch.from_numpy(tile).cuda().permute(2, 0, 1)[None].half() / 255.0).contiguous(memory_format=torch.channels_last)
        y = m(x)[0].clamp_(0, 1).mul_(255).add_(0.5).byte().permute(1, 2, 0).cpu().numpy()
    return y

# ---------------- ControlNet inpaint: new garment drawn inside the clothing mask, outline held by a lineart map ----------------
CN_BASE = "Lykon/dreamshaper-8-inpainting"  # SD1.5 inpainting fine-tune, much better at anime/illustration (CreativeML OpenRAIL-M)
CN_REPO = "lllyasviel/control_v11p_sd15s2_lineart_anime"
_cn = {"pipe": None}
def _cn_pipe(status=None):
    import torch, diffusers
    if _cur["name"] == "cn": return _cur["pipe"]
    _cur["pipe"] = None; _cur["name"] = None; torch.cuda.empty_cache()
    if status: status("Loading ControlNet inpaint on NVIDIA (first time downloads ~2.7 GB)...")
    t = time.time()
    try: cn = diffusers.ControlNetModel.from_pretrained(CN_REPO, torch_dtype=torch.float16, cache_dir=HF, variant="fp16")
    except Exception: cn = diffusers.ControlNetModel.from_pretrained(CN_REPO, torch_dtype=torch.float16, cache_dir=HF)
    kw = dict(controlnet=cn, torch_dtype=torch.float16, cache_dir=HF, safety_checker=None, requires_safety_checker=False)
    try: p = diffusers.StableDiffusionControlNetInpaintPipeline.from_pretrained(CN_BASE, variant="fp16", **kw)
    except Exception: p = diffusers.StableDiffusionControlNetInpaintPipeline.from_pretrained(CN_BASE, **kw)
    p = p.to("cuda"); p.set_progress_bar_config(disable=True)
    _cur.update(name="cn", pipe=p); logging.info("cuda controlnet inpaint loaded %.1fs", time.time() - t); return p

def inpaint_cn(img, mask, control, prompt, negative, steps=30, guidance=8.0, cn_scale=0.8, seed=1, progress=None, status=None):
    """img/mask/control are the same size (multiple of 8). Returns the generated image at that size."""
    import torch
    with _lock:
        p = _cn_pipe(status)
        return p(prompt, image=img, mask_image=mask, control_image=control, negative_prompt=negative, num_inference_steps=steps,
                 guidance_scale=guidance, controlnet_conditioning_scale=cn_scale, strength=1.0, width=img.size[0], height=img.size[1],
                 generator=torch.Generator("cuda").manual_seed(int(seed)), callback_on_step_end=_cb(progress, steps)).images[0]
