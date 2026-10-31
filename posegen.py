"""Pose changes for AI Change: OpenPose skeleton -> edit arm keypoints for a named pose -> redraw arms + surroundings
with openpose + lineart ControlNets (NVIDIA). Torso/outfit silhouette is kept as lineart and protected; no body reshaping.
Each candidate is re-detected and must match the target arm pose; hands on the chest/torso are rejected."""
import os, logging, math, numpy as np
from PIL import Image, ImageFilter, ImageDraw
APP = os.path.dirname(os.path.abspath(__file__)); HF = os.path.join(APP, "models", "hf")
_det = {"m": None}
NECK, RSH, REL, RWR, LSH, LEL, LWR, RHIP, LHIP, NOSE = 1, 2, 3, 4, 5, 6, 7, 8, 11, 0

def detector():
    if _det["m"] is None:
        from controlnet_aux import OpenposeDetector
        m = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=HF)
        try:
            import torch
            if torch.cuda.is_available(): m = m.to("cuda")
        except Exception: pass
        _det["m"] = m
    return _det["m"]

def detect(im):
    """-> list of 18 (x,y) in pixels or None, for the biggest person; None if nothing found."""
    W, H = im.size; ps = detector().detect_poses(np.asarray(im.convert("RGB")))
    if not ps: return None
    best = max(ps, key=lambda p: sum(k is not None for k in p.body.keypoints))
    return [None if k is None else (k.x * W, k.y * H) for k in best.body.keypoints]

def _fallback_kp(im):
    """Anime figures often defeat OpenPose: estimate shoulders/hips from the outfit + face masks."""
    import aichange
    W, H = im.size
    hard = np.asarray(aichange.outfit_mask(im, Image.new("L", im.size, 0), "outfit")[0]) > 127
    if not hard.any(): return None
    ys, xs = np.nonzero(hard); y0, y1 = ys.min(), ys.max()
    row = lambda y: np.nonzero(hard[int(y)])[0]
    ysh = y0 + (y1 - y0) * 0.06; r = row(ysh)
    if len(r) < 2: return None
    sw = (r.max() - r.min()); cx = (r.max() + r.min()) / 2
    kp = [None] * 18
    kp[NECK] = (cx, y0); kp[RSH] = (cx - sw * 0.42, ysh); kp[LSH] = (cx + sw * 0.42, ysh)
    tl = (y1 - y0) * 0.55
    kp[RHIP] = (cx - sw * 0.3, y0 + tl); kp[LHIP] = (cx + sw * 0.3, y0 + tl)
    kp[REL] = (kp[RSH][0] - sw * 0.05, ysh + tl * 0.5); kp[RWR] = (kp[RSH][0], ysh + tl * 0.95)
    kp[LEL] = (kp[LSH][0] + sw * 0.05, ysh + tl * 0.5); kp[LWR] = (kp[LSH][0], ysh + tl * 0.95)
    kp[NOSE] = (cx, y0 - tl * 0.35)
    return kp

def frame(kp):
    rs, ls = kp[RSH], kp[LSH]
    sw = math.dist(rs, ls); ysh = (rs[1] + ls[1]) / 2
    hips = [kp[i] for i in (RHIP, LHIP) if kp[i]]
    tl = (sum(h[1] for h in hips) / len(hips) - ysh) if hips else sw * 1.4
    return rs, ls, sw, ysh, max(tl, sw * 0.8)

def target(kp, pose):
    """Edit the arm keypoints (person's right arm = image left when facing the viewer)."""
    kp = list(kp); rs, ls, sw, ysh, tl = frame(kp)
    left_img, right_img = (RSH, REL, RWR), (LSH, LEL, LWR)
    if rs[0] > ls[0]: left_img, right_img = right_img, left_img
    L, R = kp[left_img[0]], kp[right_img[0]]
    if pose == "wave":  # image-right arm raised beside the head, open hand
        kp[right_img[1]] = (R[0] + sw * 0.35, R[1] - tl * 0.05); kp[right_img[2]] = (R[0] + sw * 0.4, R[1] - tl * 0.6)
    elif pose == "raise":
        kp[right_img[1]] = (R[0] + sw * 0.15, R[1] - tl * 0.4); kp[right_img[2]] = (R[0] + sw * 0.2, R[1] - tl * 0.85)
        kp[left_img[1]] = (L[0] - sw * 0.15, L[1] - tl * 0.4); kp[left_img[2]] = (L[0] - sw * 0.2, L[1] - tl * 0.85)
    elif pose == "crossed":  # forearms crossed in front of the waist / lower chest
        y = ysh + tl * 0.7
        kp[left_img[1]] = (L[0] - sw * 0.05, ysh + tl * 0.6); kp[left_img[2]] = (R[0] - sw * 0.15, y)
        kp[right_img[1]] = (R[0] + sw * 0.05, ysh + tl * 0.6); kp[right_img[2]] = (L[0] + sw * 0.15, y - tl * 0.04)
    elif pose == "hips":
        kp[left_img[1]] = (L[0] - sw * 0.45, ysh + tl * 0.45); kp[left_img[2]] = (L[0] + sw * 0.05, ysh + tl * 0.9)
        kp[right_img[1]] = (R[0] + sw * 0.45, ysh + tl * 0.45); kp[right_img[2]] = (R[0] - sw * 0.05, ysh + tl * 0.9)
    return kp

def pose_image(kp, W, H):
    from controlnet_aux.open_pose import util
    from controlnet_aux.open_pose.body import Keypoint
    pts = [None if p is None else Keypoint(p[0] / W, p[1] / H) for p in kp]
    return Image.fromarray(util.draw_bodypose(np.zeros((H, W, 3), np.uint8), pts))

def arm_mask(im, kp_old, kp_new, protect):
    """Old arms + the space along the new arm bones (thick strokes) + hands; face/hair/torso-outfit protected
    except where the new arm must cross in front of it."""
    import aichange
    W, H = im.size; rs, ls, sw, ysh, tl = frame(kp_old); w = max(8, int(sw * 0.28))
    old = np.asarray(aichange.pose_mask(im, Image.new("L", im.size, 0))[0]) > 127
    new = Image.new("L", im.size, 0); d = ImageDraw.Draw(new); oldi = Image.new("L", im.size, 0); od = ImageDraw.Draw(oldi)
    for a, b, c in ((RSH, REL, RWR), (LSH, LEL, LWR)):
        if kp_new[b] != kp_old[b] or kp_new[c] != kp_old[c]:
            pts = [kp_new[a], kp_new[b], kp_new[c]]
            d.line(pts, fill=255, width=w * 2, joint="curve")
            x, y = kp_new[c]; d.ellipse((x - w * 1.3, y - w * 1.3, x + w * 1.3, y + w * 1.3), fill=255)
            for p in (kp_old[b], kp_old[c]):  # where the old arm was
                if p: od.ellipse((p[0] - w * 1.2, p[1] - w * 1.2, p[0] + w * 1.2, p[1] + w * 1.2), fill=255)
            od.line([kp_old[a], kp_old[b] or kp_old[a], kp_old[c] or kp_old[a]], fill=255, width=int(w * 2.2))
    n = np.asarray(new) > 127; global _last_new; _last_new = n
    old = old | ((np.asarray(oldi) > 127) & ~protect)
    m = (old | n) & ~(protect & ~n)
    hard = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    return hard, hard.filter(ImageFilter.GaussianBlur(2))

def hand_on_chest(img, kp_target=None):
    """True when a wrist lands in the chest band (between the shoulders, shoulder line to ~45% of the torso) and the
    target pose did not ask for it."""
    try: kp = detect(img)
    except Exception: return False
    if not kp or not kp[RSH] or not kp[LSH]: return False
    rs, ls, sw, ysh, tl = frame(kp); x0, x1 = sorted((rs[0], ls[0]))
    for wi in (RWR, LWR):
        p = kp[wi]
        if p and x0 + sw * 0.05 < p[0] < x1 - sw * 0.05 and ysh < p[1] < ysh + tl * 0.5: return True
    return False

def pose_match(img, kp_t):
    """0..1: how close the re-detected arm keypoints are to the target (1 = exact). Missing joints count as misses."""
    try: kp = detect(img)
    except Exception: return 0.0
    if not kp: return 0.0
    rs, ls, sw, ysh, tl = frame(kp_t); sc = []
    for j in (REL, RWR, LEL, LWR):
        if kp_t[j] is None: continue
        # left/right can swap in detection - take the nearer of the two same-type joints
        alt = {REL: LEL, LEL: REL, RWR: LWR, LWR: RWR}[j]
        ds = [math.dist(kp[k], kp_t[j]) for k in (j, alt) if kp[k]]
        sc.append(max(0.0, 1 - (min(ds) / sw if ds else 2)))
    return float(np.mean(sc)) if sc else 0.0

def redraw_pose(im, pose, prompt_neg, progress=None, status=None, N=4):
    """-> (image, how) or None when no NVIDIA / no person."""
    import cudaeng, aichange
    if not cudaeng.available(): return None
    im = im.convert("RGB"); W, H = im.size
    kp = None
    try: kp = detect(im)
    except Exception: logging.exception("openpose detect")
    src = "OpenPose"
    if not kp or not kp[RSH] or not kp[LSH]: kp = _fallback_kp(im); src = "outfit-estimated skeleton"
    if not kp: return None
    for j, (s, d) in {REL: (RSH, 0.5), RWR: (RSH, 0.95), LEL: (LSH, 0.5), LWR: (LSH, 0.95)}.items():  # fill missing arm joints
        if kp[j] is None: kp[j] = (kp[s][0], kp[s][1] + frame(kp)[4] * d)
    kt = target(kp, pose)
    # protect face/hair + the outfit silhouette (lineart keeps the outline); arms may pass in front where the pose needs it
    prot = np.zeros((H, W), bool); outfit = np.zeros((H, W), bool)
    try:
        outfit = np.asarray(aichange.outfit_mask(im, Image.new("L", im.size, 0), "outfit")[0]) > 127; prot |= outfit
        s = aichange._seg()
        if s is not None:
            q = np.max([s.probs(im, t) for t in ("face", "hair")], axis=0)
            prot |= np.asarray(Image.fromarray(((q > max(0.35, q.max() * 0.5)) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) > 127
    except Exception: logging.exception("pose protect")
    hard, soft = arm_mask(im, kp, kt, prot)
    lineart = aichange.control_map(im, hard, "loose")
    a0 = np.asarray(im, np.float32); ref = np.median(a0[outfit], 0) if outfit.sum() > 200 else None
    _r = max(3, int(frame(kp)[2] * 0.12)); nb = Image.fromarray((_last_new * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(_r)).point(lambda v: 255 if v > 8 else 0)
    R = (np.asarray(hard) > 127) & ~prot & ~(np.asarray(nb) > 127)  # margin: the drawn arm may sit a little off the bone line  # around the old arm, outside her outline: must stay background
    ring = (np.asarray(hard.filter(ImageFilter.MaxFilter(31))) > 127) & ~(np.asarray(hard) > 127) & ~prot
    bgc = np.median(a0[ring], 0) if ring.sum() > 50 else None
    keepbg = None
    if bgc is not None:
        kb = R & (np.sqrt(((a0 - bgc) ** 2).sum(-1)) < 45)
        keepbg = Image.fromarray((kb * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.5))
    def spill(r):
        """Share of the area outside her old outline (minus the new arm path) that now holds something other than background."""
        if bgc is None or R.sum() < 100: return 0.0
        b = np.asarray(r, np.float32)
        newstuff = (np.sqrt(((b - bgc) ** 2).sum(-1)) > 45) & (np.sqrt(((b - a0) ** 2).sum(-1)) > 30)
        return float(newstuff[R].mean())
    pimg = pose_image(kt, W, H)
    p, neg = prompt_neg
    p = {"wave": "open palm waving hello, fingers spread, ", "crossed": "forearms folded in front of her waist, "}.get(pose, "") + p
    neg += ", fist, changed body shape, extra breast, deformed torso"
    best = None
    for i in range(N * 2):
        if i == N and best is not None and best[0] >= 0.5: break  # second seed batch only if the first all failed
        if status: status(f"Drawing the new pose - version {i + 1} of {N} (OpenPose + lineart ControlNet, NVIDIA)...")
        r = cudaeng.inpaint_pose(im, hard, lineart, pimg, p, neg, seed=21 + i * 13, progress=progress)
        r = Image.composite(r, im, soft)
        if keepbg is not None: r = Image.composite(im, r, keepbg)  # outside her outline and off the new arm path: background stays
        chest = hand_on_chest(r) and pose != "crossed"
        sp = spill(r); m = pose_match(r, kt); sc = -1 if chest else (-0.5 if sp > 0.3 else m)
        logging.info("pose candidate %d match %.2f chest %s silhouette-spill %.2f", i, m, chest, sp)
        if os.environ.get("NPUSTUDIO_DEBUG"): r.save(f"pose_cand_{i}.jpg", quality=85)
        if best is None or sc > best[0]: best = (sc, r, m)
    sc, r, m = best
    if sc <= -1: return im, "Method: pose redraw - every version put a hand on the body, so nothing was kept"
    if sc < 0: return im, "NOT DONE - every version changed her outline (silhouette check). Original kept; try again."
    how = f"Method: pose redraw (OpenPose + lineart ControlNet, {src}, best of {N}, pose match {m:.2f})"
    if m < 0.5: how += " - weak pose match"
    return r, how
