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

POSES = {  # key: (prompt prefix, what the hands should NOT touch is enforced by hand_on_chest for every pose)
    "wave": "open palm waving hello beside her head, fingers spread, ",
    "crossed": "arms crossed, both forearms horizontal and stacked in front of her waist, hands tucked at the opposite elbows, ",
    "hips": "both hands on her hips, elbows pointing outward, ",
    "onehip": "one hand on her hip, elbow out, other arm relaxed, ",
    "raise": "one arm raised up high, ",
    "overhead": "both arms raised straight up above her head, hands in the air, ",
    "peace": "peace sign hand beside her face, index and middle finger raised in a V, other fingers folded, ",
    "behindhead": "both hands behind her head, elbows out to the sides, ",
    "thumbs": "giving a thumbs up with one hand held out to the side, ",
    "point": "pointing to the side with her index finger, arm extended, ",
    "sides": "arms relaxed hanging down at her sides, ",
    "salute": "giving a military salute, flat hand at her brow, ",
    "chin": "thinking pose, hand resting under her chin, ",
}

_body = {"m": None}
def _edge(y, x0):
    """Image-right edge of her body outline at row y (so a hand held 'out to the side' clears her bust)."""
    m = _body["m"]
    if m is None: return x0
    y = int(min(max(y, 0), m.shape[0] - 1)); xs = np.nonzero(m[max(0, y - 10):y + 10].any(0))[0]
    return max(x0, xs.max()) if len(xs) else x0

def target(kp, pose):
    """Edit the arm keypoints (person's right arm = image left when facing the viewer)."""
    kp = list(kp); rs, ls, sw, ysh, tl = frame(kp)
    left_img, right_img = (RSH, REL, RWR), (LSH, LEL, LWR)
    if rs[0] > ls[0]: left_img, right_img = right_img, left_img
    L, R = kp[left_img[0]], kp[right_img[0]]
    cx = (L[0] + R[0]) / 2; hy = kp[NOSE][1] if kp[NOSE] else ysh - sw * 0.55  # head (nose) height
    def setarm(side, el, wr):
        a = left_img if side == "L" else right_img; kp[a[1]] = el; kp[a[2]] = wr
    def hip(side):
        S, s = (L, -1) if side == "L" else (R, 1)
        setarm(side, (S[0] + s * sw * 0.45, ysh + tl * 0.45), (S[0] - s * sw * 0.05, ysh + tl * 0.9))
    def side_(side):
        S, s = (L, -1) if side == "L" else (R, 1)
        setarm(side, (S[0] + s * sw * 0.06, ysh + tl * 0.5), (S[0] + s * sw * 0.04, ysh + tl * 0.97))
    if pose == "wave":
        setarm("R", (R[0] + sw * 0.35, R[1] - tl * 0.05), (R[0] + sw * 0.4, R[1] - tl * 0.6))
    elif pose == "raise":
        setarm("R", (R[0] + sw * 0.15, R[1] - tl * 0.4), (R[0] + sw * 0.2, R[1] - tl * 0.85))
    elif pose == "overhead":
        top = hy - sw * 0.75
        setarm("L", (L[0] - sw * 0.12, (ysh + top) / 2), (L[0] + sw * 0.05, top))
        setarm("R", (R[0] + sw * 0.12, (ysh + top) / 2), (R[0] - sw * 0.05, top))
    elif pose == "crossed":  # forearms horizontal, stacked in front of the waist; each hand at the opposite elbow
        y = ysh + tl * 0.66
        setarm("L", (L[0] - sw * 0.06, y), (R[0] - sw * 0.02, y + tl * 0.02))
        setarm("R", (R[0] + sw * 0.06, y + tl * 0.07), (L[0] + sw * 0.02, y + tl * 0.08))
    elif pose == "hips": hip("L"); hip("R")
    elif pose == "onehip": hip("R")
    elif pose == "sides": side_("L"); side_("R")
    elif pose == "behindhead":  # hands hidden behind the head: wrists at the back edge of the head, elbows out wide
        setarm("L", (L[0] - sw * 0.35, hy - sw * 0.1), (cx - sw * 0.38, hy - sw * 0.25))
        setarm("R", (R[0] + sw * 0.35, hy - sw * 0.1), (cx + sw * 0.38, hy - sw * 0.25))
    elif pose == "peace":
        setarm("R", (R[0] + sw * 0.32, ysh + tl * 0.3), (R[0] + sw * 0.18, hy - sw * 0.05))
    elif pose == "thumbs":
        y = ysh + tl * 0.28; setarm("R", (R[0] + sw * 0.2, ysh + tl * 0.45), (max(R[0] + sw * 0.45, _edge(y, 0) + sw * 0.3), y))
    elif pose == "point":
        y = ysh + tl * 0.02; setarm("R", (max(R[0] + sw * 0.45, _edge(y + tl * 0.06, 0) + sw * 0.1), ysh + tl * 0.08), (max(R[0] + sw * 0.9, _edge(y, 0) + sw * 0.5), y))
    elif pose == "salute":
        setarm("R", (R[0] + sw * 0.45, ysh - sw * 0.05), (R[0] - sw * 0.12, hy - sw * 0.2))
    elif pose == "chin":
        setarm("R", (R[0] + sw * 0.05, ysh + tl * 0.45), (cx + sw * 0.08, hy + sw * 0.32))
    return kp

def pose_ok(img, kp_t, pose):
    """Pose-specific sanity: crossed forearms must be roughly horizontal and low (not running up the chest)."""
    if pose != "crossed": return True
    try: kp = detect(img)
    except Exception: return True
    if not kp or not kp[RSH] or not kp[LSH]: return False
    if not (kp[RWR] and kp[LWR]): return False  # both forearms must be visible and recognisable
    rs, ls, sw, ysh, tl = frame(kp_t)
    for e, w in ((REL, RWR), (LEL, LWR)):
        if kp[w] and kp[w][1] < ysh + tl * 0.45: return False  # a hand up on the chest / shoulder
        if kp[e] and kp[w] and abs(kp[w][1] - kp[e][1]) > tl * 0.3: return False  # diagonal forearm
    return True

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
                if p: od.ellipse((p[0] - w * 1.6, p[1] - w * 1.6, p[0] + w * 1.6, p[1] + w * 1.6), fill=255)
            od.line([kp_old[a], kp_old[b] or kp_old[a], kp_old[c] or kp_old[a]], fill=255, width=int(w * 3))
    thin = Image.new("L", im.size, 0); td = ImageDraw.Draw(thin)
    for a, b, c in ((RSH, REL, RWR), (LSH, LEL, LWR)):
        if kp_new[b] != kp_old[b] or kp_new[c] != kp_old[c]:
            td.line([kp_new[a], kp_new[b], kp_new[c]], fill=255, width=int(w * 1.1), joint="curve")
            x, y = kp_new[c]; td.ellipse((x - w * 0.9, y - w * 0.9, x + w * 0.9, y + w * 0.9), fill=255)
    n = np.asarray(new) > 127; n = (n & ~protect) | ((np.asarray(thin) > 127) & protect)  # only a hand-width path through hair/face
    global _last_new, _last_old, _last_thin; _last_new = n; _last_thin = (np.asarray(thin) > 127) & ~protect
    old = old | ((np.asarray(oldi) > 127) & ~protect); _last_old = old & ~protect
    m = (old | n) & ~(protect & ~n)
    hard = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    return hard, hard.filter(ImageFilter.GaussianBlur(2))

def hand_on_chest(img, kp_target=None, body=None):
    """True when a wrist lands in the chest band (between the shoulders, shoulder line to ~45% of the torso) and the
    target pose did not ask for it."""
    try: kp = detect(img)
    except Exception: return False
    if not kp or not kp[RSH] or not kp[LSH]: return False
    rs, ls, sw, ysh, tl = frame(kp); x0, x1 = sorted((rs[0], ls[0]))
    for wi in (RWR, LWR):
        p = kp[wi]
        if p and x0 + sw * 0.05 < p[0] < x1 - sw * 0.05 and ysh < p[1] < ysh + tl * 0.5: return True
        if p is not None and body is not None and ysh + tl * 0.05 < p[1] < ysh + tl * 0.55:  # hand in front of her chest/bust outline
            x, y = int(p[0]), int(p[1])
            if 0 <= y < body.shape[0] and 0 <= x < body.shape[1] and body[max(0, y - 6):y + 7, max(0, x - 6):x + 7].mean() > 0.5: return True
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
    try: _body["m"] = np.asarray(aichange.outfit_mask(im, Image.new("L", im.size, 0), "outfit")[0]) > 127
    except Exception: _body["m"] = None
    kt = target(kp, pose)
    kt = [None if p is None else (min(max(p[0], 4), W - 4), min(max(p[1], 4), H - 4)) for p in kt]
    # protect face/hair + the outfit silhouette (lineart keeps the outline); arms may pass in front where the pose needs it
    prot = np.zeros((H, W), bool); outfit = np.zeros((H, W), bool)
    try:
        outfit = np.asarray(aichange.outfit_mask(im, Image.new("L", im.size, 0), "outfit")[0]) > 127; prot |= outfit
        s = aichange._seg()
        if s is not None:
            q = np.max([s.probs(im, t) for t in (("face",) if pose in ("overhead", "behindhead") else ("face", "hair"))], axis=0)  # raised arms go in front of the hair
            prot |= np.asarray(Image.fromarray(((q > max(0.35, q.max() * 0.5)) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) > 127
    except Exception: logging.exception("pose protect")
    hard, soft = arm_mask(im, kp, kt, prot)
    try:  # her face (eyes, mouth) is never repainted, even where a hand comes close (salute / chin / behind head)
        s = aichange._seg(); fc = np.asarray(Image.fromarray(((s.probs(im, "face") > 0.5) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR).filter(ImageFilter.MinFilter(5))) > 127
        try:
            import bodyparse
            for part in ("eyes", "eyebrows", "mouth"):
                r0 = bodyparse.mask(im, part)
                if r0 is not None and r0[1] > 0: fc |= np.asarray(r0[0].convert("L").resize((W, H))) > 127
        except Exception: pass
        h2 = (np.asarray(hard) > 127) & ~fc; hard = Image.fromarray((h2 * 255).astype(np.uint8)); soft = hard.filter(ImageFilter.GaussianBlur(2))
    except Exception: logging.exception("pose face protect")
    # clean plate: the old arms are first painted out as plain background, so nothing of them (or fabric strips) can survive
    clean = im
    try:
        om = Image.fromarray((_last_old * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))
        if om.getbbox():
            if status: status("Removing the old arm position (clean background)...")
            clean = aichange.cn_redraw(im, om, om.filter(ImageFilter.GaussianBlur(2)), "empty background only, plain continuous background, nothing in front",
                                       "person, arm, hand, fingers, skin, fabric, clothing, cloth, strip, body", seed=7, progress=progress, mode="loose")
    except Exception: logging.exception("pose clean plate")
    lineart = aichange.control_map(clean, hard, "loose")
    a0 = np.asarray(im, np.float32); ref = np.median(a0[outfit], 0) if outfit.sum() > 200 else None
    _r = max(3, int(frame(kp)[2] * 0.12)); nb = Image.fromarray((_last_new * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(_r)).point(lambda v: 255 if v > 8 else 0)
    R = (np.asarray(hard) > 127) & ~prot & ~(np.asarray(nb) > 127)  # margin: the drawn arm may sit a little off the bone line  # around the old arm, outside her outline: must stay background
    ring = (np.asarray(hard.filter(ImageFilter.MaxFilter(31))) > 127) & ~(np.asarray(hard) > 127) & ~prot
    bgc = np.median(a0[ring], 0) if ring.sum() > 50 else None
    keepbg = None
    if bgc is not None:
        kb = R & (np.sqrt(((a0 - bgc) ** 2).sum(-1)) < 45)
        keepbg = Image.fromarray((kb * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.5))
    Rsoft = Image.fromarray((R * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.5))
    def spill(r):
        """Share of the area outside her old outline (minus the new arm path) that now holds something other than background."""
        if bgc is None or R.sum() < 100: return 0.0
        b = np.asarray(r, np.float32)
        newstuff = (np.sqrt(((b - bgc) ** 2).sum(-1)) > 45) & (np.sqrt(((b - a0) ** 2).sum(-1)) > 30)
        return float(newstuff[R].mean())
    skin = None
    try:
        oa = _last_old & ~outfit
        if bgc is not None: oa &= np.sqrt(((a0 - bgc) ** 2).sum(-1)) > 45
        if oa.sum() > 200: skin = np.median(a0[oa], 0)
    except Exception: pass
    newarm = _last_thin
    def skin_ok(r):
        """The redrawn arms must be bare skin like hers (no black gloves / mesh sleeves)."""
        if skin is None or newarm.sum() < 200: return True
        b = np.asarray(r, np.float32)[newarm]
        return float((np.sqrt(((b - skin) ** 2).sum(-1)) < 70).mean()) > 0.3
    tone = "" if skin is None else ("dark brown" if skin.mean() < 100 else "brown" if skin.mean() < 150 else "fair")
    pimg = pose_image(kt, W, H)
    p, neg = prompt_neg
    p = POSES.get(pose, "") + f"bare arms, {tone} skin arms and hands, " + p
    neg += ", changed body shape, extra breast, deformed torso, hand on chest, hand on breast, touching body, extra arm, extra hand, torn fabric strip, gloves, black sleeves, fishnet, mesh, arm warmers, dark arms"
    if pose not in ("thumbs", "crossed", "hips", "onehip"): neg += ", fist"
    best = None; pre = {}
    for i in range(N * 2):
        if i == N and best is not None and best[0] >= 0.5: break  # second seed batch only if the first all failed
        if i % N == 0:  # Both machines: this batch in parallel, every other version on the desktop
            pre = {}
            try:
                import remote
                if remote.both():
                    if status: status(f"Drawing the new pose - versions {i + 1}-{i + N} on both machines in parallel...")
                    _i0 = i
                    rs = remote.run_split(lambda k: cudaeng.inpaint_pose(clean, hard, lineart, pimg, p, neg, seed=21 + (_i0 + k) * 13, progress=progress if remote.is_local() else None), N)
                    pre = {i + k: x for k, x in enumerate(rs) if x is not None}
            except Exception: logging.exception("both machines split")
        if status and i not in pre: status(f"Drawing the new pose - version {i + 1} of {N} (OpenPose + lineart ControlNet, NVIDIA)...")
        r = pre.pop(i) if i in pre else cudaeng.inpaint_pose(clean, hard, lineart, pimg, p, neg, seed=21 + i * 13, progress=progress)
        r = Image.composite(r, im, soft)
        if clean is not im: r = Image.composite(clean, r, Rsoft)  # vacated old-arm area off the new arm path = clean background
        elif keepbg is not None: r = Image.composite(im, r, keepbg)  # outside her outline and off the new arm path: background stays
        chest = hand_on_chest(r, body=outfit); ok = pose_ok(r, kt, pose) and skin_ok(r)
        sp = spill(r); m = pose_match(r, kt); sc = -1 if chest else (-0.5 if sp > (0.45 if clean is not im else 0.3) or not ok else m)  # with the clean plate the vacated area is already background
        logging.info("pose %s candidate %d match %.2f chest %s spill %.2f shape-ok %s", pose, i, m, chest, sp, ok)
        if os.environ.get("NPUSTUDIO_DEBUG"): r.save(f"pose_cand_{i}.jpg", quality=85)
        if best is None or sc > best[0]: best = (sc, r, m)
    sc, r, m = best
    if sc <= -1: return im, "NOT DONE - every version put a hand on her chest/body, so nothing was kept. Original kept; try again."
    if sc < 0: return im, "NOT DONE - every version changed her outline or got the arms wrong (pose checks). Original kept; try again."
    if m < 0.4: return im, f"NOT DONE - no version passed the checks and matched the pose (best match {m:.2f}). Original kept; try again."
    how = f"Method: pose redraw (OpenPose + lineart ControlNet, {src}, best of {N}, pose match {m:.2f})"
    if m < 0.5: how += " - weak pose match"
    return r, how
