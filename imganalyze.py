"""Analyze tab (phone + PC): plain-English description + a Gen-ready prompt for a picture, using the same vision model as
AI Change 'Analysis'. The prompt is sanitized: no body size/shape exaggeration words, no nudity terms."""
import re, time, logging, os
import numpy as np

TAG_Q = ("List comma-separated tags for an image-generation prompt describing this picture, in this order: subject; "
         "any animal or demi-human features (cat ears, fox ears, tail, horns, wings, elf ears) if present; hair colour and style; "
         "eye colour; outfit; pose; background; art style; lighting. Output only the tags, no sentences.")
NSFW = r"(?i)\b(nude|naked|nsfw|topless|bottomless|nipples?|areola\w*|genitals?|pussy|penis|vagina|sex\w*|erotic|lewd|hentai|ecchi|lingerie|underwear|panties|bra|cleavage|see[- ]?through|transparent clothes|sheer|undress\w*|exposed|bare (breasts?|chest|butt))\b"
EXAG = r"(?i)\b(seductive|sexy|sensual|alluring|provocative|busty|voluptuous|curvy|curvaceous|thicc|thick|chubby|plump|skinny|slim|slender|petite|huge|giant|large|big|small|tiny|massive|enormous|wide|narrow|long legs?|hourglass|breasts?|boobs?|bust|butt|ass|hips?|thighs?|waist)\b"

def sanitize(tags):
    import facefix
    out, seen = [], set()
    for t in re.split(r"[,\n;]+", tags or ""):
        t = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", t.strip()).strip(" .\"'")
        m_ = re.match(r"(?i)^(hair|eyes?|eye colou?r|hair colou?r|lighting|background)\s*:\s*(.+)$", t)
        if m_: k_ = m_.group(1).lower().split()[0]; t = m_.group(2) + " " + {"hair": "hair", "eye": "eyes", "eyes": "eyes", "lighting": "lighting", "background": "background"}.get(k_, "")
        t = re.sub(r"(?i)^(subject|features?|outfit( details)?|pose|art style|style)\s*:\s*", "", t).strip(" .")
        if re.fullmatch(r"(?i)(subject|features?|demi-human features|animal features|hair( colou?r)?( and style)?|eyes?( colou?r)?|outfit|pose|background|art style|style|lighting|colou?rs?)", t): continue
        if not t or len(t) > 60 or re.search(NSFW, t) or re.search(EXAG, t) or re.search(facefix.SIZE_WORDS, t): continue
        if re.search(r"(?i)\b(none|not present|no (animal|demi)|n/a)\b", t): continue
        k = t.lower()
        if k not in seen: seen.add(k); out.append(t)
    return ", ".join(out[:30])

def clean_desc(d):
    """Plain description, with nudity terms / body-size wording removed sentence by sentence."""
    s = [x for x in re.split(r"(?<=[.!?])\s+", (d or "").strip()) if x and not re.search(NSFW, x)]
    return " ".join(re.sub(r"(?i)\b(seductive|sexy|sensual|provocative|alluring)\b", "relaxed", re.sub(r"(?i)\b(busty|voluptuous|curvy|curvaceous|thicc)\b\s*", "", x)) for x in s)

def analyze(changer, im, device="Auto", status=None):
    t = time.time()
    changer.prefer(device)
    desc, dev, _ = changer.analyze(im, status)
    if status: status("Writing the Gen prompt...")
    with changer.lock:
        changer._load_vlm(status); raw = changer._ask(TAG_Q, im, 120)
    res = {"description": clean_desc(desc), "prompt": sanitize(raw), "width": im.size[0], "height": im.size[1], "device": dev}
    if status: status("Checking the pose (skeleton)...")
    try: apply_pose(res, im)
    except Exception: logging.exception("pose check"); res["pose"] = "pose uncertain (pose check failed)"
    res["seconds"] = round(time.time() - t, 1); return res

def save_txt(res, tag=""):
    import outputs
    d = os.path.join(outputs.folder(), "analyze", time.strftime("%Y-%m-%d")); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, time.strftime("%H%M%S") + ("_" + re.sub(r"[^A-Za-z0-9]+", "_", tag)[:30] if tag else "") + ".txt")
    open(p, "w", encoding="utf-8").write(f"Size: {res['width']}x{res['height']}\nPose: {res.get('pose', '-')}\n\nDescription:\n{res['description']}\n\nPrompt:\n{res['prompt']}\n")
    return p

# ---------- pose check: OpenPose keypoints -> plain pose phrases (overrides the captioner's pose words when confident) ----------
POSE_WORDS = (r"(?i)\b(hands? (?:on|at|near|behind|by|in|resting|placed|raised|up|down|crossed|together|clasped|folded|extended|outstretched)\w*|"
              r"arms? (?:\w+ )?(?:crossed|folded|raised|up|down|extended|outstretched|out|at|by|behind|akimbo|hanging|spread)\w*|akimbo|"
              r"(?:one|other|left|right) (?:hand|arm)\b[^,.;]*|the other (?:holding|raised|resting)[^,.;]*|on (?:her|his|their) hips?|hand on hip|reaching\w*|waving|pointing|"
              r"sitting|seated|kneeling|knelt|crouch\w*|squat\w*|standing|cross-legged|legs? crossed|crossed legs|lying|reclin\w*|"
              r"knees? bent|bent knees|pose[ds]?|posing)\b")
_KP = dict(nose=0, neck=1, rsh=2, rel=3, rwr=4, lsh=5, lel=6, lwr=7, rhip=8, rkn=9, rank=10, lhip=11, lkn=12, lank=13)

def keypoints(im):
    """-> ({name: (x, y, score)}, mean score) from OpenPose (posegen's detector), or ({}, 0) if it fails."""
    try:
        import posegen
        W, H = im.size; ps = posegen.detector().detect_poses(np.asarray(im.convert("RGB")))
        if not ps: return {}, 0.0
        best = max(ps, key=lambda p: sum(k is not None for k in p.body.keypoints))
        kp = {n: (k.x * W, k.y * H, float(getattr(k, "score", 1.0) or 1.0)) for n, i in _KP.items() for k in [best.body.keypoints[i]] if k is not None}
        return kp, (float(np.mean([v[2] for v in kp.values()])) if kp else 0.0)
    except Exception: logging.exception("pose keypoints"); return {}, 0.0

def pose_phrases(im):
    """-> (list of phrases, confidence 0..1, debug). Person's right arm = image left when facing the viewer."""
    kp, sc = keypoints(im)
    g = lambda n: np.array(kp[n][:2]) if n in kp else None
    d = lambda a, b: float(np.linalg.norm(a - b))
    sh = [g("rsh"), g("lsh")]; hips = [g("rhip"), g("lhip")]
    if sh[0] is None or sh[1] is None: return [], 0.0, {"kp": len(kp)}
    msh = (sh[0] + sh[1]) / 2; swid = d(sh[0], sh[1])
    mh = (hips[0] + hips[1]) / 2 if hips[0] is not None and hips[1] is not None else (hips[0] if hips[0] is not None else hips[1])
    T = d(msh, mh) if mh is not None else swid * 1.6; T = max(T, swid * 0.8, 1.0)
    cx = msh[0]; arms = []; nose = g("nose")
    for s, e, w, h in (("rsh", "rel", "rwr", "rhip"), ("lsh", "lel", "lwr", "lhip")):
        S, E, Wr, Hp = g(s), g(e), g(w), g(h)
        if Wr is None or S is None: arms.append(None); continue
        if Hp is None and mh is not None: Hp = mh + (S - msh)
        out_ = E is not None and abs(E[0] - cx) > abs(S[0] - cx) + 0.12 * T
        if Wr[1] < S[1] - 0.15 * T or (E is not None and Wr[1] < S[1] + 0.15 * T and E[1] - Wr[1] > 0.25 * T): arms.append("raised")
        elif nose is not None and d(Wr, nose) < 0.4 * T: arms.append("face")
        elif Hp is not None and d(Wr, Hp) < 0.35 * T and out_ and E is not None and E[1] < Wr[1]: arms.append("hip")
        elif (Wr[0] - cx) * (S[0] - cx) < 0 and abs(Wr[1] - (S[1] + 0.4 * T)) < 0.45 * T: arms.append("across")
        elif abs(Wr[0] - S[0]) > abs(Wr[1] - S[1]) and abs(Wr[0] - S[0]) > 0.55 * T: arms.append("extended")
        else: arms.append("side")
    ph = []; a = [x for x in arms if x]
    if a.count("across") == 2: ph.append("arms crossed")
    elif len(a) == 2 and a[0] == a[1]: ph.append({"raised": "both arms raised", "face": "both hands near her face", "hip": "hands on hips", "extended": "arms outstretched", "side": "arms at her sides", "across": "arms folded"}[a[0]])
    else:
        nm = {"raised": "one arm raised", "face": "one hand near her face", "hip": "one hand on her hip", "extended": "one arm extended", "side": "one arm at her side", "across": "one arm across her body"}
        ph += [nm[x] for x in a]
    # legs
    K = [g("rkn"), g("lkn")]; A = [g("rank"), g("lank")]; leg = None
    if mh is not None and any(k is not None for k in K):
        ks = [k for k in K if k is not None]; ky = np.mean([k[1] for k in ks]); an = [x for x in A if x is not None]
        if an and abs(np.mean([x[1] for x in an]) - ky) < 0.3 * T and ky - mh[1] < 0.9 * T: leg = "kneeling"
        elif ky - mh[1] < 0.45 * T: leg = "sitting"
        elif ky - mh[1] > 0.6 * T and (not an or np.mean([x[1] for x in an]) - ky > 0.5 * T): leg = "standing"
        if len(an) == 2 and hips[0] is not None and hips[1] is not None and (an[0][0] - an[1][0]) * (hips[0][0] - hips[1][0]) < 0 and leg != "kneeling": ph.append("legs crossed")
    if leg: ph.insert(0, leg)
    need = ["rsh", "lsh", "rwr", "lwr", "rel", "lel"] + (["rhip", "lhip", "rkn", "lkn"])
    conf = round(sc * sum(n in kp for n in need) / len(need), 2)
    return ph, conf, {"arms": arms, "leg": leg, "kp": len(kp), "score": round(sc, 2)}

def apply_pose(res, im, th=0.45):
    ph, conf, dbg = pose_phrases(im); res["pose_debug"] = dbg
    if not ph: res["pose"] = "pose uncertain (no skeleton found)"; return res
    if conf < th: res["pose"] = f"{', '.join(ph)} (pose uncertain, {conf:.2f})"; return res
    res["pose"] = f"{', '.join(ph)} (confidence {conf:.2f})"
    sents = [x for x in re.split(r"(?<=[.!?])\s+", res.get("description", "")) if x]
    keep = []
    for x in sents:
        if re.search(POSE_WORDS, x):
            y = re.sub(r",?\s*(with|and|while)?\s*" + POSE_WORDS.replace("(?i)", "") + r"[^,.;]*", "", x, flags=re.I).strip(" ,")
            if len(y.split()) >= 4 and not re.search(POSE_WORDS, y): keep.append(y if y.endswith(".") else y + ".")
        else: keep.append(x)
    keep.append(("She is " if leg_or(ph) else "Pose: ") + ", ".join(ph) + ".")
    res["description"] = " ".join(keep)
    tags = [t.strip() for t in res.get("prompt", "").split(",") if t.strip() and not re.search(POSE_WORDS, t)]
    res["prompt"] = ", ".join(tags[:1] + ph + tags[1:])
    return res

def leg_or(ph): return bool(ph) and ph[0] in ("standing", "sitting", "kneeling")
