"""Analyze tab (phone + PC): plain-English description + a Gen-ready prompt for a picture, using the same vision model as
AI Change 'Analysis'. The prompt is sanitized: no body size/shape exaggeration words, no nudity terms."""
import re, time, logging, os

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
    return {"description": clean_desc(desc), "prompt": sanitize(raw), "width": im.size[0], "height": im.size[1],
            "device": dev, "seconds": round(time.time() - t, 1)}

def save_txt(res, tag=""):
    import outputs
    d = os.path.join(outputs.folder(), "analyze", time.strftime("%Y-%m-%d")); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, time.strftime("%H%M%S") + ("_" + re.sub(r"[^A-Za-z0-9]+", "_", tag)[:30] if tag else "") + ".txt")
    open(p, "w", encoding="utf-8").write(f"Size: {res['width']}x{res['height']}\n\nDescription:\n{res['description']}\n\nPrompt:\n{res['prompt']}\n")
    return p
